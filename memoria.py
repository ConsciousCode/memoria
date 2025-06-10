from datetime import datetime
from typing import Iterable, Optional

from pydantic import TypeAdapter

from ipld import CIDv1

from db import Database, MemoryRow
from models import ACThread, DAGEdge, Edge, Memory, MemoryDataAdapter, MemoryDAG, RecallConfig, build_memory, model_dump
from util import todo_list

class Memoria:
    '''
    Wraps all memoria-related functionality to abstract away the details
    of the underlying database, but doesn't implement the MCP server.
    '''

    def __init__(self, db: Database):
        super().__init__()
        self.db = db
    
    def lookup_memory(self, cid: CIDv1) -> Optional[Memory]:
        return self.db.lookup_ipld_memory(cid)

    def insert(self,
            memory: Memory,
            sona: Optional[str] = None,
            index: Optional[str] = None,
            importance: Optional[float] = None
        ) -> tuple[int, CIDv1]:
        '''
        Append a memory to the sona file.
        '''

        rowid = self.db.insert_memory(
            memory.cid,
            memory.kind,
            model_dump(memory.data),
            memory.timestamp,
            importance
        )
        
        self.db.link_memory_edges(rowid, memory.edges or {})
        
        if index:
            self.db.insert_text_embedding(rowid, index)
            self.db.insert_text_fts(rowid, index)
        
        if sona:
            sona_row = self.db.find_sona(sona)
            self.db.link_sona(sona_row.rowid, rowid)
        
        self.db.commit()
        return rowid, memory.cid
    
    def find_sona(self, sona: str):
        return self.db.find_sona(sona)
    
    def act_push(self,
            sona: str,
            prompt: Memory|str,
            index: Optional[str]=None,
            include: Optional[dict[str, list[Edge]]]=None,
            importance: Optional[float]=None
        ) -> Optional[int]:
        '''Push a prompt to the sona for processing.'''
        # Find or create the sona
        sona_row = self.db.find_sona(sona)

        if isinstance(prompt, str):
            if row := self.db.select_memory_cid(CIDv1(prompt)):
                prompt_cid = CIDv1(row.cid)
                prompt_id = row.rowid
            else:
                return None
        else:
            prompt_cid = prompt.cid
            # Insert the prompt
            prompt_id = self.db.insert_memory(
                prompt.cid,
                prompt.kind,
                model_dump(prompt.data),
                prompt.timestamp,
                importance
            )
            if index:
                self.db.index(prompt_id, index)
        
        # Figure out where it's going
        
        if pending_thread := self.db.get_act_pending(sona_row.rowid):
            # Pending thread already exists, add to its context
            response_id = pending_thread.memory_id
            pending_id = pending_thread.rowid
        else:
            # No pending thread, we need to create one
            active_thread = (
                self.db.get_act_active(sona_row.rowid) or
                self.db.get_last_act(sona_row.rowid)
            )

            # Create the incomplete memory to receive the response
            response_id = self.db.insert_memory(
                cid=None,
                kind="self",
                data=None,
                timestamp=datetime.now().timestamp(),
                importance=importance
            )

            # Create a new pending thread
            pending_id = self.db.insert_act(
                cid=None,
                sona_id=sona_row.rowid,
                memory_id=prompt_id,
                prev_id=active_thread and active_thread.rowid
            )
        
        if include is None:
            include = {}
        
        # Now that we have the pending thread, link it to the prompt
        prompt_edges = include.get("prompt", [])
        prompt_edges.append(Edge(
            target=prompt_cid,
            weight=1.0
        ))

        self.db.link_memory_edges(response_id, {
            **include,
            "prompt": prompt_edges
        })

        return pending_id
    
    def act_stream(self,
            sona: str,
            prompt: Memory|str,
            index: Optional[str]=None,
            include: Optional[dict[str, list[Edge]]]=None,
            importance: Optional[float]=None
        ) -> ACThread:
        '''
        Stream a prompt to the sona for processing.
        This will create a new thread and return it.
        '''
        pending_id = self.act_push(
            sona=sona,
            prompt=prompt,
            index=index,
            include=include,
            importance=importance
        )

        if pending_id is None:
            raise ValueError("Failed to push prompt to sona")

        return ACThread(
            sona=sona,
            memory=CIDv1(prompt.cid if isinstance(prompt, Memory) else prompt),
            prev=None
    )

    def recall(self,
            sona: Optional[str],
            prompt: str,
            timestamp: Optional[float]=None,
            config: Optional[RecallConfig]=None,
            include: Optional[Iterable[CIDv1]]=None
        ) -> MemoryDAG:
        '''
        Recall memories based on a prompt. This incorporates all indices
        and returns a topological sort of relevant memories.
        '''
        g = MemoryDAG()
        
        for c in include or []:
            if pm := self.db.select_memory_cid(c):
                g.insert(CIDv1(pm.cid), build_memory(
                    pm.kind,
                    MemoryDataAdapter.validate_json(pm.data),
                    pm.timestamp
                ))
        
        rows: list[tuple[MemoryRow, float]] = []
        
        memories = self.db.recall(sona, prompt, timestamp, config)
        # Populate the graph with nodes so we can detect when there are edges
        #  between our seletions
        for rs in memories:
            row, _ = rs
            rows.append(rs)
            g.insert(CIDv1(row.cid), build_memory(
                kind=row.kind,
                data=row.data,
                timestamp=row.timestamp
            ))

        # Populate backward and forward edges
        bw: list[tuple[float, int, CIDv1]] = []
        fw: list[tuple[float, int, CIDv1]] = []
        
        for rs in rows:
            row, score = rs
            if score <= 0:
                break

            rowid = row.rowid
            origcid = CIDv1(row.cid)

            budget = score*20

            b = 0
            for dst, label, weight in self.db.backward_edges(rowid):
                dstcid = CIDv1(dst.cid)
                if dstcid in g:
                    if not g.has_edge(origcid, dstcid):
                        g.add_edge(origcid, dstcid, DAGEdge(
                            label=label,
                            weight=weight
                        ))
                    continue

                b += weight
                if b >= budget:
                    break
                
                bw.append((budget*weight, dst.rowid, CIDv1(dst.cid)))
            
            b = 0
            for label, weight, src in self.db.forward_edges(rowid):
                dstcid = CIDv1(src.cid)
                if dstcid in g:
                    if not g.has_edge(dstcid, origcid):
                        g.add_edge(dstcid, origcid, DAGEdge(
                            label=label,
                            weight=weight
                        ))
                    continue

                if not src.importance:
                    break
                
                b += src.importance
                if b >= budget:
                    break
                
                fw.append((budget*src.importance, src.rowid, CIDv1(src.cid)))
        
        # Note: These feel so similar, maybe there's a way to reduce boilerplate?

        # Search backwards for supporting memories using their edge weight
        #  to determine how relevant they are to the current memory
        for budget, src_id, srccid in todo_list(bw):
            b = 0
            for dst, label, weight in self.db.backward_edges(src_id):
                b += weight
                if b >= budget:
                    break
                
                dstcid = CIDv1(dst.cid)

                g.insert(srccid, build_memory(
                    kind=dst.kind,
                    data=dst.data,
                    timestamp=dst.timestamp
                ))
                g.add_edge(srccid, dstcid, DAGEdge(
                    label=label,
                    weight=weight
                ))

                bw.append((budget*weight, dst.rowid, dstcid))
        
        # Search forwards for response memories using their annotated
        #  importance - important conclusions are more relevant
        for budget, dst_id, dstcid in todo_list(fw):
            b = 0
            for label, weight, src in self.db.forward_edges(dst_id):
                b += (imp := src.importance or 0)
                if b >= budget:
                    break

                srccid = CIDv1(src.cid)

                g.insert(srccid, build_memory(
                    kind=src.kind,
                    data=src.data,
                    timestamp=src.timestamp
                ))
                g.add_edge(srccid, dstcid, DAGEdge(
                    label=label,
                    weight=weight
                ))

                fw.append((budget*imp, dst_id, dstcid))
        
        return g