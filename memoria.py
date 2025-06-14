from collections import defaultdict
from datetime import datetime
from typing import Iterable, Optional
from uuid import UUID

from ipld.cid import CIDv1

from db import Database, MemoryRow
from models import DAGEdge, Edge, Memory, MemoryDataAdapter, MemoryDAG, RecallConfig, SelfMemory, StopReason, build_memory, memory_document, model_dump
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
        ) -> CIDv1:
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
        return memory.cid
    
    def find_sona(self, sona: str):
        return self.db.find_sona(sona)
    
    def build_subgraph(self, edges: dict[str, list[Edge]], budget: float=20) -> MemoryDAG:
        '''
        Build a subgraph of memories.
        
        This is used to build a subgraph of memories that are related to the
        initial memories. It will return a MemoryDAG containing the memories
        and their edges.
        '''
        g = MemoryDAG()
        
        # Populate initial backward and forward edges
        bw: list[tuple[float, int, CIDv1]] = [] # [(score, rowid, cid)]
        fw: list[tuple[float, int, CIDv1]] = []
        
        for label, es in edges.items():
            for e in es:
                score = e.weight
                print("Score", score)
                if score <= 0:
                    break
                
                origcid = e.target
                if (mr := self.db.select_memory_cid(origcid)) is None:
                    continue

                g.insert(origcid, mr.to_memory())

                energy = score*budget

                b = 0
                for edge in self.db.backward_edges(mr.rowid):
                    dst, label, weight = edge.dst, edge.label, edge.weight
                    assert dst.cid # Incomplete memories must not be referenced
                    dstcid = CIDv1(dst.cid)
                    if dstcid in g:
                        if not g.has_edge(origcid, dstcid):
                            g.add_edge(origcid, dstcid, DAGEdge(
                                label=label,
                                weight=weight
                            ))
                        continue

                    b += weight
                    if b >= energy:
                        break
                    
                    bw.append((energy*weight, dst.rowid, CIDv1(dst.cid)))
                
                b = 0
                for edge in self.db.forward_edges(mr.rowid):
                    label, weight, src = edge.label, edge.weight, edge.src
                    if not src.cid:
                        continue
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
                    if b >= energy:
                        break
                    
                    fw.append((energy*src.importance, src.rowid, CIDv1(src.cid)))
        
        # Note: These feel so similar, maybe there's a way to reduce boilerplate?

        # Search backwards for supporting memories using their edge weight
        #  to determine how relevant they are to the current memory
        for energy, src_id, srccid in todo_list(bw):
            b = 0
            for edge in self.db.backward_edges(src_id):
                dst, label, weight = edge.dst, edge.label, edge.weight
                b += weight
                if b >= energy:
                    break
                
                assert dst.cid
                dstcid = CIDv1(dst.cid)

                g.insert(srccid, dst.to_memory())
                g.add_edge(srccid, dstcid, DAGEdge(
                    label=label,
                    weight=weight
                ))

                bw.append((energy*weight, dst.rowid, dstcid))
        
        # Search forwards for response memories using their annotated
        #  importance - important conclusions are more relevant.
        # These iterate over *edges* which is why we needed to populate them
        #  in the first place
        for energy, dst_id, dstcid in todo_list(fw):
            b = 0
            for edge in self.db.forward_edges(dst_id):
                label, weight, src = edge.label, edge.weight, edge.src
                b += (imp := src.importance or 0)
                if b >= energy:
                    break

                assert src.cid
                srccid = CIDv1(src.cid)

                g.insert(srccid, src.to_memory())
                g.add_edge(srccid, dstcid, DAGEdge(
                    label=label,
                    weight=weight
                ))

                fw.append((energy*imp, dst_id, dstcid))
        print(g)
        return g

    def act_push(self,
            sona: str,
            prompts: dict[str, list[Edge]] # label: [edge]
        ) -> Optional[UUID]:
        '''
        Push prompts to the sona for processing. Return the receiving
        sona's UUID.
        '''
        try:
            # Find or create the sona
            sona_row = self.db.find_sona(sona)
            
            # Figure out where it's going
            
            if pending_thread := self.db.get_act_pending(sona_row.rowid):
                # Pending thread already exists, add to its context
                response_id = pending_thread.memory_id
            else:
                # No pending thread, we need to create one
                prev_thread = ( # Previous thread to link to
                    self.db.get_act_active(sona_row.rowid) or
                    self.db.get_last_act(sona_row.rowid)
                )

                # Create the incomplete memory to receive the response
                response_id = self.db.insert_memory(
                    cid=None,
                    kind="self",
                    data=None,
                    timestamp=datetime.now().timestamp()
                )

                # Create a new pending thread
                self.db.update_sona_pending(sona_row.rowid,
                    self.db.insert_act(
                        cid=None,
                        sona_id=sona_row.rowid,
                        memory_id=response_id,
                        prev_id=prev_thread and prev_thread.rowid
                    )
                )
            
            self.db.link_memory_edges(response_id, prompts)

            return UUID(bytes=sona_row.uuid)
        except Exception:
            import traceback
            traceback.print_exc()
            raise
    
    def act_next(self,
            sona: str,
            timestamp: Optional[float]=None,
            config: Optional[RecallConfig]=None
        ) -> Optional[MemoryDAG]:
        '''
        Get the next pending thread for the sona.
        
        Returns the rowid of the pending thread or None if there is no
        pending thread.
        '''
        # Find or stage the active thread
        sona_row = self.db.find_sona(sona)
        if act := self.db.get_act_active(sona_row.rowid):
            memory_id = act.memory_id
        else: # No active thread, check for pending
            if (memory_id := sona_row.pending_id) is None:
                return None # No threads at all
            self.db.sona_stage_active(sona_row.rowid)
        
        # We used the incomplete memory's edges to store prompts, only now
        #  do we actually run recall on them.
        edges: dict[str, list[Edge]] = defaultdict(list)
        for e in self.db.backward_edges(memory_id):
            prompt = memory_document(e.dst.to_memory())
            for row, score in self.db.recall(sona, prompt, timestamp, config):
                if cid := row.cid:
                    edges[e.label].append(Edge(
                        target=CIDv1(cid),
                        weight=e.weight*score
                    ))
        
        return self.build_subgraph(edges)

    def act_stream(self,
            sona: str,
            delta: Optional[str],
            model: Optional[str]=None,
            stop_reason: Optional[StopReason] = None
        ) -> Optional[str]:
        '''
        Stream a delta to the active thread of the sona.
        
        Returns a failure reason.
        '''
        
        if (sona_row := self.db.find_sona(sona)) is None:
            return "sona not found"
        
        if (thread := self.db.get_act_active(sona_row.rowid)) is None:
            # No active thread, check for pending
            if (thread := self.db.get_act_pending(sona_row.rowid)) is None:
                return "no active or pending thread"
            
            # Move pending to active as long as we're not ending immediately
            if stop_reason is None:
                self.db.sona_stage_active(sona_row.rowid)
        
        # Update the memory data
        if (memory := self.db.select_memory_rowid(thread.memory_id)) is None:
            return "thread memory not found"
        
        if memory.kind != "self":
            return "thread memory is not a self memory"
        
        data = SelfMemory.Data.model_validate_json(memory.data)
        last = data.parts[-1] if data.parts else None
        if last and (model is None or last.model == model):
            last.content += delta or ""
        else:
            data.parts.append(SelfMemory.Data.Part(
                content=delta or "",
                model=model
            ))
        
        # Commit the updates
        if stop_reason:
            data.stop_reason = stop_reason
            self.db.update_sona_active(sona_row.rowid, None)
            self.db.finalize_memory(memory.rowid)
            self.db.finalize_act(thread.rowid)
        
        self.db.update_memory_data(memory.rowid, data.model_dump_json())

    def recall(self,
            sona: Optional[str],
            prompt: str,
            timestamp: Optional[float]=None,
            config: Optional[RecallConfig]=None,
            include: Optional[dict[str, list[Edge]]]=None
        ) -> MemoryDAG:
        '''Recall memories based on a prompt as a memory subgraph.'''
        try:
            edges: dict[str, list[Edge]] = defaultdict(list, {
                label: es.copy()
                    for label, es in (include or {}).items()
            })
            for row, score in self.db.recall(sona, prompt, timestamp, config):
                if cid := row.cid:
                    edges[""].append(Edge(
                        weight=score,
                        target=CIDv1(cid)
                    ))
            print(edges)
            return self.build_subgraph(edges)
        except Exception:
            import traceback
            traceback.print_exc()
            raise