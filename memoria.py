from datetime import datetime
from typing import Optional, overload
from uuid import UUID

from ipld.cid import CIDv1

from db import Database
from models import ACThread, AnyMemory, Edge, IncompleteMemory, Memory, MemoryDAG, RecallConfig, Sona, StopReason
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
    
    def lookup_act(self, cid: CIDv1) -> Optional[ACThread]:
        return self.db.lookup_ipld_act(cid)

    def insert(self,
            memory: AnyMemory,
            importance: Optional[float] = None,
            sona: Optional[UUID|str] = None,
            index: Optional[str] = None
        ):
        '''Append a memory to the sona file.'''
        with self.db.transaction() as db:
            rowid = db.insert_memory(memory, importance)
            db.link_memory_edges(rowid, memory.edges or [])
            
            if index:
                db.insert_text_embedding(rowid, index)
                db.insert_text_fts(rowid, index)
            
            if sona:
                if sona_row := db.find_sona(sona):
                    db.link_sona(sona_row.rowid, rowid)
    
    @overload
    def find_sona(self, sona: UUID) -> Optional[Sona]: ...
    @overload
    def find_sona(self, sona: str) -> Sona: ...

    def find_sona(self, sona: UUID|str) -> Optional[Sona]:
        with self.db.transaction() as db:
            if row := db.find_sona(sona):
                return Sona(
                    uuid=UUID(bytes=row.uuid),
                    aliases=self.db.select_sona_aliases(row.rowid),
                    pending=self.db.get_incomplete_act(row.pending_id)
                        if row.pending_id else None,
                    active=self.db.get_incomplete_act(row.active_id)
                        if row.active_id else None
                )
            return None
    
    def build_subgraph(self, edges: list[Edge[CIDv1]], budget: float=20) -> MemoryDAG:
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
        
        for e in edges:
            score = e.weight
            if score <= 0:
                break
            
            origcid = e.target
            if (mr := self.db.select_memory_cid(origcid)) is None:
                continue

            g.insert(origcid, mr.to_partial())

            energy = score*budget

            b = 0
            for edge in self.db.backward_edges(mr.rowid):
                dst, weight = edge.target, edge.weight
                dstcid = CIDv1(dst.cid)
                if dstcid in g:
                    if not g.has_edge(origcid, dstcid):
                        g.add_edge(origcid, dstcid, weight)
                    continue

                b += weight
                if b >= energy:
                    break
                
                bw.append((energy*weight, dst.rowid, CIDv1(dst.cid)))
            
            b = 0
            for edge in self.db.forward_edges(mr.rowid):
                weight, src = edge.weight, edge.target
                if not src.cid:
                    continue
                dstcid = CIDv1(src.cid)
                if dstcid in g:
                    if not g.has_edge(dstcid, origcid):
                        g.add_edge(dstcid, origcid, weight)
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
                dst, weight = edge.target, edge.weight
                b += weight
                if b >= energy:
                    break
                
                dstcid = CIDv1(dst.cid)

                g.insert(srccid, dst.to_partial())
                g.add_edge(srccid, dstcid, weight)

                bw.append((energy*weight, dst.rowid, dstcid))
        
        # Search forwards for response memories using their annotated
        #  importance - important conclusions are more relevant.
        # These iterate over *edges* which is why we needed to populate them
        #  in the first place
        for energy, dst_id, dstcid in todo_list(fw):
            b = 0
            for edge in self.db.forward_edges(dst_id):
                weight, src = edge.weight, edge.target
                # Skip incomplete memories in forward edge recall
                if src.cid is None:
                    continue
                
                b += (imp := src.importance or 0)
                if b >= energy:
                    break

                srccid = CIDv1(src.cid)

                g.insert(srccid, src.to_partial())
                g.add_edge(srccid, dstcid, weight)

                fw.append((energy*imp, dst_id, dstcid))
        return g

    def act_push(self,
            sona: UUID|str,
            prompts: list[Edge[CIDv1]]
        ) -> Optional[UUID]:
        '''
        Push prompts to the sona for processing. Return the receiving
        sona's UUID.
        '''
        with self.db.transaction() as db:
            # Find or create the sona
            if (sona_row := db.find_sona(sona)) is None:
                # No such sona (UUID)
                return None
            
            # Figure out where it's going
            
            if pending_thread := db.get_act_pending(sona_row.rowid):
                # Pending thread already exists, add to its context
                response_id = pending_thread.memory_id
            else:
                # No pending thread, we need to create one
                prev_thread = ( # Previous thread to link to
                    db.get_act_active(sona_row.rowid) or
                    db.get_last_act(sona_row.rowid)
                )

                # Create the incomplete memory to receive the response
                response_id = db.insert_memory(IncompleteMemory(
                    data=IncompleteMemory.SelfData(
                        parts=[],
                    ),
                    timestamp=datetime.now().timestamp(),
                    edges=prompts,
                ))

                # Create a new pending thread
                db.update_sona_pending(sona_row.rowid,
                    db.insert_act(
                        cid=None,
                        sona_id=sona_row.rowid,
                        memory_id=response_id,
                        prev_id=prev_thread and prev_thread.rowid
                    )
                )
            
            db.link_memory_edges(response_id, prompts)

            return UUID(bytes=sona_row.uuid)
    
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
        with self.db.transaction() as db:
            # Find or stage the active thread
            if (sona_row := db.find_sona(sona)) is None:
                return None
            
            if act := db.get_act_active(sona_row.rowid):
                memory_id = act.memory_id
            else: # No active thread, check for pending
                if (memory_id := sona_row.pending_id) is None:
                    return None # No threads at all
                db.sona_stage_active(sona_row.rowid)
            
            # We used the incomplete memory's edges to store prompts, only now
            #  do we actually run recall on them.
            edges: list[Edge[CIDv1]] = []
            for e in db.backward_edges(memory_id):
                prompt = e.target.to_incomplete().data.document()
                for row, score in db.recall(sona, prompt, timestamp, config):
                    if cid := row.cid:
                        edges.append(Edge(
                            target=CIDv1(cid),
                            weight=e.weight*score
                        ))
            
            return self.build_subgraph(edges)

    def act_stream(self,
            sona: UUID|str,
            delta: Optional[str],
            model: Optional[str]=None,
            stop_reason: Optional[StopReason] = None
        ) -> Optional[str]:
        '''
        Stream a delta to the active thread of the sona.
        
        Returns a failure reason.
        '''
        with self.db.transaction() as db:
            if (sona_row := db.find_sona(sona)) is None:
                return "sona not found"
            
            if (thread := db.get_act_active(sona_row.rowid)) is None:
                # No active thread, check for pending
                if (thread := db.get_act_pending(sona_row.rowid)) is None:
                    return "no active or pending thread"
                
                # Move pending to active as long as we're not ending immediately
                if stop_reason is None:
                    db.sona_stage_active(sona_row.rowid)
            
            # Update the memory data
            if (mr := db.select_memory_rowid(thread.memory_id)) is None:
                return "thread memory not found"
            
            if mr.kind != "self":
                return "thread memory is not a self memory"
            
            data = Memory.build_data('self', mr.data)
            last = data.parts[-1] if data.parts else None
            if last and (model is None or last.model == model):
                last.content += delta or ""
            else:
                data.parts.append(Memory.SelfData.Part(
                    content=delta or "",
                    model=model
                ))
            
            # Commit the updates
            if stop_reason:
                data.stop_reason = stop_reason
                db.update_sona_active(sona_row.rowid, None)
                db.finalize_memory(mr.rowid)
                db.finalize_act(thread.rowid)
            
            db.update_memory_data(mr.rowid, data.model_dump_json())

    def recall(self,
            sona: Optional[str],
            prompt: Optional[str],
            timestamp: Optional[float]=None,
            config: Optional[RecallConfig]=None,
            include: Optional[list[CIDv1]]=None
        ) -> MemoryDAG:
        '''Recall memories based on a prompt as a memory subgraph.'''
        edges: list[Edge[CIDv1]] = []
        for row, score in self.db.recall(sona, prompt, timestamp, config):
            edges.append(Edge(
                weight=score,
                target=CIDv1(row.cid)
            ))
        return self.build_subgraph(edges)