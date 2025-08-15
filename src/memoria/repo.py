'''
Memoria is the immutable state which can't be advanced without external
intervention.
'''

from datetime import datetime
from typing import Iterable, Optional, overload, override
from uuid import UUID

from cid import CIDv1, CID
from ipfs import Blocksource

from .db import DatabaseRO, FileRow
from .memory import ACThread, AnyMemory, Edge, Memory, MemoryContext, MemoryDAG, MemoryDataAdapter, SelfData, Sona, StopReason
from .config import RecallConfig
from .util import todo_list

__all__ = (
    'Repository',
)

class Repository(Blocksource):
    '''
    Wraps all memoria-related functionality to abstract away the details
    of the underlying database, but doesn't implement the MCP server.
    '''

    def __init__(self, db: DatabaseRO):
        super().__init__()
        self.db = db
    
    @override
    def block_has(self, cid: CID) -> bool:
        return isinstance(cid, CIDv1) and self.db.has_cid(cid)

    @override
    def block_get(self, cid: CID) -> Optional[bytes]:
        if not isinstance(cid, CIDv1):
            return None
        data = (
            self.lookup_memory(cid) or
            self.lookup_act(cid)
        )
        return data and data.ipld_block()
    
    def register_file(self,
            cid: CID,
            filename: str,
            mimetype: str,
            filesize: int,
            overhead: int
        ):
        '''
        Register a file in the database.
        
        This is used to register files that are uploaded to the system.
        '''
        with self.db.transaction() as db:
            db.register_file(cid, filename, mimetype, filesize, overhead)

    def lookup_memory(self, cid: CID) -> Optional[Memory]:
        if isinstance(cid, CIDv1):
            return self.db.select_memory_ipld(cid=cid)
    
    def lookup_act(self, cid: CID) -> Optional[ACThread]:
        if isinstance(cid, CIDv1):
            return self.db.select_act_ipld(cid=cid)

    def lookup_file(self, cid: CID) -> Optional[FileRow]:
        return self.db.select_file(cid=cid)

    def insert(self,
            memory: Memory,
            index: Optional[list[str]] = None,
            timestamp: Optional[int] = None
        ):
        '''Append a memory to the sona file.'''
        with self.db.transaction() as db:
            db.insert_memory(memory, index, timestamp)

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
            if (mr := self.db.select_memory(cid=origcid)) is None:
                continue

            g.insert(origcid, MemoryContext(
                memory=mr.to_partial(),
                timestamp=mr.timestamp
            ))

            energy = score*budget

            b = 0
            for edge in self.db.dependencies(rowid=mr.rowid):
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
            
            # TODO: Forward recall previously depended on importance
            raise NotImplementedError
            b = 0
            for edge in self.db.references(rowid=mr.rowid):
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
            for edge in self.db.dependencies(rowid=src_id):
                dst, weight = edge.target, edge.weight
                b += weight
                if b >= energy:
                    break
                
                dstcid = CIDv1(dst.cid)

                g.insert(srccid, MemoryContext(
                    memory=dst.to_partial(),
                    timestamp=dst.timestamp
                ))
                g.add_edge(srccid, dstcid, weight)

                bw.append((energy*weight, dst.rowid, dstcid))
        
        # Search forwards for response memories using their annotated
        #  importance - important conclusions are more relevant.
        # These iterate over *edges* which is why we needed to populate them
        #  in the first place
        for energy, dst_id, dstcid in todo_list(fw):
            b = 0
            for edge in self.db.references(rowid=dst_id):
                weight, src = edge.weight, edge.target
                # Skip incomplete memories in forward edge recall
                if src.cid is None:
                    continue
                
                if b >= energy:
                    break

                srccid = CIDv1(src.cid)

                g.insert(srccid, MemoryContext(
                    memory=src.to_partial()
                ))
                g.add_edge(srccid, dstcid, weight)

                # TODO: Again forward recall previously depended on importance
                raise NotImplementedError
                fw.append((energy*imp, dst_id, dstcid))
        return g

    def act_push(self,
            sona: UUID|str,
            include: list[Edge[CIDv1]]
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
                response_id = db.insert_memory(
                    Memory(
                        data=SelfData(parts=[]),
                        edges=include
                    ),
                    timestamp=int(datetime.now().timestamp())
                )

                # Create a new pending thread
                db.update_sona_pending(sona_row.rowid,
                    db.insert_act(
                        cid=None,
                        sona_id=sona_row.rowid,
                        memory_id=response_id,
                        prev_id=prev_thread and prev_thread.rowid
                    )
                )
            
            db.link_memory_edges(response_id, include)

            return UUID(bytes=sona_row.uuid)
    
    def act_next(self,
            sona: UUID|str,
            timestamp: int,
            config: Optional[RecallConfig]=None
        ) -> MemoryDAG | bool:
        '''
        Get the next pending thread for the sona.
        
        Returns False if the sona was not found, True if the sona was found but
        there are no pending ACTs, and otherwise the memory subgraph of the
        pending thread.
        '''
        with self.db.transaction() as db:
            # Find or stage the active thread
            if (sona_row := db.find_sona(sona)) is None:
                return False
            
            if act := db.get_act_active(sona_row.rowid):
                memory_id = act.memory_id
            else: # No active thread, check for pending
                if (memory_id := sona_row.pending_id) is None:
                    return True # No threads at all
                db.sona_stage_active(sona_row.rowid)
            
            # We used the incomplete memory's edges to store prompts, only now
            #  do we actually run recall on them.
            edges: list[Edge[CIDv1]] = []
            for e in db.dependencies(rowid=memory_id):
                prompt = e.target.to_partial()
                # Don't bother with index, it should've been used for insertion
                # before this point.
                for row, score in db.recall(sona, prompt, None, timestamp, config):
                    if cid := row.cid:
                        edges.append(Edge(
                            target=CIDv1(cid),
                            weight=e.weight*score
                        ))
            
            return self.build_subgraph(edges)
    
    def update_invalid(self) -> bool:
        with self.db.transaction() as db:
            return db.update_invalid()

    def recall(self,
            sona: Optional[UUID|str],
            prompt: AnyMemory,
            timestamp: int,
            index: Optional[list[str]]=None,
            config: Optional[RecallConfig]=None
        ) -> MemoryDAG:
        '''Recall memories based on a prompt as a memory subgraph.'''
        with self.db.transaction() as db:
            return self.build_subgraph(
                prompt.edges + [
                    Edge(target=CIDv1(row.cid), weight=score)
                        for row, score in db.recall(
                            sona, prompt, index, timestamp, config
                        )
                ]
            )
    
    def list_messages(self,
            page: int,
            perpage: int
        ) -> Iterable[Memory]:
        '''List messages in a sona.'''
        for row in self.db.list_memories(page, perpage):
            yield Memory(
                data=MemoryDataAdapter.validate_json(row.data),
                edges=list(self.db.backward_edges(rowid=row.rowid))
            )
    
    def list_sonas(self,
            page: int,
            perpage: int
        ) -> Iterable[Sona]:
        '''List sonas in the database.'''
        with self.db.transaction() as db:
            for row in db.list_sonas(page, perpage):
                yield Sona(
                    uuid=UUID(bytes=row.uuid),
                    aliases=db.select_sona_aliases(row.rowid),
                    pending=db.get_incomplete_act(row.pending_id)
                        if row.pending_id else None,
                    active=db.get_incomplete_act(row.active_id)
                        if row.active_id else None
                )