'''
Memoria is the immutable state which can't be advanced without external
intervention.
'''

from typing import override
from collections.abc import Iterable

from cid import CIDv1, CID
from ipfs import Blocksource

from .db import DatabaseRO, FileRow
from .memory import AnyMemory, Edge, Memory, MemoryContext, MemoryDAG, MemoryDataAdapter
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
        self.db: DatabaseRO = db
    
    @override
    def block_has(self, cid: CID) -> bool:
        return isinstance(cid, CIDv1) and self.db.has_cid(cid)

    @override
    def block_get(self, cid: CID) -> bytes | None:
        if not isinstance(cid, CIDv1):
            return None
        data = self.lookup_memory(cid)
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
            _ = db.register_file(cid, filename, mimetype, filesize, overhead)

    def lookup_memory(self, cid: CID) -> Memory | None:
        if isinstance(cid, CIDv1):
            return self.db.select_memory_ipld(cid=cid)

    def lookup_file(self, cid: CID) -> FileRow | None:
        return self.db.select_file(cid=cid)

    def insert(self,
            memory: Memory,
            index: list[str] | None = None,
            timestamp: int | None = None
        ):
        '''Append a memory to the sona file.'''
        with self.db.transaction() as db:
            _ = db.insert_memory(memory, index, timestamp)
    
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

    def update_invalid(self) -> bool:
        with self.db.transaction() as db:
            return db.update_invalid()

    def recall(self,
            prompt: AnyMemory,
            timestamp: int,
            index: list[str] | None=None,
            config: RecallConfig | None=None
        ) -> MemoryDAG:
        '''Recall memories based on a prompt as a memory subgraph.'''
        with self.db.transaction() as db:
            return self.build_subgraph(
                prompt.edges + [
                    Edge(target=CIDv1(row.cid), weight=score)
                        for row, score in db.recall(
                            prompt, index, timestamp, config
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
