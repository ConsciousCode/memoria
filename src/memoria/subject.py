'''
Memoria is the immutable state which can't be advanced without external
intervention.
'''

from collections import defaultdict
from collections.abc import Iterable
from typing import override

from cid import CIDv1, CID
from ipfs import Blocksource

from .db import DatabaseRO, MemoryRow
from .memory import Memory, MemoryDAG, PartialMemory
from .config import RecallConfig

__all__ = (
    'Subject',
)

class Subject(Blocksource):
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
        if isinstance(cid, CIDv1):
            if data := self.get_memory(cid):
                if m := data.complete():
                    return m.ipld_block()
        return None
    
    def get_memory(self, cid: CID) -> PartialMemory | None:
        if isinstance(cid, CIDv1):
            return self.db.select_memory_ipld(cid=cid)

    def add_memory(self, memory: Memory):
        '''Append a memory to DAG.'''
        with self.db.transaction() as db:
            _ = db.insert_memory(memory)

    def recall(self,
            roots: list[CIDv1],
            config: RecallConfig | None=None
        ) -> MemoryDAG:
        '''Recall memories based on a prompt as a memory subgraph.'''
        # Implements a BFS to populate the subgraph. Sweeps from the roots
        # separately backward in time through dependencies and forward in time
        # through references. Once the combined working sets exceed the node
        # limit, they're sorted by refcount and then only the top are added up
        # to the limit.
        
        # Normalize the config
        config = config or RecallConfig()
        refs_depth = config.refs or float('inf')
        deps_depth = config.deps or float('inf')
        memories = config.memories
        
        g = MemoryDAG()
        seen = set[CIDv1](roots) # g ∪ fringe for quick lookup
        rc = defaultdict[CIDv1, int](int) # Refcounts of fringe nodes (xor g)
        refs = dict[CIDv1, tuple[CIDv1, MemoryRow]]() # {src: (dst, row)}
        deps = dict[CIDv1, tuple[CIDv1, MemoryRow]]() # {dst: (src, row)}
        refs_next = dict[CIDv1, tuple[CIDv1, MemoryRow]]()
        deps_next = dict[CIDv1, tuple[CIDv1, MemoryRow]]()
        
        # Load the roots and their reference/dependency fringes
        for mem in self.db.select_memories(roots):
            cid = CIDv1(mem.cid)
            g.insert(cid, mem.to_partial())
            
            for ref in self.db.references(rowid=mem.rowid):
                rcid = CIDv1(ref.cid)
                rc[rcid] += 1
                if rcid not in seen:
                    seen.add(rcid)
                    refs[rcid] = cid, ref
            
            for dep in self.db.dependencies(rowid=mem.rowid):
                dcid = CIDv1(dep.cid)
                rc[dcid] += 1
                if dcid not in seen:
                    seen.add(dcid)
                    deps[dcid] = cid, dep
        
        # Depth limit is checked at the end of the loop so we have to do it here
        if refs_depth < 1:
            refs.clear()
        if deps_depth < 1:
            deps.clear()
        
        depth = 0
        
        # Only stop if we run out or would exceed the memory limit
        while (refs or deps) and len(seen) < memories:
            depth += 1
            rc.clear()
            
            # Unlike the roots, the fringes only advance in their
            # respective directions. We also add edges in the graph.
            for cid, (dst, ref) in refs.items():
                g.insert(cid, ref.to_partial())
                g.add_edge(cid, dst)
                
                for rref in self.db.references(rowid=ref.rowid):
                    rcid = CIDv1(rref.cid)
                    rc[rcid] += 1
                    if rcid not in seen:
                        seen.add(rcid)
                        refs_next[rcid] = cid, rref
            
            for cid, (src, dep) in deps.items():
                g.insert(cid, dep.to_partial())
                g.add_edge(src, cid)
                
                for ddep in self.db.dependencies(rowid=dep.rowid):
                    dcid = CIDv1(ddep.cid)
                    rc[dcid] += 1
                    if dcid not in seen:
                        seen.add(dcid)
                        deps_next[dcid] = cid, ddep
            
            # Swap the working sets
            refs.clear()
            deps.clear()
            refs, refs_next = refs_next, refs
            deps, deps_next = deps_next, deps
            
            # Check depth limits
            if refs_depth < depth:
                refs.clear()
            if deps_depth < depth:
                deps.clear()
        
        # Sort everything by reference count
        rest = list[tuple[CIDv1, CIDv1, CIDv1, MemoryRow]]()
        rest.extend(((src, src, dst, row) for src, (dst, row) in refs.items()))
        rest.extend(((dst, src, dst, row) for dst, (src, row) in deps.items()))
        rest.sort(key=lambda x: rc[x[0]], reverse=True)
        
        # Add the sorted items to the result
        for cid, src, dst, row in rest[:memories - len(g)]:
            g.insert(cid, row.to_partial())
            g.add_edge(src, dst)
        
        return g
    
    def list_memories(self,
            page: int,
            perpage: int
        ) -> Iterable[Memory]:
        '''List messages in a sona.'''
        for row in self.db.list_memories(page, perpage):
            m = row.to_partial(
                edges=set(self.db.backward_edges(rowid=row.rowid))
            ).complete()
            if m is not None:
                yield m