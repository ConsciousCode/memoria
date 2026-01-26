'''
Memoria is the immutable state which can't be advanced without external
intervention.
'''

from collections import defaultdict
from collections.abc import Iterable
from uuid import UUID

from .db import DatabaseRO, MemoryRow
from .memory import Memory, MemoryDAG
from .config import RecallConfig

__all__ = (
    'Subject',
)

class Subject:
    '''
    Wraps all memoria-related functionality to abstract away the details
    of the underlying database.
    '''

    def __init__(self, db: DatabaseRO):
        self.db: DatabaseRO = db

    def get_memory(self, uuid: UUID) -> Memory | None:
        return self.db.select_memory_full(uuid=uuid)

    def add_memory(self, memory: Memory):
        '''Append a memory to DAG.'''
        with self.db.transaction() as db:
            _ = db.insert_memory(memory)

    def recall(self,
            roots: list[UUID],
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
        seen = set[UUID](roots) # g ∪ fringe for quick lookup
        rc = defaultdict[UUID, int](int) # Refcounts of fringe nodes (xor g)
        refs = dict[UUID, tuple[UUID, MemoryRow]]() # {src: (dst, row)}
        deps = dict[UUID, tuple[UUID, MemoryRow]]() # {dst: (src, row)}
        refs_next = dict[UUID, tuple[UUID, MemoryRow]]()
        deps_next = dict[UUID, tuple[UUID, MemoryRow]]()

        # Load the roots and their reference/dependency fringes
        for mem in self.db.select_memories(roots):
            uuid = UUID(bytes=mem.uuid)
            g.insert(uuid, mem.to_memory())

            for ref in self.db.references(rowid=mem.rowid):
                ruuid = UUID(bytes=ref.uuid)
                rc[ruuid] += 1
                if ruuid not in seen:
                    seen.add(ruuid)
                    refs[ruuid] = uuid, ref

            for dep in self.db.dependencies(rowid=mem.rowid):
                duuid = UUID(bytes=dep.uuid)
                rc[duuid] += 1
                if duuid not in seen:
                    seen.add(duuid)
                    deps[duuid] = uuid, dep

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
            for uuid, (dst, ref) in refs.items():
                g.insert(uuid, ref.to_memory())
                g.add_edge(uuid, dst)

                for rref in self.db.references(rowid=ref.rowid):
                    ruuid = UUID(bytes=rref.uuid)
                    rc[ruuid] += 1
                    if ruuid not in seen:
                        seen.add(ruuid)
                        refs_next[ruuid] = uuid, rref

            for uuid, (src, dep) in deps.items():
                g.insert(uuid, dep.to_memory())
                g.add_edge(src, uuid)

                for ddep in self.db.dependencies(rowid=dep.rowid):
                    duuid = UUID(bytes=ddep.uuid)
                    rc[duuid] += 1
                    if duuid not in seen:
                        seen.add(duuid)
                        deps_next[duuid] = uuid, ddep

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
        rest = list[tuple[UUID, UUID, UUID, MemoryRow]]()
        rest.extend(((src, src, dst, row) for src, (dst, row) in refs.items()))
        rest.extend(((dst, src, dst, row) for dst, (src, row) in deps.items()))
        rest.sort(key=lambda x: rc[x[0]], reverse=True)

        # Add the sorted items to the result
        for uuid, src, dst, row in rest[:memories - len(g)]:
            g.insert(uuid, row.to_memory())
            g.add_edge(src, dst)

        return g
    
    def list_memories(self,
            page: int,
            perpage: int
        ) -> Iterable[Memory]:
        '''List memories from the database.'''
        for row in self.db.list_memories(page, perpage):
            yield row.to_memory(
                edges=set(self.db.backward_edges(rowid=row.rowid))
            )