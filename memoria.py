from typing import Iterable, Optional

from pydantic import BaseModel

from ipld import CIDv1

from db import Database, MemoryRow, Memory
from models import DAGEdge, Memory, MemoryDAG, RecallConfig
from util import todo_list

class Memoria:
    '''
    Wraps all memoria-related functionality to abstract away the details
    of the underlying database, but doesn't implement the MCP server.
    '''
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
    
    def insert(self,
            memory: Memory,
            index: Optional[str] = None,
            importance: Optional[float] = None
        ) -> CIDv1:
        '''
        Append a memory to the sona file.
        '''

        _, cid = self.db.insert_memory(memory, index, importance)
        return cid
    
    def find_sona(self, sona: str):
        return self.db.find_sona(sona)
    
    def recall(self,
            prompt: str,
            include: Optional[Iterable[CIDv1]]=None,
            timestamp: Optional[float]=None,
            config: Optional[RecallConfig]=None
        ) -> MemoryDAG:
        '''
        Recall memories based on a prompt. This incorporates all indices
        and returns a topological sort of relevant memories.
        '''
        g = MemoryDAG()
        
        for c in include or []:
            if pm := self.db.select_memory(c):
                g.insert(CIDv1(pm.cid), Memory(
                    kind=pm.kind,
                    data=pm.data,
                    timestamp=pm.timestamp
                ))
        
        rows: list[tuple[MemoryRow, float]] = []
        
        if config:
            importance = config.importance
            recency = config.recency
            fts = config.fts
            vss = config.vss
            k = config.k
        else:
            importance = recency = fts = vss = k = None

        memories = self.db.recall(
            prompt, timestamp, importance, recency, fts, vss, k
        )
        # Populate the graph with nodes so we can detect when there are edges
        #  between our seletions
        for rs in memories:
            row, _ = rs
            rows.append(rs)
            g.insert(CIDv1(row.cid), Memory(
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

                g.insert(srccid, Memory(
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

                g.insert(srccid, Memory(
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