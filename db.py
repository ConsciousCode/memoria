
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Literal, NamedTuple, Optional, Self, TypedDict, cast, overload
import sqlite3
import json

from graph import Graph
from util import json_t

import sqlite_vec

type MemoryKind = Literal["self", "other", "text", "image", "file", "entity"]

class SelfMemory(TypedDict):
    name: Optional[str]
    content: str

class OtherMemory(TypedDict):
    name: str
    content: str

@dataclass
class MemoryRow:
    rowid: int
    timestamp: Optional[datetime]
    kind: MemoryKind
    data: json_t
    importance: Optional[float] = None

    @classmethod
    @overload
    def from_row(cls, row: None) -> None: ...

    @classmethod
    @overload
    def from_row(cls, row: tuple[int, float, MemoryKind, str, float]) -> Self: ...

    @classmethod
    def from_row(cls, row: Optional[tuple[int, float, MemoryKind, str, float]]):
        if row is None:
            return None
        
        rowid, ts, kind, data, importance = row
        if ts is not None:
            ts = datetime.fromtimestamp(ts)
        return cls(rowid, ts, kind, json.loads(data), importance)

class BackwardEdge(NamedTuple):
    dst: MemoryRow
    label: str
    weight: float

class ForwardEdge(NamedTuple):
    label: str
    weight: float
    src: MemoryRow

@dataclass
class Database:
    db_path: str = ":memory:"
    file_path: str = "files"
    
    def __enter__(self):
        conn = self.conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        with open("schema.sql", "r") as f:
            self.cursor().executescript(f.read())
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.rollback() if exc_type else self.commit()
        self.close()
        del self.conn
        return False
    
    def cursor(self):
        return self.conn.cursor()

    def commit(self):
        self.conn.commit()
    
    def rollback(self):
        self.conn.rollback()
    
    def close(self):
        self.conn.close()
    
    def file_lookup(self, mh: str, ext: str):
        fn, x, yz, rest = mh[:2], mh[2], mh[3:5], mh[5:]
        return f"{self.file_path}/{fn}/{x}/{yz}/{rest}{ext}"
    
    def select_memory(self, rowid: Optional[int]) -> Optional[MemoryRow]:
        if rowid is None:
            return None
        cur = self.cursor()
        cur.execute("""
            SELECT * FROM memories WHERE rowid = ?
        """, (rowid,))
        return MemoryRow.from_row(cur.fetchone())

    def insert_memory(self, kind: MemoryKind, data: Any, fts: Optional[str], importance: Optional[float] = None, timestamp: Optional[datetime] = None) -> int:
        ts = timestamp and timestamp.timestamp()
        cur = self.cursor()
        cur.execute("""
            INSERT INTO memories (timestamp, kind, data, importance)
                VALUES (?, ?, JSONB(?), ?)
        """, (ts, kind, json.dumps(data), importance))
        
        if (rowid := cur.lastrowid) is None:
            raise RuntimeError("Failed to insert memory")
        
        if fts is not None:
            cur.execute("""
                INSERT INTO memory_fts (rowid, content) VALUES (?, ?)
            """, (rowid, fts))
        
        return rowid

    def insert_text(self, text: str, importance: Optional[float] = None, timestamp: Optional[datetime] = None) -> int:
        return self.insert_memory("text", text, text, importance, timestamp)
    
    def insert_self(self, text: str, timestamp: Optional[datetime] = None) -> int:
        return self.insert_memory("self", text, text, None, timestamp or datetime.now())
    
    def insert_other(self, name: str, text: str, importance: Optional[float] = None, timestamp: Optional[datetime] = None) -> int:
        return self.insert_memory("other", {
            "name": name,
            "content": text
        }, text, importance, timestamp)

    '''
    def insert_image(self, memory_id: int, image: ImageInput) -> int:
        e, = nomic_image.embed(image)
        rowid = self.insert_memory("image", e, None)
        cur = self.cursor()
        cur.execute("""
            INSERT INTO vss_nomic_v1_5_index
                (memory_id, embedding) VALUES (?, ?)
        """, (memory_id, e.tobytes()))

        return rowid
    '''

    def link(self, label: str, weight: float, src_id: int, dst_id: int):
        cur = self.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO edges
                (label, weight, src_id, dst_id) VALUES (?, ?, ?, ?)
        """, (label, weight, src_id, dst_id))

    def link_many(self, edges: Iterable[tuple[str, float, int, int]]):
        cur = self.cursor()
        cur.executemany("""
            INSERT OR IGNORE INTO edges
                (label, weight, src_id, dst_id) VALUES (?, ?, ?, ?)
        """, edges)
    
    # dst <- src

    def backward_edges(self, src_id: int) -> Iterable[BackwardEdge]:
        '''
        Get all edges leading to the given memory, returning the source id
        and the label and weight of the edge.
        '''
        cur = self.cursor()
        cur.execute("""
            SELECT
                m.rowid, m.timestamp, m.kind, JSON(m.data), m.importance,
                e.label, e.weight
            FROM edges e JOIN memories m ON m.rowid = e.dst_id
            WHERE e.src_id = ?
            ORDER BY e.weight DESC
        """, (src_id,))
        
        for row in cur:
            yield BackwardEdge(MemoryRow.from_row(row[:5]), row[5], row[6])
    
    def forward_edges(self, dst_id: int) -> Iterable[ForwardEdge]:
        '''
        Get all edges leading from the given memory, returning the destination id
        and the label and weight of the edge.
        '''
        cur = self.cursor()
        cur.execute("""
            SELECT
                m.rowid, m.timestamp, m.kind, JSON(m.data), m.importance,
                e.label, e.weight
            FROM edges e JOIN memories m ON m.rowid = e.src_id
            WHERE e.dst_id = ?
            ORDER BY m.importance DESC
        """, (dst_id,))
        
        for row in cur:
            yield ForwardEdge(row[5], row[6], MemoryRow.from_row(row[:5]))
    
    def edge(self, src_id: int, dst_id: int) -> Optional[tuple[str, float]]:
        '''
        Get the edge label and weight between two memories.
        Returns None if no edge exists.
        '''
        cur = self.cursor()
        return cur.execute("""
            SELECT label, weight FROM edges
            WHERE src_id = ? AND dst_id = ?
        """, (src_id, dst_id)).fetchone()

    def recall(self, prev: Optional[int], prompt: str) -> Graph[int, tuple[str, float], MemoryRow]:
        '''
        Recall memories based on a prompt. This incorporates all indices
        and returns a topological sort of relevant memories.
        '''

        g = Graph[int, tuple[str, float], MemoryRow]()
        if pm := self.select_memory(prev):
            g.insert(pm.rowid, pm)
        
        e, = nomic_text.embed(prompt)
        cur = self.cursor()
        cur.execute("""
            WITH
                fts AS (
                    SELECT rowid, bm25(memory_fts) AS score
                    FROM memory_fts
                    WHERE memory_fts MATCH ?
                    LIMIT 20
                ),
                vec AS (
                    SELECT memory_id, distance
                    FROM vss_nomic_v1_5_index
                    WHERE embedding MATCH ? AND k = 20
                )
            SELECT
                m.rowid, m.timestamp, m.kind, JSON(m.data), m.importance,
                IFNULL(0.30 * m.importance, 0) +
                IFNULL(0.30 * POWER(0.995, ? - m.timestamp), 0) +
                IFNULL(0.15 *
                    (fts.score - MIN(fts.score) OVER())
                    / (MAX(fts.score) OVER() - MIN(fts.score) OVER()), 0) +
                IFNULL(0.25 / (1 + vec.distance), 0) AS score
            FROM memories m
                LEFT JOIN fts ON m.rowid = fts.rowid
                LEFT JOIN vec ON m.rowid = vec.memory_id
            ORDER BY score DESC
            LIMIT 20
        """, (json.dumps(prompt), e.tobytes(), datetime.now().timestamp()))

        rows = cur.fetchall()
        
        # Populate the graph with nodes so we can detect when there are edges
        #  between our seletions
        for row in rows:
            m = MemoryRow.from_row(row[:5])
            g.insert(m.rowid, m)

        # Populate backward and forward edges
        bw: list[tuple[float, int]] = []
        fw: list[tuple[float, int]] = []
        
        for row in rows:
            rowid, score = row[0], row[-1]
            print(f"{score=}")
            if score <= 0:
                break

            budget = score*20

            b = 0
            for dst, label, weight in self.backward_edges(rowid):
                if dst.rowid in g:
                    if not g.has_edge(rowid, dst.rowid):
                        g.add_edge(rowid, dst.rowid, (label, weight))
                    continue

                b += weight
                if b >= budget:
                    break
                
                bw.append((budget*weight, dst.rowid))
            
            b = 0
            for label, weight, src in self.forward_edges(rowid):
                if src.rowid in g:
                    if not g.has_edge(src.rowid, rowid):
                        g.add_edge(src.rowid, rowid, (label, weight))
                    continue

                if not src.importance:
                    break
                
                b += src.importance
                if b >= budget:
                    break
                
                fw.append((budget*src.importance, src.rowid))
        
        print(fw, bw)
        # Note: These feel so similar, maybe there's a way to reduce boilerplate?

        # Search backwards for supporting memories using their edge weight
        #  to determine how relevant they are to the current memory
        for budget, src_id in todo_list(bw):
            print(f"{src_id=}")
            b = 0
            for dst, label, weight in self.backward_edges(src_id):

                print(f"{b=}, {weight=}")
                b += weight
                if b >= budget:
                    break
                
                g.insert(src_id, dst)
                g.add_edge(src_id, dst.rowid, (label, weight))

                bw.append((budget*weight, dst.rowid))
        
        # Search forwards for response memories using their annotated
        #  importance - important conclusions are more relevant
        for budget, dst_id in todo_list(fw):
            b = 0
            for label, weight, src in self.forward_edges(dst_id):
                b += (imp := src.importance or 0)
                if b >= budget:
                    break

                g.insert(src.rowid, src)
                g.add_edge(src.rowid, dst_id, (label, weight))

                fw.append((budget*imp, dst_id))

        print(g.adj)
        return g