
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Iterable, Literal, Mapping, NamedTuple, Optional, Self, Sequence, TypedDict, cast, overload
import sqlite3
import json
from uuid import UUID

from numpy import ndarray
import numpy
from openai import BaseModel
from uuid_extensions import uuid7

from util import finite, json_t

import sqlite_vec
from fastembed import TextEmbedding

nomic_text = TextEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
)

DEFAULT_IMPORTANCE = 0.30
DEFAULT_RECENCY = 0.30
DEFAULT_FTS = 0.15
DEFAULT_VSS = 0.25
DEFAULT_K = 20

type MemoryKind = Literal["self", "other", "text", "image", "file", "entity"]

class SelfMemory(BaseModel):
    name: Optional[str]
    content: str

class OtherMemory(BaseModel):
    name: str
    content: str

class MemoryRow(NamedTuple):
    '''Raw memory row from the database.'''
    rowid: int
    uuid: bytes
    timestamp: Optional[float]
    kind: MemoryKind
    data: str
    importance: Optional[float]

@dataclass
class Memory:
    rowid: int
    uuid: UUID
    timestamp: Optional[datetime]
    kind: MemoryKind
    data: json_t
    importance: Optional[float]
    #edges: dict[str, list[int]]
    #role: Optional[Literal['prev']] = None

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
    
    def cursor(self): return self.conn.cursor()
    def commit(self): self.conn.commit()
    def rollback(self): self.conn.rollback()
    def close(self): self.conn.close()
    
    def file_lookup(self, mh: str, ext: str):
        fn, x, yz, rest = mh[:2], mh[2], mh[3:5], mh[5:]
        return f"{self.file_path}/{fn}/{x}/{yz}/{rest}{ext}"
    
    def select_memory_rowid(self, rowid: Optional[int]) -> Optional[MemoryRow]:
        if rowid is None:
            return None
        cur = self.cursor()
        cur.execute("""
            SELECT
                m.rowid, m.uuid, m.timestamp, m.kind, JSON(m.data), m.importance
            FROM memories WHERE rowid = ?
        """, (rowid,))
        return MemoryRow(*cur.fetchone())

    def select_memory(self, uuid: UUID) -> Optional[MemoryRow]:
        cur = self.cursor()
        cur.execute("""
            SELECT * FROM memories WHERE uuid = ?
        """, (uuid.bytes,))
        return MemoryRow(*cur.fetchone())

    def memory_uuid_to_rowid(self, uuid: UUID) -> Optional[int]:
        cur = self.cursor()
        cur.execute("""
            SELECT uuid FROM memories WHERE rowid = ?
        """, (uuid,))
        row = cur.fetchone()
        return row and row[0]

    def select_embedding(self, memory_id: int) -> Optional[ndarray]:
        cur = self.cursor()
        cur.execute("""
            SELECT embedding FROM vss_nomic_v1_5_index WHERE memory_id = ?
        """, (memory_id,))
        if row := cur.fetchone():
            return numpy.frombuffer(row[0], dtype=numpy.float32)
        return None

    def insert_text_embedding(self, memory_id: int, text: str):
        e, = nomic_text.embed(text)
        cur = self.cursor()
        cur.execute("""
            INSERT INTO vss_nomic_v1_5_index (memory_id, embedding)
            VALUES (?, ?)
        """, (memory_id, e.tobytes()))

    def insert_memory(self, kind: MemoryKind, data: json_t, fts: Optional[str], importance: Optional[float] = None, timestamp: Optional[datetime] = None) -> Memory:
        u = cast(UUID, uuid7())
        cur = self.cursor()
        cur.execute("""
            INSERT INTO memories (uuid, timestamp, kind, data, importance)
                VALUES (?, ?, JSONB(?), ?)
        """, (u, timestamp and timestamp.timestamp(), kind, json.dumps(data), importance))
        
        if (rowid := cur.lastrowid) is None:
            raise RuntimeError("Failed to insert memory")

        if fts is not None:
            cur.execute("""
                INSERT INTO memory_fts (rowid, content) VALUES (?, ?)
            """, (rowid, fts))
            self.insert_text_embedding(rowid, fts)
        
        return Memory(
            rowid=rowid,
            uuid=u,
            timestamp=timestamp,
            kind=kind,
            data=data,
            importance=importance
        )

    def insert_text(self, text: str, importance: Optional[float] = None, timestamp: Optional[datetime] = None) -> Memory:
        return self.insert_memory("text", text, text, importance, timestamp)
    
    def insert_self(self, text: str, timestamp: Optional[datetime] = None) -> Memory:
        return self.insert_memory("self", text, text, None, timestamp or datetime.now())
    
    def insert_other(self, name: str, text: str, importance: Optional[float] = None, timestamp: Optional[datetime] = None) -> Memory:
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

    def link_many(self, edges: Iterable[tuple[str, float, int|UUID, int|UUID]]):
        cur = self.cursor()
        cur.executemany("""
            INSERT OR IGNORE INTO edges
                (label, weight, src_id, dst_id) VALUES (?, ?, ?, ?)
        """, edges)

    def link(self, label: str, weight: float, src_id: int, dst_id: int):
        self.link_many([(label, weight, src_id, dst_id)])
    
    # dst <- src

    def backward_edges(self, src_id: int) -> Iterable[BackwardEdge]:
        '''
        Get all edges leading to the given memory, returning the source id
        and the label and weight of the edge.
        '''
        cur = self.cursor()
        cur.execute("""
            SELECT
                m.rowid, m.uuid, m.timestamp, m.kind, JSON(m.data), m.importance,
                e.label, e.weight
            FROM edges e JOIN memories m ON m.rowid = e.dst_id
            WHERE e.src_id = ?
            ORDER BY e.weight DESC
        """, (src_id,))
        
        for row in cur:
            yield BackwardEdge(MemoryRow(*row[:6]), row[5], row[6])
    
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
            yield ForwardEdge(row[5], row[6], MemoryRow(*row[:6]))
    
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

    def recall(self,
            prompt: str,
            timestamp: Optional[datetime]=None,
            importance: Optional[float]=None,
            recency: Optional[float]=None,
            fts: Optional[float]=None,
            vss: Optional[float]=None,
            k: Optional[int]=None
        ) -> Iterable[tuple[MemoryRow, float]]:
        # Be defensive against SQL injection
        importance = finite(float(importance or DEFAULT_IMPORTANCE))
        recency = finite(float(recency or DEFAULT_RECENCY))
        fts = finite(float(fts or DEFAULT_FTS))
        vss = finite(float(vss or DEFAULT_VSS))
        k = int(k or DEFAULT_K)

        e, = nomic_text.embed(prompt)
        cur = self.cursor()
        cur.execute(f"""
            WITH
                fts AS (
                    SELECT rowid, bm25(memory_fts) AS score
                    FROM memory_fts
                    WHERE memory_fts MATCH ?
                    LIMIT {k}
                ),
                vec AS (
                    SELECT memory_id, distance
                    FROM vss_nomic_v1_5_index
                    WHERE embedding MATCH ? AND k = {k}
                )
            SELECT
                m.rowid, m.timestamp, m.kind, JSON(m.data), m.importance,
                IFNULL({importance} * m.importance, 0) +
                IFNULL({recency} * POWER(0.995, ? - m.timestamp), 0) +
                IFNULL({fts} *
                    (fts.score - MIN(fts.score) OVER())
                    / (MAX(fts.score) OVER() - MIN(fts.score) OVER()), 0) +
                IFNULL({vss} / (1 + vec.distance), 0) AS score
            FROM memories m
                LEFT JOIN fts ON m.rowid = fts.rowid
                LEFT JOIN vec ON m.rowid = vec.memory_id
            ORDER BY score DESC
            LIMIT {k}
        """, (json.dumps(prompt), e.tobytes(), timestamp and timestamp.timestamp()))

        for row in cur:
            yield MemoryRow(*row[:-1]), row[-1]
