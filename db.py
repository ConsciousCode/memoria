from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal, NamedTuple, Optional, TypedDict, cast
import sqlite3
import json
from uuid import UUID

from cid import CIDv1
from numpy import ndarray
import numpy
from pydantic import validate_call
import sqlite_vec
from fastembed import TextEmbedding
from uuid_extensions import uuid7

import ipld
from models import Edge, Memory, MemoryKind
from util import finite, json_t

nomic_text = TextEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
)

DEFAULT_IMPORTANCE = 0.30
DEFAULT_RECENCY = 0.30
DEFAULT_FTS = 0.15
DEFAULT_VSS = 0.25
DEFAULT_K = 20

class SelfMemory(TypedDict):
    name: Optional[str]
    content: str

class OtherMemory(TypedDict):
    name: str
    content: str

class MemoryRow(NamedTuple):
    '''Raw memory row from the database.'''
    rowid: int
    cid: bytes
    timestamp: Optional[float]
    kind: MemoryKind
    data: str
    importance: Optional[float]

class SonaRow(NamedTuple):
    rowid: int
    uuid: UUID
    last_id: Optional[int]

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
    
    def find_sona(self, name: str):
        '''
        Find the sona which is closest to the given name, or else create
        a new one.
        '''
        e, = nomic_text.embed(name)
        ebs = e.tobytes()
        cur = self.cursor()
        cur.execute("""
            SELECT rowid, sona_id, last_id FROM sona_embedding
            WHERE embedding MATCH ? AND distance > 0.75
            LIMIT 1 -- vec requires k to be set
        """, (ebs,))
        if row := cur.fetchone():
            return SonaRow(row[0], UUID(bytes=row[1]), row[2])
        
        u = cast(UUID, uuid7())

        cur.execute("INSERT INTO sona (uuid) VALUES (?)", (u.bytes,))
        if (rowid := cur.lastrowid) is None:
            raise RuntimeError("Failed to insert sona")

        cur.execute("""
            INSERT INTO sona_embedding (sona_id, embedding)
            VALUES (?, ?)
        """, (rowid, ebs))
        
        return SonaRow(rowid, u, None)

    def select_memory(self, key: Optional[int|bytes|CIDv1]) -> Optional[MemoryRow]:
        if key is None:
            return None
        cur = self.cursor()
        if isinstance(key, int):
            cur.execute("""
                SELECT
                    m.rowid, m.cid, m.timestamp,
                    m.kind, JSON(m.data), m.importance
                FROM memories WHERE rowid = ?
            """, (key,))
        else:
            if isinstance(key, CIDv1):
                key = key.buffer
            cur.execute("""
                SELECT
                    m.rowid, m.cid, m.timestamp,
                    m.kind, JSON(m.data), m.importance
                FROM memories WHERE cid = ?
            """, (key,))
        return MemoryRow(*cur.fetchone())

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

    def insert_memory(self,
            memory: Memory,
            index: Optional[str] = None,
            importance: Optional[float] = None,
            edges: dict[str, list[Edge]] = {}
        ) -> tuple[int, CIDv1]:
        cid = memory.cid()
        cur = self.cursor()
        cur.execute("""
            INSERT INTO memories (cid, timestamp, kind, data, importance)
                VALUES (?, ?, ?, JSONB(?), ?)
        """, (cid, memory.timestamp, memory.kind, json.dumps(memory.data), importance))
        
        if (rowid := cur.lastrowid) is None:
            raise RuntimeError("Failed to insert memory")

        if index is not None:
            cur.execute("""
                INSERT INTO memory_fts (rowid, content) VALUES (?, ?)
            """, (rowid, index))
            self.insert_text_embedding(rowid, index)

        cur.executemany("""
            INSERT INTO edges (src_id, dst_id, label, weight)
            SELECT ?, dst_id, ?, ?
            FROM memories WHERE rowid cid = ?
        """, (
            (rowid, label, e.weight, rowid, e.target)
                for label, dsts in edges.items()
                    for e in dsts
        ))

        return rowid, cid
    
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

    # Note: Because we're using parameters to build the SQL query, we need to be absolutely sure that they're actually the types they say to avoid SQL injection.
    @validate_call
    def recall(self,
            prompt: str,
            timestamp: Optional[float]=None,
            importance: Optional[float]=None,
            recency: Optional[float]=None,
            fts: Optional[float]=None,
            vss: Optional[float]=None,
            k: Optional[int]=None
        ) -> Iterable[tuple[MemoryRow, float]]:
        # Even 1
        # Be defensive against SQL injection
        importance = finite(float(importance or DEFAULT_IMPORTANCE))
        recency = finite(float(recency or DEFAULT_RECENCY))
        fts = finite(float(fts or DEFAULT_FTS))
        vss = finite(float(vss or DEFAULT_VSS))
        k = int(k or DEFAULT_K)

        e, = nomic_text.embed(prompt)
        cur = self.cursor()
        ### DANGER ZONE ###
        # DO NOT SUFFER A BARE INTERPOLATION HERE - EVERY SINGLE INTERPOLATION
        # ****MUST**** BE WRAPPED IN A PYTHON PRIMITIVE TYPE CONSTRUCTOR
        cur.execute(f"""
            WITH
                fts AS (
                    SELECT rowid, bm25(memory_fts) AS score
                    FROM memory_fts
                    WHERE memory_fts MATCH ?
                    LIMIT {int(k)}
                ),
                vec AS (
                    SELECT memory_id, distance
                    FROM vss_nomic_v1_5_index
                    WHERE embedding MATCH ? AND k = {int(k)}
                )
            SELECT
                m.rowid, m.timestamp, m.kind, JSON(m.data), m.importance,
                IFNULL({float(importance):.4} * m.importance, 0) +
                IFNULL({float(recency):.4} * POWER(0.995, ? - m.timestamp), 0) +
                IFNULL({float(fts):.4} *
                    (fts.score - MIN(fts.score) OVER())
                    / (MAX(fts.score) OVER() - MIN(fts.score) OVER()), 0) +
                IFNULL({float(vss):.4} / (1 + vec.distance), 0) AS score
            FROM memories m
                LEFT JOIN fts ON m.rowid = fts.rowid
                LEFT JOIN vec ON m.rowid = vec.memory_id
            ORDER BY score DESC
            LIMIT {int(k)}
        """, (prompt, e.tobytes(), timestamp and timestamp))

        for row in cur:
            yield MemoryRow(*row[:-1]), row[-1]
