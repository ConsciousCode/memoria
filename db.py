from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, Iterable, NamedTuple, Optional, TypedDict, cast
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

from models import ACThread, Edge, FileMemory, Memory, MemoryKind, OtherMemory, RecallConfig, SelfMemory, TextMemory, build_memory
from util import classproperty, finite, json_t

nomic_text = TextEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
)

type JSONB = str
'''Alias for JSONB type in SQLite, which is selected as a string.'''

DEFAULT_IMPORTANCE = 0.30
DEFAULT_RECENCY = 0.30
DEFAULT_FTS = 0.15
DEFAULT_VSS = 0.25
DEFAULT_K = 20

## Rows are represented as the raw output from a select query with no processing
## - this allows us to avoid processing columns we don't need
## PrimaryKey aliases also give us little semantic type hints for the linter

class FileRow(NamedTuple):
    type PrimaryKey = int
    rowid: PrimaryKey
    cid: bytes
    filename: Optional[str]
    mimetype: str
    metadata: Optional[JSONB]
    size: int
    content: Optional[bytes]

class MemoryRow(NamedTuple):
    type PrimaryKey = int
    rowid: PrimaryKey
    cid: bytes
    timestamp: Optional[float]
    kind: MemoryKind
    data: JSONB
    importance: Optional[float]

class EdgeRow(NamedTuple):
    src_id: MemoryRow.PrimaryKey
    dst_id: MemoryRow.PrimaryKey
    label: str
    weight: float

class SonaRow(NamedTuple):
    type PrimaryKey = int
    rowid: PrimaryKey
    uuid: bytes
    active_id: Optional['ACThreadRow.PrimaryKey']
    pending_id: Optional['ACThreadRow.PrimaryKey']

class ACThreadRow(NamedTuple):
    type PrimaryKey = int
    rowid: PrimaryKey
    cid: bytes
    sona_id: SonaRow.PrimaryKey
    memory_id: MemoryRow.PrimaryKey
    prev_id: Optional[int]

class BackwardEdge(NamedTuple):
    dst: MemoryRow
    label: str
    weight: float

class ForwardEdge(NamedTuple):
    label: str
    weight: float
    src: MemoryRow

with open("schema.sql", "r") as f:
    SCHEMA = f.read()

@dataclass
class Database:
    '''
    All SQL queries are contained within this database class. This prevents the
    proliferation of SQL queries throughout the codebase, allowing for easier
    maintenance and updates.

    However, these queries do not respect data integrity on their own - they
    must be used correctly to ensure that the database remains consistent. For
    instance, memories are inserted separately from their edges, but memories
    themselves have a `cid` which depends on those edges.
    '''

    db_path: str = ":memory:"
    file_path: str = "files"
    
    def __enter__(self):
        conn = self.conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        self.cursor().executescript(SCHEMA)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.rollback() if exc_type else self.commit()
        self.close()
        del self.conn
        return False
    
    def cursor[T](self, factory: Optional[type[T]]=None):
        cur = self.conn.cursor()
        if factory:
            cur.row_factory = lambda cur, row: row and factory(*row)
        return cur
    
    def commit(self): self.conn.commit()
    def rollback(self): self.conn.rollback()
    def close(self): self.conn.close()
    
    def file_lookup(self, mh: str, ext: str):
        fn, x, yz, rest = mh[:2], mh[2], mh[3:5], mh[5:]
        return f"{self.file_path}/{fn}/{x}/{yz}/{rest}{ext}"
    
    def index(self, memory_id: int, index: str):
        '''Index a memory with a text embedding.'''
        self.insert_text_embedding(memory_id, index)
        self.insert_text_fts(memory_id, index)
    
    def lookup_ipld_memory(self, cid: CIDv1) -> Optional[Memory]:
        cur = self.cursor(MemoryRow)
        mem: Optional[MemoryRow] = cur.execute("""
            SELECT cid, timestamp, kind, data, importance
            FROM memories
            WHERE cid = ?
        """, (cid,)).fetchone()
        if mem is None:
            return None
        
        cur = self.cursor(EdgeRow)
        cur.execute("""
            SELECT src_id, dst_id, label, weight
            FROM edges
            WHERE src_id = ?
        """, (mem.rowid,))

        edges: dict[str, list[Edge]] = defaultdict(list)
        for edge in cur:
            edges[edge.label].append(Edge(
                weight=edge.weight,
                target=CIDv1(edge.dst_id)
            ))
        
        m = build_memory(mem.kind, mem.data, mem.timestamp, edges)
        if m.cid != cid:
            raise ValueError(f"Memory CID {m.cid} does not match requested CID {cid}")
        
        return m
    
    def lookup_ipld_act(self, cid: CIDv1) -> Optional[ACThread]:
        cur = self.cursor()
        row = cur.execute("""
            SELECT s.cid, m.cid, p.cid
            FROM acthreads act
            JOIN sonas s ON s.rowid = act.sona_id
            JOIN memories m ON m.rowid = act.memory_id
            JOIN acthreads p ON p.rowid = act.prev_id
            WHERE act.cid = ?
        """, (cid,)).fetchone()
        if row is None:
            return None
        
        s, m, p = row
        t = ACThread(
            sona=CIDv1(s),
            memory=CIDv1(m),
            prev=p and CIDv1(p)
        )
        if t.cid != cid:
            raise ValueError(f"ACT CID {t.cid} does not match requested CID {cid}")
        return t

    def lookup_ipld(self, cid: CIDv1) -> Optional[FileRow|MemoryRow|ACThreadRow]:
        '''
        Lookup a CID in the database, returning the associated row if it exists.
        '''

        c = cid.buffer

        cur = self.cursor()
        cur.execute("""
            SELECT rowid, cid, timestamp, kind, JSON(data), importance
            FROM memories WHERE cid = ?
        """, (c,))
        if row := cur.fetchone():
            return MemoryRow(*row)

        cur.execute("""
            SELECT rowid, cid, filename, mimetype, JSON(metadata), size, content
            FROM files WHERE cid = ?
        """, (c,))
        if row := cur.fetchone():
            return FileRow(*row)

        cur.execute("""
            SELECT rowid, cid, sona_id, memory_id, prev_id
            FROM acthreads WHERE cid = ?
        """, (c,))
        if row := cur.fetchone():
            return ACThreadRow(*row)
        
        return None

    def find_sona(self, name: str):
        '''Find or create the sona closest to the given name.'''
        cur = self.cursor()
        cur.execute("""
            SELECT rowid, sona_id, pending
            FROM sona_aliases
            WHERE name = ?
        """, name)
        if row := cur.fetchone():
            return SonaRow(*row)

        e, = nomic_text.embed(name)
        ebs = e.tobytes()
        cur.execute("""
            SELECT rowid, sona_id, pending
            FROM sona_vss
            WHERE embedding MATCH ? AND distance > 0.75
            LIMIT 1 -- vec requires k to be set
        """, (ebs,))
        if row := cur.fetchone():
            return SonaRow(*row)
        
        u = cast(UUID, uuid7())

        cur.execute("INSERT INTO sona (uuid) VALUES (?)", (u.bytes,))
        if (rowid := cur.lastrowid) is None:
            raise RuntimeError("Failed to insert sona")

        cur.execute("""
            INSERT INTO sona_vss (sona_id, embedding) VALUES (?, ?)
        """, (rowid, ebs))
        cur.execute("""
            INSERT INTO sona_aliases (sona_id, name) VALUES (?, ?)
        """, (rowid, name))
        
        return SonaRow(rowid, u.bytes, None, None)

    def get_act_active(self, sona_id: int) -> Optional[ACThreadRow]:
        '''Get the sona's active thread node currently receiving updates.'''
        cur = self.cursor(ACThreadRow)
        return cur.execute("""
            SELECT act.rowid, cid, sona_id, memory_id, prev_id
            FROM sonas s
            JOIN acthreads act ON s.active_id = acthreads.rowid
            WHERE s.rowid = ?
        """, (sona_id,)).fetchone()

    def get_act_pending(self, sona_id: int) -> Optional[ACThreadRow]:
        '''Get the sona's pending thread node which is aggregating requests.'''
        cur = self.cursor(ACThreadRow)
        return cur.execute("""
            SELECT act.rowid, cid, sona_id, memory_id, last_id
            FROM sona s
            JOIN acthreads act ON s.pending_id = act.rowid
            WHERE s.rowid = ?
        """, (sona_id,)).fetchone()
    
    def select_memory_rowid(self, rowid: int) -> Optional[MemoryRow]:
        cur = self.cursor(MemoryRow)
        return cur.execute("""
            SELECT rowid, cid, timestamp, kind, JSON(data), importance
            FROM memories WHERE rowid = ?
        """, (rowid,)).fetchone()
    
    def select_memory_cid(self, cid: CIDv1) -> Optional[MemoryRow]:
        cur = self.cursor(MemoryRow)
        return cur.execute("""
            SELECT rowid, cid, timestamp, kind, JSON(data), importance
            FROM memories WHERE cid = ?
        """, (cid.buffer,)).fetchone()

    def select_embedding(self, memory_id: int) -> Optional[ndarray]:
        cur = self.cursor()
        cur.execute("""
            SELECT embedding FROM memory_vss WHERE memory_id = ?
        """, (memory_id,))
        if row := cur.fetchone():
            return numpy.frombuffer(row[0], dtype=numpy.float32)
        return None

    def insert_text_embedding(self, memory_id: int, index: str):
        e, = nomic_text.embed(index)
        cur = self.cursor()
        cur.execute("""
            INSERT INTO memory_vss (memory_id, embedding)
            VALUES (?, ?)
        """, (memory_id, e.tobytes()))

    def insert_text_fts(self, memory_id: int, index: str):
        '''Index a memory by inserting it into the full-text search index.'''
        cur = self.cursor()
        cur.execute("""
            INSERT INTO memory_fts (rowid, content)
            VALUES (?, ?)
        """, (memory_id, index))

    def insert_memory(self,
            cid: Optional[CIDv1],
            kind: MemoryKind,
            data: json_t,
            timestamp: Optional[float],
            importance: Optional[float] = None
        ) -> int:
        cur = self.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO memories
            (cid, timestamp, kind, data, importance)
            VALUES (?, ?, ?, JSONB(?), ?)
        """, (cid, timestamp, kind, json.dumps(data), importance))
        
        if (rowid := cur.lastrowid) is None:
            # Memory already exists
            rowid, = cur.execute("""
                SELECT rowid FROM memories WHERE cid = ?
            """, (cid,)).fetchone()

        return rowid
    
    def insert_act(self,
            cid: Optional[CIDv1],
            sona_id: int,
            memory_id: int,
            prev_id: Optional[int]
        ) -> int:
        '''Insert an acthread for a sona.'''
        cur = self.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO acthreads
            (cid, sona_id, memory_id, prev_id)
            VALUES (?, ?, ?, ?)
        """, (cid and cid.buffer, sona_id, memory_id, prev_id))
        
        if rowid := cur.lastrowid:
            return rowid
        raise RuntimeError("Failed to insert acthread, it may already exist.")
    
    def get_last_act(self, sona_id: int) -> Optional[ACThreadRow]:
        '''
        Get the last act thread for a sona.
        '''
        cur = self.cursor(ACThreadRow)
        return cur.execute("""
            SELECT rowid, cid, sona_id, memory_id, prev_id
            FROM acthreads
            WHERE sona_id = ?
            ORDER BY rowid DESC LIMIT 1
        """, (sona_id,)).fetchone()

    def link_memory_edges(self, rowid: int, edges: dict[str, list[Edge]]):
        '''
        Link the edges of a memory to the database. This is used when inserting
        a memory with edges that are already in the database.
        '''
        cur = self.cursor()
        cur.executemany("""
            INSERT OR IGNORE INTO edges (src_id, dst_id, label, weight)
            SELECT ?, rowid, ?, ?
            FROM memories m WHERE cid = ?
        """, (
            (rowid, label, e.weight, e.target.buffer)
                for label, dsts in edges.items()
                    for e in dsts
        ))
    
    def link_sona(self, sona_id: int, memory_id: int):
        '''Link a memory to a sona.'''
        cur = self.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO sona_memories
            (sona_id, memory_id) VALUES (?, ?)
        """, (sona_id, memory_id))
    
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
            sona: Optional[str],
            prompt: str,
            timestamp: Optional[float]=None,
            config: Optional[RecallConfig]=None
        ) -> Iterable[tuple[MemoryRow, float]]:
        config = config or RecallConfig()
        # Be defensive against SQL injection
        importance = finite(float(config.importance or DEFAULT_IMPORTANCE))
        recency = finite(float(config.recency or DEFAULT_RECENCY))
        fts = finite(float(config.fts or DEFAULT_FTS))
        vss = finite(float(config.vss or DEFAULT_VSS))
        k = int(config.k or DEFAULT_K)

        if sona:
            s, = nomic_text.embed(sona)
        else:
            s = None
        e, = nomic_text.embed(prompt)
        cur = self.cursor()
        ### DANGER ZONE ###
        # DO NOT SUFFER A BARE INTERPOLATION HERE - EVERY SINGLE INTERPOLATION
        # ****MUST**** BE WRAPPED IN A PYTHON PRIMITIVE TYPE CONSTRUCTOR
        # *AND* A REPR formatting function
        cur.execute(f"""
            WITH
                fts AS (
                    SELECT rowid, bm25(memory_fts) AS score
                    FROM memory_fts
                    WHERE memory_fts MATCH ?
                    LIMIT {int(k)!r}
                ),
                vec AS (
                    SELECT memory_id, distance
                    FROM vss_nomic_v1_5_index
                    WHERE embedding MATCH ? AND k = {int(k)}
                )
            SELECT
                m.rowid, m.timestamp, m.kind, JSON(m.data), m.importance,
                IFNULL({float(importance)!r:.4} * m.importance, 0) +
                IFNULL({float(recency)!r:.4} * POWER(0.995, ? - m.timestamp), 0) +
                IFNULL({float(fts)!r:.4} *
                    (fts.score - MIN(fts.score) OVER())
                    / (MAX(fts.score) OVER() - MIN(fts.score) OVER()), 0) +
                IFNULL({float(vss)!r:.4} / (1 + vec.distance), 0) AS score
            FROM memories m
                LEFT JOIN fts ON m.rowid = fts.rowid
                LEFT JOIN vec ON m.rowid = vec.memory_id
            ORDER BY score DESC
            LIMIT {int(k)!r}
        """, (prompt, e.tobytes(), timestamp and timestamp))

        for row in cur:
            yield MemoryRow(*row[:-1]), row[-1]
