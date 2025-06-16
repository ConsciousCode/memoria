from typing import Callable, Iterable, Optional, Protocol, cast
import sqlite3
import json
from uuid import UUID

from pydantic import BaseModel

from ipld.cid import CIDv1
from numpy import ndarray
import numpy
import sqlite_vec
from fastembed import TextEmbedding
from uuid_extensions import uuid7

from models import ACThread, Edge, IncompleteACThread, Memory, MemoryDataAdapter, MemoryKind, RecallConfig, build_memory
from util import finite, json_t

nomic_text = TextEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
)

type JSONB = str
'''Alias for JSONB type in SQLite, which is selected as a string.'''

DEFAULT_IMPORTANCE = 0.30
DEFAULT_RECENCY = 0.30
DEFAULT_FTS = 0.15
DEFAULT_VSS = 0.25
DEFAULT_SONA = 0.10
DEFAULT_K = 20

## Rows are represented as the raw output from a select query with no processing
## - this allows us to avoid processing columns we don't need
## PrimaryKey aliases also give us little semantic type hints for the linter

class Factory(Protocol):
    factory: Callable

class FileRow(BaseModel):
    type PrimaryKey = int

    rowid: PrimaryKey
    cid: bytes
    filename: Optional[str]
    mimetype: str
    metadata: Optional[JSONB]
    size: int
    content: Optional[bytes]

    @classmethod
    def factory(cls, rowid, cid, filename, mimetype, metadata, size, content) -> 'FileRow':
        '''Create a FileRow from a raw database row.'''
        return cls(
            rowid=rowid,
            cid=cid,
            filename=filename,
            mimetype=mimetype,
            metadata=metadata,
            size=size,
            content=content
        )

class MemoryRow(BaseModel):
    type PrimaryKey = int

    rowid: PrimaryKey
    cid: Optional[bytes] # None when the memory is incomplete
    timestamp: Optional[float]
    kind: MemoryKind
    data: JSONB
    importance: Optional[float]

    def to_memory(self):
        '''Convert this row to a Memory object.'''
        return build_memory(self.kind, json.loads(self.data), self.timestamp)
    
    @classmethod
    def factory(cls, rowid, cid, timestamp, kind, data, importance) -> 'MemoryRow':
        '''Create a MemoryRow from a raw database row.'''
        return cls(
            rowid=rowid,
            cid=cid,
            timestamp=timestamp,
            kind=kind,
            data=data,
            importance=importance
        )

class EdgeRow(BaseModel):
    src_id: MemoryRow.PrimaryKey
    dst_id: MemoryRow.PrimaryKey
    weight: float

class SonaRow(BaseModel):
    type PrimaryKey = int

    rowid: PrimaryKey
    uuid: bytes
    active_id: Optional['ACThreadRow.PrimaryKey']
    pending_id: Optional['ACThreadRow.PrimaryKey']

    @classmethod
    def factory(cls, rowid, uuid, active_id, pending_id) -> 'SonaRow':
        '''Create a SonaRow from a raw database row.'''
        return cls(
            rowid=rowid,
            uuid=uuid,
            active_id=active_id,
            pending_id=pending_id
        )

class ACThreadRow(BaseModel):
    type PrimaryKey = int

    rowid: PrimaryKey
    cid: Optional[bytes] # Depends on memory CID and previous CID
    sona_id: SonaRow.PrimaryKey
    memory_id: MemoryRow.PrimaryKey
    prev_id: Optional[int]

    @classmethod
    def factory(cls, rowid, cid, sona_id, memory_id, prev_id) -> 'ACThreadRow':
        '''Create an ACThreadRow from a raw database row.'''
        return cls(
            rowid=rowid,
            cid=cid,
            sona_id=sona_id,
            memory_id=memory_id,
            prev_id=prev_id
        )

class BackwardEdge(BaseModel):
    dst: MemoryRow
    weight: float

class ForwardEdge(BaseModel):
    weight: float
    src: MemoryRow

with open("schema.sql", "r") as f:
    SCHEMA = f.read()

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

    def __init__(self, db_path: str=":memory:", file_path: str="files"):
        self.db_path = db_path
        self.file_path = file_path

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
    
    def cursor[T](self, factory: Optional[Callable|Factory]=None):
        cur = self.conn.cursor()
        if factory:
            if f := getattr(factory, 'factory', None):
                # If factory is a class with a factory method
                cur.row_factory = lambda cur, row: row and f(*row)
            else:
                cur.row_factory = lambda cur, row: row and factory(*row) # type: ignore
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
    
    def finalize_memory(self, rowid: int) -> Memory:
        '''
        Finalize a memory by setting its CID and returning the memory object.
        This is used after all edges have been linked to the memory.
        '''
        if (memory := self.select_memory_rowid(rowid)) is None:
            raise ValueError(f"Memory with rowid {rowid} does not exist.")
        
        if memory.cid is not None:
            raise ValueError(f"Memory with rowid {rowid} already has a CID: {memory.cid}")
        
        # Build the CID from the memory data
        edges: dict[CIDv1, float] = {}
        for edge in self.backward_edges(rowid):
            if cid := edge.dst.cid:
                edges[CIDv1(cid)] = edge.weight
            else:
                raise ValueError(
                    f"Memory with rowid {rowid} has an edge to a memory without a CID: {edge.dst.rowid}"
                )
        
        memory = build_memory(memory.kind, memory.data, memory.timestamp, edges)
        cur = self.cursor()
        cur.execute("""
            UPDATE memories SET cid = ? WHERE rowid = ?
        """, (memory.cid.buffer, rowid))
        return memory
    
    def finalize_act(self, rowid: int) -> ACThread:
        '''Finalize an ACT by setting its CID and returning it.'''

        cur = self.cursor()
        cur.execute("""
            SELECT s.cid, m.cid, p.cid
            FROM acthreads act
                JOIN sonas s ON s.rowid = act.sona_id
                JOIN memories m ON m.rowid = act.memory_id
                LEFT JOIN acthreads p ON p.rowid = act.prev_id
            WHERE act.rowid = ?
        """, (rowid,))
        if (row := cur.fetchone()) is None:
            raise ValueError(f"ACT with rowid {rowid} does not exist.")
        
        s, m, p = row

        act = ACThread(
            sona=CIDv1(s),
            memory=CIDv1(m),
            prev=p and CIDv1(p)
        )

        cur.execute("""
            UPDATE acthreads SET cid = ?
            WHERE rowid = ?
        """, (act.cid.buffer, rowid))

        return act
    
    def lookup_ipld_memory(self, cid: CIDv1) -> Optional[Memory]:
        '''Lookup an IPLD memory object by CID, returning its IPLD model.'''
        cur = self.cursor(MemoryRow.factory)
        mem: Optional[MemoryRow] = cur.execute("""
            SELECT cid, timestamp, kind, data, importance
            FROM memories
            WHERE cid = ?
        """, (cid,)).fetchone()
        if mem is None:
            return None
        
        cur = self.cursor(EdgeRow)
        cur.execute("""
            SELECT dst.cid, weight
            FROM edges
                JOIN memories dst ON dst.rowid = edges.dst_id
            WHERE src_id = ?
        """, (mem.rowid,))

        m = build_memory(mem.kind, mem.data, mem.timestamp, {
            CIDv1(dst): weight
                for dst, weight in cur
        })
        if m.cid != cid:
            raise ValueError(f"Memory CID {m.cid} does not match requested CID {cid}")
        
        return m
    
    def lookup_ipld_act(self, cid: CIDv1) -> Optional[ACThread]:
        '''Lookup an IPLD act thread by CID.'''
        cur = self.cursor()
        row = cur.execute("""pmem.rowid
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

    def lookup_ipld(self, cid: CIDv1):
        '''
        Lookup a CID in the database, returning the associated row if it exists.
        '''
        return (
            self.lookup_ipld_memory(cid)
            or self.lookup_ipld_act(cid)
            #or self.lookup_file(cid)
            #or self.lookup_sona(cid)
        )

    def find_sona(self, name: UUID|str):
        '''Find or create the sona closest to the given name.'''

        cur = self.cursor()

        if isinstance(name, UUID):
            cur.execute("""
                SELECT rowid, uuid, active_id, pending_id
                FROM sonas
                WHERE uuid = ?
            """, (name.bytes,))

            if row := cur.fetchone():
                rowid, uuid, active_id, pending_id = row
                return SonaRow(
                    rowid=rowid,
                    uuid=uuid,
                    active_id=active_id,
                    pending_id=pending_id
                )
            else:
                return None
        
        cur.execute("""
            SELECT rowid, uuid, active_id, pending_id
            FROM sona_aliases
                JOIN sonas ON sonas.rowid = sona_aliases.sona_id
            WHERE name = ?
        """, (name,))
        
        if row := cur.fetchone():
            rowid, uuid, active_id, pending_id = row
            return SonaRow(
                rowid=rowid,
                uuid=uuid,
                active_id=active_id,
                pending_id=pending_id
            )

        e, = nomic_text.embed(name)
        ebs = e.tobytes()
        cur.execute("""
            SELECT rowid, uuid, active_id, pending_id
            FROM sona_vss
                JOIN sonas ON sonas.rowid = sona_vss.sona_id
            WHERE embedding MATCH ? AND distance > 0.75 AND k = 1
        """, (ebs,))
        if row := cur.fetchone():
            cur.execute("""
                INSERT INTO sona_aliases (sona_id, name)
                VALUES (?, ?)
            """, (row[0], name))
            return SonaRow.factory(*row)
        
        # Doesn't exist, create a new one.
        
        u = cast(UUID, uuid7())
        cur.execute("INSERT INTO sonas (uuid) VALUES (?)", (u.bytes,))
        if (rowid := cur.lastrowid) is None:
            raise RuntimeError("Failed to insert sona")

        # Don't need to deduplicate because we just created it and already
        # know the embedding is far from any existing embedding.
        cur.execute("""
            INSERT INTO sona_vss (sona_id, embedding) VALUES (?, ?)
        """, (rowid, ebs))
        cur.execute("""
            INSERT INTO sona_aliases (sona_id, name) VALUES (?, ?)
        """, (rowid, name))
        
        return SonaRow.factory(rowid, u.bytes, None, None)

    def select_sona_aliases(self, rowid: int) -> list[str]:
        '''Select all aliases for a sona by its rowid.'''
        cur = self.cursor()
        rows = cur.execute("""
            SELECT name FROM sona_aliases WHERE sona_id = ?
        """, (rowid,))
        return [name for (name,) in rows]

    def select_act(self, rowid: int) -> Optional[ACThreadRow]:
        '''Select an act thread by its rowid.'''
        cur = self.cursor(ACThreadRow)
        return cur.execute("""
            SELECT rowid, cid, sona_id, memory_id, prev_id
            FROM acthreads WHERE rowid = ?
        """, (rowid,)).fetchone()

    def select_fts_rowid(self, rowid: int) -> list[str]:
        cur = self.cursor()
        rows = cur.execute("""
            SELECT content FROM memory_fts WHERE rowid = ?
        """, (rowid,))
        return [content for (content,) in rows]

    def get_act_active(self, sona_id: int) -> Optional[ACThreadRow]:
        '''Get the sona's active thread node currently receiving updates.'''
        cur = self.cursor(ACThreadRow)
        return cur.execute("""
            SELECT act.rowid, cid, sona_id, memory_id, prev_id
            FROM sonas s
            JOIN acthreads act ON act.rowid = s.active_id
            WHERE s.rowid = ?
        """, (sona_id,)).fetchone()

    def get_act_pending(self, sona_id: int) -> Optional[ACThreadRow]:
        '''Get the sona's pending thread node which is aggregating requests.'''
        cur = self.cursor(ACThreadRow)
        return cur.execute("""
            SELECT act.rowid, cid, sona_id, memory_id, prev_id
            FROM sonas s
            JOIN acthreads act ON act.rowid = s.pending_id
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

    def ipld_memory_rowid(self, rowid: int) -> Optional[Memory]:
        '''
        Build a Memory object from a memory rowid. This will also fetch the
        edges for the memory.
        '''
        if (memory := self.select_memory_rowid(rowid)) is None:
            return None
        
        cur = self.cursor()
        cur.execute("""
            SELECT dst.cid, weight
            FROM edges
                JOIN memories dst ON dst.rowid = edges.dst_id
            WHERE src_id = ?
        """, (rowid,))
        
        return build_memory(memory.kind, memory.data, memory.timestamp, {
            CIDv1(cid): weight
                for cid, weight in cur
        })

    def get_incomplete_act(self, rowid: int) -> Optional[IncompleteACThread]:
        '''Get the act thread for a sona and memory.'''
        cur = self.cursor()
        row = cur.execute("""
            SELECT m.kind, m.data, m.timestamp, a2.cid
            FROM acthreads a1
                LEFT JOIN acthreads a2 ON a2.rowid = a1.prev_id
                LEFT JOIN memories m ON m.rowid = a1.memory_id
            WHERE sona_id = ? AND memory_id = ?
        """, (rowid,)).fetchone()

        if row is None:
            return None
        
        kind, data, timestamp, prev = row
        edges: dict[CIDv1, float] = {}
        for edge in self.backward_edges(rowid):
            if cid := edge.dst.cid:
                edges[CIDv1(cid)] = edge.weight
            else:
                raise ValueError(
                    f"Memory with rowid {rowid} has an edge to an incomplete memory: {edge.dst.rowid}"
                )

        return IncompleteACThread(
            memory=build_memory(
                kind, MemoryDataAdapter.validate_json(data), timestamp, edges
            ),
            prev=CIDv1(prev)
        )

    def select_embedding(self, memory_id: int) -> Optional[ndarray]:
        cur = self.cursor()
        cur.execute("""
            SELECT embedding FROM memory_vss WHERE memory_id = ?
        """, (memory_id,))
        if row := cur.fetchone():
            return numpy.frombuffer(row[0], dtype=numpy.float32)
        return None
    
    def get_active_memory(self, sona_id: int) -> list[MemoryRow]:
        '''Get the active memory for a sona.'''
        cur = self.cursor(EdgeRow)
        return cur.execute("""
            SELECT m.rowid, m.cid, m.timestamp, m.kind, JSON(m.data), m.importance
            FROM sonas s
            JOIN acthreads act ON act.rowid = s.active_id
            JOIN memories m ON m.rowid = act.memory_id
            WHERE s.rowid = ?
        """, (sona_id,)).fetchall()

    def insert_text_embedding(self, memory_id: int, index: str):
        '''Insert a text embedding for a memory.'''
        e, = nomic_text.embed(index)
        cur = self.cursor()
        # Deduplicate because sqlite-vec can't.
        cur.execute("""
            SELECT rowid FROM memory_vss
            WHERE embedding = ?
        """, (e,))
        if cur.fetchone():
            return
        # Insert the embedding
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
        c = cid and cid.buffer
        cur = self.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO memories
            (cid, timestamp, kind, data, importance)
            VALUES (?, ?, ?, JSONB(?), ?)
        """, (c, timestamp, kind, json.dumps(data), importance))
        
        if not (rowid := cur.lastrowid):
            # Memory already exists
            rowid, = cur.execute("""
                SELECT rowid FROM memories WHERE cid = ?
            """, (c,)).fetchone()
        
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
    
    def sona_stage_active(self, sona_id: int):
        '''Stage pending thread as the active thread for a sona.'''
        cur = self.cursor()
        cur.execute("""
            UPDATE sonas SET active_id = pending_id, pending_id = NULL
            WHERE rowid = ?
        """, (sona_id,))

    def update_memory_data(self, rowid: int, data: str):
        '''
        Insert the data for a memory. This is used for file memories and other
        large data blobs.
        '''
        cur = self.cursor()
        cur.execute("""
            UPDATE memories SET data = JSONB(?)
            WHERE rowid = ?
        """, (data, rowid))

    def update_sona_active(self, sona_id: int, thread_id: Optional[int]):
        '''
        Update the active thread for a sona. This is the thread that is currently
        receiving updates.
        '''
        cur = self.cursor()
        cur.execute("""
            UPDATE sonas SET active_id = ?
            WHERE rowid = ?
        """, (thread_id, sona_id))

    def update_sona_pending(self, sona_id: int, thread_id: Optional[int]):
        '''
        Update the pending thread for a sona. This is the thread that is currently
        receiving requests.
        '''
        cur = self.cursor()
        cur.execute("""
            UPDATE sonas SET pending_id = ?
            WHERE rowid = ?
        """, (thread_id, sona_id))

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

    def link_memory_edges(self, rowid: int, edges: list[Edge]):
        '''
        Link the edges of a memory to the database. This is used when inserting
        a memory with edges that are already in the database.
        '''
        cur = self.cursor()
        cur.executemany("""
            INSERT OR IGNORE INTO edges (src_id, dst_id, weight)
            SELECT ?, rowid, ?
            FROM memories m WHERE cid = ?
        """, (
            (rowid, e.weight, e.target.buffer)
                for e in edges
        ))
    
    def link_sona(self, sona_id: int, memory_id: int):
        '''Link a memory to a sona.'''
        print(f"link_sona({sona_id=}, {memory_id=})")
        cur = self.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO sona_memories
            (sona_id, memory_id) VALUES (?, ?)
        """, (sona_id, memory_id))
    
    # dst <- src

    def backward_edges(self, src_id: int) -> Iterable[BackwardEdge]:
        '''
        Get all edges leading to the given memory, returning the source id
        and weight of the edge.
        '''
        cur = self.cursor()
        cur.execute("""
            SELECT
                m.rowid, m.cid, m.timestamp, m.kind, JSON(m.data), m.importance,
                e.weight
            FROM edges e JOIN memories m ON m.rowid = e.dst_id
            WHERE e.src_id = ?
            ORDER BY e.weight DESC
        """, (src_id,))
        
        for row in cur:
            yield BackwardEdge(
                dst=MemoryRow.factory(*row[:6]),
                weight=row[6]
            )
    
    def forward_edges(self, dst_id: int) -> Iterable[ForwardEdge]:
        '''
        Get all edges leading from the given memory, returning the destination id
        and weight of the edge.
        '''
        cur = self.cursor()
        cur.execute("""
            SELECT
                m.rowid, m.cid, m.timestamp, m.kind, JSON(m.data), m.importance,
                e.weight
            FROM edges e JOIN memories m ON m.rowid = e.src_id
            WHERE e.dst_id = ?
            ORDER BY m.importance DESC
        """, (dst_id,))
        
        for row in cur:
            yield ForwardEdge(
                weight=row[6],
                src=MemoryRow.factory(*row[:6])
            )

    def recall(self,
            sona: Optional[str],
            prompt: Optional[str],
            timestamp: Optional[float]=None,
            config: Optional[RecallConfig]=None
        ) -> Iterable[tuple[MemoryRow, float]]:
        config = config or RecallConfig()
        
        if prompt:
            e, = nomic_text.embed(prompt)
        else:
            config.vss = 0
            e = numpy.zeros(1536, dtype=numpy.float32)

        if sona:
            s, = nomic_text.embed(sona)
        else:
            config.sona = 0
            s = numpy.zeros(1536, dtype=numpy.float32)
        
        # Be defensive against SQL injection
        k = int(config.k or DEFAULT_K)

        ### DANGER ZONE ###
        # DO NOT SUFFER A BARE INTERPOLATION HERE - EVERY SINGLE INTERPOLATION
        # ****MUST**** BE WRAPPED IN A PYTHON PRIMITIVE TYPE CONSTRUCTOR
        # *AND* A REPR formatting function
        cur = self.cursor()
        cur.execute(f"""
            WITH
                fts AS (
                    SELECT rowid, bm25(memory_fts) AS score
                    FROM memory_fts
                    WHERE memory_fts MATCH :prompt
                    LIMIT {int(k)!r}
                ),
                mvss AS (
                    SELECT memory_id, distance
                    FROM memory_vss
                    WHERE embedding MATCH :vss_e AND k = {int(k)!r}
                ),
                svss AS (
                    SELECT sona_id, distance
                    FROM sona_vss
                    WHERE embedding MATCH :sona_e AND k = {int(k)!r}
                )
            SELECT
                m.rowid, m.cid, m.timestamp, m.kind, JSON(m.data), m.importance,
                (
                    IFNULL(:w_importance * m.importance, 0) +
                    IFNULL(:w_recency * POWER(0.995, :timestamp - m.timestamp), 0) +
                    IFNULL(:w_fts *
                        (fts.score - MIN(fts.score) OVER())
                        / (MAX(fts.score) OVER() - MIN(fts.score) OVER()), 0) +
                    IFNULL(:w_vss / (1 + mvss.distance), 0) +
                    IFNULL(:w_sona / (1 + svss.distance), 0)
                ) / (:w_importance + :w_recency + :w_fts + :w_vss + :w_sona) AS score
            FROM memories m
                LEFT JOIN sona_memories sm ON sm.memory_id = m.rowid
                LEFT JOIN fts ON fts.rowid = m.rowid
                LEFT JOIN mvss ON mvss.memory_id = m.rowid
                LEFT JOIN svss ON svss.sona_id = sm.sona_id
            ORDER BY score DESC
            LIMIT {int(k)!r}
        """, {
            "prompt": prompt,
            "vss_e": e.tobytes(),
            "sona_e": s.tobytes(),
            "timestamp": timestamp,
            "w_importance": finite(config.importance or DEFAULT_IMPORTANCE),
            "w_recency": finite(config.recency or DEFAULT_RECENCY),
            "w_fts": finite(config.fts or DEFAULT_FTS),
            "w_vss": finite(config.vss or DEFAULT_VSS),
            "w_sona": finite(config.sona or DEFAULT_SONA)
        })

        for row in cur:
            yield MemoryRow.factory(*row[:-1]), row[-1]
