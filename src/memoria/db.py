from contextlib import contextmanager
from typing import Callable, Iterable, Optional, Protocol, cast, overload
import sqlite3
from uuid import UUID
import os

from pydantic import BaseModel
from numpy import ndarray
import numpy
import sqlite_vec
from fastembed import TextEmbedding
from uuid_extensions import uuid7

from src.ipld import CIDv1
from src.models import ACThread, AnyACThread, AnyMemory, Edge, IncompleteACThread, IncompleteMemory, DraftMemory, Memory, MemoryKind, PartialMemory, RecallConfig
from src.util import finite

__all__ = (
    'FileRow',
    'MemoryRow', 'IncompleteMemoryRow', 'AnyMemoryRow',
    'EdgeRow',
    'SonaRow',
    'ACThreadRow', 'IncompleteACThreadRow', 'AnyACThreadRow',
    'Database'
)

with open(os.path.join(os.path.dirname(__file__), "schema.sql"), "r") as f:
    SCHEMA = f.read()

nomic_text = TextEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    cache_dir="private/local_cache"
)

type JSONB = str
'''Alias for JSONB type in SQLite, which is selected as a string.'''

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

class BaseMemoryRow(BaseModel):
    '''
    Split MemoryRow into complete and incomplete varieties for type-level
    guarantees about whether a memory has a CID.
    '''
    type PrimaryKey = int

    rowid: PrimaryKey
    timestamp: Optional[int]
    kind: MemoryKind
    data: JSONB
    importance: Optional[float]

    @classmethod
    def factory(cls, rowid, cid, timestamp, kind, data, importance) -> 'AnyMemoryRow':
        '''Create a MemoryRow from a raw database row.'''
        return (MemoryRow if cid else IncompleteMemoryRow).factory(
            rowid, cid, timestamp, kind, data, importance
        )

class MemoryRow(BaseMemoryRow):
    '''A completed memory, has a CID.'''
    cid: CIDv1

    @classmethod
    def factory(cls, rowid, cid, timestamp, kind, data, importance) -> 'MemoryRow':
        '''Create a MemoryRow from a raw database row.'''
        if timestamp and not timestamp.is_integer():
            raise TypeError("Timestamp must be an integer.")
        return MemoryRow(
            rowid=rowid,
            cid=cid and CIDv1(cid),
            timestamp=timestamp,
            kind=kind,
            data=data,
            importance=importance
        )
    
    def to_mutable(self, edges: list[Edge[CIDv1]] = []) -> DraftMemory:
        '''Convert this row to a MaybeMemory object.'''
        if self.cid:
            return self.to_partial(edges)
        else:
            return self.to_incomplete(edges)
    
    def to_incomplete(self, edges: list[Edge[CIDv1]] = []):
        '''Convert this row to an IncompleteMemory object.'''
        return IncompleteMemory(
            data=IncompleteMemory.build_data(self.kind, self.data),
            timestamp=self.timestamp,
            edges=edges
        )
    
    def to_partial(self, edges: list[Edge[CIDv1]] = []):
        '''Convert this row to a Memory object without finalizing it.'''
        return PartialMemory(
            cid=self.cid,
            data=Memory.build_data(self.kind, self.data),
            timestamp=self.timestamp,
            edges=edges
        )
    
    def to_memory(self, edges: list[Edge[CIDv1]]):
        '''Convert this row to a Memory object.'''
        return Memory(
            data=Memory.build_data(self.kind, self.data),
            timestamp=self.timestamp,
            edges=edges
        )

class IncompleteMemoryRow(BaseMemoryRow):
    '''An incomplete memory, does not have a CID.'''
    cid: None = None

    @classmethod
    def factory(cls, rowid, cid, timestamp, kind, data, importance) -> 'IncompleteMemoryRow':
        '''Create an IncompleteMemoryRow from a raw database row.'''
        return IncompleteMemoryRow(
            rowid=rowid,
            cid=cid,
            timestamp=timestamp,
            kind=kind,
            data=data,
            importance=importance
        )
    
    def to_maybe(self, edges: list[Edge[CIDv1]] = []) -> DraftMemory:
        '''Convert this row to a MaybeMemory object.'''
        return self.to_incomplete()
    
    def to_incomplete(self, edges: list[Edge[CIDv1]] = []):
        '''Convert this row to an IncompleteMemory object.'''
        return IncompleteMemory(
            data=IncompleteMemory.build_data(self.kind, self.data),
            timestamp=self.timestamp,
            edges=edges
        )

    def to_memory(self, edges: list[Edge[CIDv1]]):
        return IncompleteMemory(
            data=IncompleteMemory.build_data(self.kind, self.data),
            timestamp=self.timestamp,
            edges=edges
        )

type AnyMemoryRow = MemoryRow | IncompleteMemoryRow

class EdgeRow(BaseModel):
    src_id: MemoryRow.PrimaryKey
    dst_id: MemoryRow.PrimaryKey
    weight: float

    @classmethod
    def factory(cls, src_id, dst_id, weight) -> 'EdgeRow':
        '''Create an EdgeRow from a raw database row.'''
        return cls(
            src_id=src_id,
            dst_id=dst_id,
            weight=weight
        )

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

class BaseACThreadRow(BaseModel):
    '''
    Split ACThreadRow into complete and incomplete varieties for type-level
    guarantees about whether an ACT has a CID.
    '''
    type PrimaryKey = int

    rowid: PrimaryKey
    sona_id: SonaRow.PrimaryKey
    memory_id: BaseMemoryRow.PrimaryKey
    prev_id: Optional[int]

    @classmethod
    def factory(cls, rowid, cid, sona_id, memory_id, prev_id) -> 'AnyACThreadRow':
        '''Create an ACThreadRow from a raw database row.'''
        return (ACThreadRow if cid else IncompleteACThreadRow).factory(
            rowid, cid, sona_id, memory_id, prev_id
        )

class ACThreadRow(BaseACThreadRow):
    '''A completed act thread, has a CID.'''
    cid: CIDv1

    @classmethod
    def factory(cls, rowid, cid, sona_id, memory_id, prev_id) -> 'ACThreadRow':
        '''Create an ACThreadRow from a raw database row.'''
        return ACThreadRow(
            rowid=rowid,
            cid=CIDv1(cid),
            sona_id=sona_id,
            memory_id=memory_id,
            prev_id=prev_id
        )

class IncompleteACThreadRow(BaseACThreadRow):
    '''An incomplete act thread, does not have a CID.'''
    cid: None = None

    @classmethod
    def factory(cls, rowid, cid, sona_id, memory_id, prev_id) -> 'IncompleteACThreadRow':
        '''Create an IncompleteACThread from a raw database row.'''
        return IncompleteACThreadRow(
            rowid=rowid,
            cid=cid,
            sona_id=sona_id,
            memory_id=memory_id,
            prev_id=prev_id
        )

type AnyACThreadRow = ACThreadRow | IncompleteACThreadRow

class CancelTransaction(Exception):
    '''
    Exception to raise to cancel a transaction. This will cause the transaction
    to be rolled back and the database to be closed.
    '''

class Database:
    '''
    All SQL queries are contained within this database class. This prevents the
    proliferation of SQL queries throughout the codebase, allowing for easier
    maintenance and updates.

    However, these queries do not respect data integrity on their own - they
    must be used correctly to ensure that the database remains consistent. For
    instance, memories are inserted separately from their edges, but memories
    themselves have a `cid` which depends on those edges.

    By default the Database only supports read operations to provide interface-
    level safety wrt mutations outside of transactions. Write operations are
    only exposed through the `DatabaseRW` subclass provided in the`transaction`
    context manager.
    '''

    def __init__(self, db_path: str=":memory:", file_path: str="files"):
        self.db_path = db_path
        self.file_path = file_path
        print("Hello world")

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
    
    def cursor(self, factory: Optional[Factory|Callable]=None):
        cur = self.conn.cursor()
        if factory:
            if f := getattr(factory, "factory", None):
                cur.row_factory = lambda cur, row: row and f(*row)
            else:
                cur.row_factory = lambda cur, row: row and factory(*row) # type: ignore
        return cur
    
    def commit(self): self.conn.commit()
    def rollback(self): self.conn.rollback()
    def close(self): self.conn.close()
    
    @contextmanager
    def transaction(self):
        '''
        Context manager for a transaction. This will automatically commit or
        rollback the transaction depending on whether an exception was raised.
        '''
        try:
            yield DatabaseRW._construct(self)
            self.commit()
        except CancelTransaction:
            self.rollback()
        except:
            self.rollback()
            raise
    
    def has_cid(self, cid: CIDv1) -> bool:
        '''Check if the database has a CID.'''
        cur = self.cursor()
        cur.execute("""
            SELECT 1 FROM memories WHERE cid = ?
        """, (cid.buffer,))
        if cur.fetchone():
            return True
        
        cur.execute("""
            SELECT 1 FROM acthreads WHERE cid = ?
        """, (cid.buffer,))
        if cur.fetchone():
            return True
        
        return False
    
    def lookup_ipld_memory(self, cid: CIDv1) -> Optional[Memory]:
        '''Lookup an IPLD memory object by CID, returning its IPLD model.'''
        cur = self.cursor(MemoryRow)
        mem: Optional[MemoryRow] = cur.execute("""
            SELECT rowid, cid, timestamp, kind, JSON(data), importance
            FROM memories
            WHERE cid = ?
        """, (cid.buffer,)).fetchone()
        if mem is None:
            return None
        
        cur = self.cursor()
        cur.execute("""
            SELECT dst.cid, weight
            FROM edges
                JOIN memories dst ON dst.rowid = edges.dst_id
            WHERE src_id = ?
        """, (mem.rowid,))

        m = mem.to_memory([
            Edge(target=CIDv1(dst), weight=weight)
                for dst, weight in cur
        ])
        if m.cid != cid:
            raise ValueError(f"Memory CID {m.cid} does not match requested CID {cid}")
        
        return m
    
    def lookup_ipld_act(self, cid: CIDv1) -> Optional[ACThread]:
        '''Lookup an IPLD act thread by CID.'''
        cur = self.cursor()
        row = cur.execute("""
            SELECT s.uuid, m.cid, p.cid
            FROM acthreads act
                JOIN sonas s ON s.rowid = act.sona_id
                JOIN memories m ON m.rowid = act.memory_id
                JOIN acthreads p ON p.rowid = act.prev_id
            WHERE act.cid = ?
        """, (cid.buffer,)).fetchone()
        if row is None:
            return None
        
        s, m, p = row
        t = ACThread(
            sona=UUID(s),
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

    def select_sona_aliases(self, rowid: int) -> list[str]:
        '''Select all aliases for a sona by its rowid.'''
        cur = self.cursor()
        rows = cur.execute("""
            SELECT name FROM sona_aliases WHERE sona_id = ?
        """, (rowid,))
        return [name for (name,) in rows]
    
    def select_act(self, rowid: int) -> Optional[AnyACThread]:
        '''Select an act thread by its rowid.'''
        cur = self.cursor(ACThreadRow)
        row: AnyACThreadRow = cur.execute("""
            SELECT rowid, cid, sona_id, memory_id, prev_id
            FROM acthreads WHERE rowid = ?
        """, (rowid,)).fetchone()
        if row is None:
            return None
        
        sona = self.select_sona_rowid(row.sona_id)
        if sona is None:
            raise ValueError(f"Sona with rowid {row.sona_id} does not exist.")
        
        memory = self.select_memory_rowid(row.memory_id)
        if memory is None:
            raise ValueError(f"Memory with rowid {row.memory_id} does not exist.")

        prev = row.prev_id
        if prev is not None:
            prev = self.select_memory_rowid(prev)

        if row.cid is None:
            return IncompleteACThread(
                sona=UUID(bytes=sona.uuid),
                memory=memory.to_incomplete(),
                prev=prev and prev.cid
            )
        
        return ACThread(
            sona=UUID(bytes=sona.uuid),
            memory=CIDv1(row.cid),
            prev=prev and prev.cid
        )

    def select_act_row(self, rowid: int) -> Optional[AnyACThreadRow]:
        '''Select an act thread by its rowid.'''
        cur = self.cursor()
        row = cur.execute("""
            SELECT rowid, cid, sona_id, memory_id, prev_id
            FROM acthreads WHERE rowid = ?
        """, (rowid,)).fetchone()
        if row is None:
            return None
        
        if row[1] is None:
            return IncompleteACThreadRow.factory(*row)
        else:
            return ACThreadRow.factory(*row)

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
    
    def select_memory_rowid(self, rowid: int) -> Optional[AnyMemoryRow]:
        cur = self.cursor(BaseMemoryRow)
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

    def select_sona_rowid(self, rowid: int) -> Optional[SonaRow]:
        '''Select a sona by its rowid.'''
        cur = self.cursor(SonaRow)
        return cur.execute("""
            SELECT rowid, uuid, active_id, pending_id
            FROM sonas WHERE rowid = ?
        """, (rowid,)).fetchone()
    
    def select_sona_uuid(self, uuid: UUID) -> Optional[SonaRow]:
        '''Select a sona by its UUID.'''
        cur = self.cursor(SonaRow)
        return cur.execute("""
            SELECT rowid, uuid, active_id, pending_id
            FROM sonas WHERE uuid = ?
        """, (uuid.bytes,)).fetchone()

    def ipld_memory_rowid(self, rowid: int) -> Optional[AnyMemory]:
        '''
        Build a Memory object from a memory rowid. This will also fetch the
        edges for the memory.
        '''
        if (mr := self.select_memory_rowid(rowid)) is None:
            return None
        
        cur = self.cursor()
        cur.execute("""
            SELECT dst.cid, weight
            FROM edges
                JOIN memories dst ON dst.rowid = edges.dst_id
            WHERE src_id = ?
        """, (rowid,))
        
        return mr.to_memory([
            Edge(target=CIDv1(dst), weight=weight)
                for dst, weight in cur
        ])

    def get_incomplete_act(self, rowid: int) -> Optional[IncompleteACThread]:
        '''Get the act thread for a sona and memory.'''
        cur = self.cursor()
        row = cur.execute("""
            SELECT
                a1.rowid, a1.cid, -- IncompleteACThread
                m.timestamp, m.kind, m.data, -- IncompleteMemory
                s.cid, a2.cid -- {Sona, Previous} CID
            FROM acthreads a1
                JOIN acthreads a2 ON a1.prev_id = a2.rowid
                JOIN memories m ON a1.memory_id = m.rowid
                JOIN sona s ON a1.sona_id = s.rowid
            WHERE memory_id = ?
        """, (rowid,)).fetchone()

        if row is None:
            return None
        
        (
            a1_rowid, a1_cid, # IncompleteACThread
            m_timestamp, m_kind, m_data, # IncompleteMemory
            s_cid, a2_cid # {Sona, Previous} CID
        ) = row
        # Just to double-check that this is actually an incomplete ACT
        if a1_cid is not None:
            raise ValueError(f"ACT with rowid {a1_rowid} already has a CID: {a1_cid}")
        return IncompleteACThread(
            sona=UUID(bytes=s_cid),
            memory=IncompleteMemory(
                data=IncompleteMemory.build_data(m_kind, m_data),
                timestamp=m_timestamp,
                edges=[
                    Edge(target=e.target.cid, weight=e.weight)
                        for e in self.backward_edges(rowid)
                ]
            ),
            prev=a2_cid and CIDv1(a2_cid)
        )
    
    def list_memory_sonas(self, memory_id: int) -> Iterable[SonaRow]:
        '''List all sonas that have this memory in their active or pending threads.'''
        cur = self.cursor(SonaRow)
        cur.execute("""
            SELECT s.rowid, s.uuid, s.active_id, s.pending_id
            FROM memories m
                LEFT JOIN sona_memories sm ON m.rowid = sm.memory_id
                JOIN sonas s ON sm.sona_id = s.rowid
            WHERE m.rowid = ?
        """, (memory_id,))
        return cur

    def select_embedding(self, memory_id: int) -> Optional[ndarray]:
        cur = self.cursor()
        cur.execute("""
            SELECT embedding FROM memory_vss WHERE memory_id = ?
        """, (memory_id,))
        if row := cur.fetchone():
            return numpy.frombuffer(row[0], dtype=numpy.float32)
        return None
    
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

    def backward_edges(self, src_id: int) -> Iterable[Edge[MemoryRow]]:
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
            yield Edge(
                target=MemoryRow.factory(*row[:6]),
                weight=row[6]
            )

    def backward_edges_cid(self, cid: CIDv1) -> Iterable[Edge[MemoryRow]]:
        '''
        Get all edges leading to the given memory, returning the source id
        and weight of the edge.
        '''
        cur = self.cursor()
        cur.execute("""
            SELECT
                m2.rowid, m2.cid, m2.timestamp, m2.kind,
                    JSON(m2.data), m2.importance,
                e.weight
            FROM memories m1
                JOIN edges e ON e.src_id = m1.rowid
                JOIN memories m2 ON e.dst_id = m2.rowid
            WHERE m1.cid = ?
            ORDER BY e.weight DESC
        """, (cid.buffer,))
        
        for row in cur:
            yield Edge(
                target=MemoryRow.factory(*row[:6]),
                weight=row[6]
            )
    
    def forward_edges(self, dst_id: int) -> Iterable[Edge[AnyMemoryRow]]:
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
            which = MemoryRow if row[1] else IncompleteMemoryRow
            yield Edge(
                weight=row[6],
                target=which.factory(*row[:6])
            )
    
    def list_memories(self, page: int, perpage: int) -> Iterable[tuple[int, DraftMemory]]:
        '''List messages in the database, paginated.'''
        cur = self.cursor(MemoryRow)
        cur.execute("""
            SELECT rowid, cid, timestamp, kind, JSON(data), importance
            FROM memories
            ORDER BY rowid DESC
            LIMIT ? OFFSET ?
        """, (perpage, (page - 1) * perpage))

        for mr in cur:
            yield mr.rowid, mr.to_maybe([
                Edge(target=e.target.cid, weight=e.weight)
                    for e in self.backward_edges(mr.rowid)
            ])
    
    def list_sonas(self, page: int, perpage: int) -> Iterable[SonaRow]:
        '''List sonas in the database, paginated.'''
        cur = self.cursor(SonaRow)
        return cur.execute("""
            SELECT rowid, uuid, active_id, pending_id
            FROM sonas
            LIMIT ? OFFSET ?
        """, (perpage, (page - 1) * perpage))

class DatabaseRW(Database):
    '''Provides mutating operations for the database.'''

    @classmethod
    def _construct(cls, db: Database):
        self = DatabaseRW.__new__(DatabaseRW)
        self.__dict__ = db.__dict__
        return self

    def file_lookup(self, mh: str, ext: str):
        fn, x, yz, rest = mh[:2], mh[2], mh[3:5], mh[5:]
        return f"{self.file_path}/{fn}/{x}/{yz}/{rest}{ext}"
    
    def index(self, memory_id: int, index: str):
        '''Index a memory with a text embedding.'''
        self.insert_text_embedding(memory_id, index)
        self.insert_text_fts(memory_id, index)
    
    def update_invalid(self) -> bool:
        '''
        One iteration of updating invalid memories in the database. Returns
        whether any invalid memories were found. For a complete update,
        `while update_invalid(): commit()` will clear everything.
        '''
        cur = self.cursor()
        cur.execute("""
            SELECT im1.memory_id
            FROM invalid_memories im1
            WHERE NOT EXISTS (
                SELECT 1
                FROM edges e
                    INNER JOIN invalid_memories im2
                        ON e.dst_id = im2.memory_id
                WHERE e.src_id = im1.memory_id
            )
        """)
        mids: list[tuple[int]] = cur.fetchall()
        if not mids:
            return False
        
        cur.executemany("""
            UPDATE memories SET cid = ?
            WHERE rowid = ?
        """, (
            (cid.buffer, rowid)
            for (rowid,) in mids
                if (m := self.select_memory_rowid(rowid))
                    if (cid := m.cid)
        ))
        cur.executemany("""
            DELETE FROM invalid_memories WHERE memory_id = ?
        """, mids)

        return True
    
    def find_sona_embedding(self, sona: UUID|str) -> Optional[ndarray]:
        '''Select the embedding for a sona by its UUID.'''
        if isinstance(sona, str):
            sona = UUID(bytes=self.find_sona(sona).uuid)
        
        cur = self.cursor()
        cur.execute("""
            SELECT embedding
            FROM sona
                JOIN sona_vss ON sona.rowid = sona_vss.sona_id
            WHERE sona.rowid = ?
        """, (sona,))
        if row := cur.fetchone():
            return numpy.frombuffer(row[0], dtype=numpy.float32)
        return None

    def recall(self,
            prompt: DraftMemory,
            config: Optional[RecallConfig]=None
        ) -> Iterable[tuple[MemoryRow, float]]:
        config = config or RecallConfig()
        
        index = prompt.document()
        e, = nomic_text.embed(index)
        s = numpy.zeros(768, dtype=numpy.float32)
        if ps := prompt.sonas:
            for sona in ps:
                if (se := self.find_sona_embedding(sona)) is None:
                    raise ValueError(f"Sona with rowid {sona} does not exist.")
                s += se
            s /= len(ps)
        else:
            config.sona = 0.0
        
        cur = self.cursor()
        cur.execute("""
            WITH
                fts AS (
                    SELECT rowid, bm25(memory_fts) AS score
                    FROM memory_fts
                    WHERE :index AND memory_fts MATCH :index
                    LIMIT :k
                ),
                mvss AS (
                    SELECT memory_id, distance
                    FROM memory_vss
                    WHERE embedding MATCH :vss_e AND k = :k
                ),
                svss AS (
                    SELECT sona_id, distance
                    FROM sona_vss
                    WHERE embedding MATCH :sona_e AND k = :k
                )
            SELECT
                m.rowid, m.cid, m.timestamp, m.kind, JSON(m.data), m.importance,
                (
                    IFNULL(:w_importance * m.importance, 0) +
                    IFNULL(:w_recency * POWER(
                        :timestamp - m.timestamp,
                        -:decay * IFNULL(EXP(-POWER(m.importance, 2)), 1)
                    ), 0) +
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
            WHERE
                m.cid IS NOT NULL AND ( -- Don't recall incomplete memories
                    :timestamp IS NULL OR
                    m.timestamp <= :timestamp -- Don't recall future memories
                )
            ORDER BY score DESC
            LIMIT :k
        """, {
            "index": index,
            "vss_e": e.astype(numpy.float32),
            "sona_e": s.astype(numpy.float32),
            "timestamp": prompt.timestamp,
            "w_importance": finite(config.importance),
            "w_recency": finite(config.recency),
            "w_fts": finite(config.fts),
            "w_vss": finite(config.vss),
            "w_sona": finite(config.sona),
            "k": int(config.k),
            "decay": finite(config.decay)
        })
        for row in cur:
            yield MemoryRow.factory(*row[:-1]), row[-1]
    
    def finalize_memory(self, rowid: int) -> Memory:
        '''
        Finalize a memory by setting its CID and returning the memory object.
        This is used after all edges have been linked to the memory.
        '''
        if (mr := self.select_memory_rowid(rowid)) is None:
            raise ValueError(f"Memory with rowid {rowid} does not exist.")
        
        if mr.cid is not None:
            raise ValueError(f"Memory with rowid {rowid} already has a CID: {mr.cid}")
        
        # Build the CID from the memory data
        m = mr.to_memory([
            Edge(target=e.target.cid, weight=e.weight)
                for e in self.backward_edges(rowid)
        ]).complete()
        cur = self.cursor()
        cur.execute("""
            UPDATE memories SET cid = ? WHERE rowid = ?
        """, (m.cid.buffer, rowid))
        return m
    
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
            sona=UUID(s),
            memory=CIDv1(m),
            prev=p and CIDv1(p)
        )

        cur.execute("""
            UPDATE acthreads SET cid = ?
            WHERE rowid = ?
        """, (act.cid.buffer, rowid))

        return act
    
    def propagate_importance(self, memory: CIDv1):
        '''
        Propagate the importance of a memory to its edges, updating the edges
        in the database.
        '''
        cur = self.cursor()
        cur.execute("""
            UPDATE memories AS m2
                SET importance = -- M2 <- M1
                    (1 - e.weight) * m2.importance + e.weight * m1.importance
            FROM edges e
                JOIN memories m1 ON e.src_id = m1.rowid
            WHERE m2.rowid = e.dst_id AND m1.cid = ?
        """, (memory.buffer,))

    @overload
    def find_sona(self, name: UUID) -> Optional[SonaRow]: ...
    @overload
    def find_sona(self, name: str) -> SonaRow: ...

    def find_sona(self, name: UUID|str):
        '''Find or create the sona closest to the given name.'''

        if isinstance(name, UUID):
            return self.select_sona_uuid(name)

        cur = self.cursor()        
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
        ebs = e.astype(numpy.float32)
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

    def insert_text_embedding(self, memory_id: int, index: str):
        '''Insert a text embedding for a memory.'''
        e, = nomic_text.embed(index)
        cur = self.cursor()
        # Deduplicate because sqlite-vec can't.
        cur.execute("""
            SELECT rowid FROM memory_vss
            WHERE embedding = ?
        """, (e.astype(numpy.float32),))
        if cur.fetchone():
            return
        # Insert the embedding
        cur.execute("""
            INSERT INTO memory_vss (memory_id, embedding)
            VALUES (?, ?)
        """, (memory_id, e.astype(numpy.float32)))

    def insert_text_fts(self, memory_id: int, index: str):
        '''Index a memory by inserting it into the full-text search index.'''
        cur = self.cursor()
        # Ignore duplicate memory_id
        cur.execute("""
            INSERT OR IGNORE INTO memory_fts (rowid, content)
            VALUES (?, ?)
        """, (memory_id, index))

    def insert_memory(self, memory: AnyMemory) -> int:
        cid = memory.cid
        cur = self.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO memories
                (cid, timestamp, kind, data, importance)
                    VALUES (?, ?, ?, JSONB(?), ?)
        """, (
            cid and cid.buffer,
            memory.timestamp,
            memory.data.kind,
            memory.data.model_dump_json(exclude={"kind"}),
            memory.importance
        ))
        
        # 0 (ignored) or None (error)
        if rowid := cur.lastrowid:
            # Continue insertion
            self.link_memory_edges(rowid, memory.edges or [])

            index = memory.document()
            self.insert_text_embedding(rowid, index)
            self.insert_text_fts(rowid, index)
            
            for sona in memory.sonas or []:
                # Link memory to sonas
                # TODO: Could be more efficient by batching
                if sona and (sona_row := self.find_sona(sona)):
                    self.link_sona(sona_row.rowid, rowid)
            
            # Propagate the memory's importance
            if memory.importance is not None:
                cur.execute("""
                    UPDATE memories AS m2
                        SET importance = -- M2 <- M1
                            (1 - e.weight) * ? + e.weight * m1.importance
                    FROM edges e
                        JOIN memories m1 ON e.src_id = m1.rowid
                    WHERE m2.rowid = e.dst_id AND m1.rowid = ?
                """, (memory.importance, rowid))
        elif cid:
            # Memory already exists
            rowid, = cur.execute("""
                SELECT rowid FROM memories WHERE cid = ?
            """, (cid.buffer,)).fetchone()
        else:
            raise RuntimeError(
                "Failed to insert memory, it may already exist."
            )
        
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

    def link_memory_edges(self, rowid: int, edges: list[Edge[CIDv1]]):
        '''
        Link the edges of a memory to the database. This is used when inserting
        a memory with edges that are already in the database.
        '''
        assert all(0 <= e.weight <= 1 for e in edges)
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
        cur = self.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO sona_memories
            (sona_id, memory_id) VALUES (?, ?)
        """, (sona_id, memory_id))