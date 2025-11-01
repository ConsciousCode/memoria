from contextlib import contextmanager
from typing import Callable, Self, cast, overload
from collections.abc import Buffer, Iterable, Iterator, Sequence
import sqlite3
import os
import json

from pydantic import BaseModel
from numpy import ndarray
import numpy
import sqlite_vec
from fastembed import TextEmbedding

from cid import CID, CIDv1

from .memory import Edge, DraftMemory, Memory, MemoryDataAdapter, PartialMemory
from .config import RecallConfig
from .util import finite, nonempty_tuple

__all__ = (
    'FileRow',
    'MemoryRow', 'IncompleteMemoryRow', 'AnyMemoryRow',
    'EdgeRow',
    'CancelTransaction',
    'database', "DatabaseRO", "DatabaseRW"
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

class FileRow(BaseModel):
    type PrimaryKey = bytes

    cid: PrimaryKey
    filename: str | None
    mimetype: str
    filesize: int
    overhead: int

class BaseMemoryRow(BaseModel):
    '''
    Split MemoryRow into complete and incomplete varieties for type-level
    guarantees about whether a memory has a CID.
    '''
    type PrimaryKey = int

    rowid: PrimaryKey
    timestamp: int | None
    data: JSONB
    metadata: JSONB | None

    @classmethod
    def factory(cls, *,
            cid: bytes | None,
            rowid: PrimaryKey,
            timestamp: int | None,
            data: JSONB,
            metadata: JSONB | None
        ) -> 'AnyMemoryRow':
        '''Create a MemoryRow from a raw database row.'''
        if cid:
            return MemoryRow(
                cid=cid,
                rowid=rowid,
                timestamp=timestamp,
                data=data,
                metadata=metadata
            )
        else:
            return IncompleteMemoryRow(
                cid=None,
                rowid=rowid,
                timestamp=timestamp,
                data=data,
                metadata=metadata
            )
    
    def to_draft(self, edges: Iterable[Edge[CIDv1]]=()) -> DraftMemory:
        '''Convert this MemoryRow to a DraftMemory object.'''
        md = self.metadata
        return DraftMemory(
            data=MemoryDataAdapter.validate_python(self.data),
            metadata=None if md is None else json.loads(md),
            edges=list(edges)
        )

class MemoryRow(BaseMemoryRow):
    '''A completed memory, has a CID.'''
    cid: bytes

    def to_partial(self, edges: Iterable[Edge[CIDv1]]=()) -> PartialMemory:
        md = self.metadata
        return PartialMemory(
            cid=CIDv1(self.cid),
            data=MemoryDataAdapter.validate_python(self.data),
            metadata=None if md is None else json.loads(md),
            edges=list(edges)
        )

    def to_memory(self, edges: Iterable[Edge[CIDv1]]=()) -> Memory:
        '''Convert this MemoryRow to a Memory object.'''
        md = self.metadata
        return Memory(
            data=MemoryDataAdapter.validate_python(self.data),
            metadata=None if md is None else json.loads(md),
            edges=list(edges),
        )

class IncompleteMemoryRow(BaseMemoryRow):
    '''An incomplete memory, does not have a CID.'''
    cid: None = None

type AnyMemoryRow = MemoryRow | IncompleteMemoryRow

class EdgeRow(BaseModel):
    src_id: MemoryRow.PrimaryKey
    dst_id: MemoryRow.PrimaryKey
    weight: float

class CancelTransaction(Exception):
    '''
    Exception to raise to cancel a transaction. This will cause the transaction
    to be rolled back and the database to be closed.
    '''

@contextmanager
def database(db_path: str=":memory:"):
    '''
    All SQL queries are contained within this database class. This prevents the
    proliferation of SQL queries throughout the codebase, allowing for easier
    maintenance and updates.

    However, these queries do not respect data integrity on their own - they
    must be used correctly to ensure that the database remains consistent. For
    instance, memories are inserted separately from their edges, but memories
    themselves have a `cid` which depends on those edges.
    '''
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    _ = conn.executescript(SCHEMA)
    conn.commit()
    db = DatabaseRO(conn)
    try:
        yield db
    except CancelTransaction:
        db.rollback()
    except:
        db.rollback()
        raise
    else:
        db.commit()

class Cursor[T]:
    '''Type-safe(r) cursor interface.'''

    def __init__(self, cursor: sqlite3.Cursor):
        self.cursor: sqlite3.Cursor = cursor
    
    def __iter__(self) -> Iterator[T]:
        return iter(self.cursor)

    def execute(self, query: str, params: dict[str, object]|Sequence[object]=()) -> Self:
        _ = self.cursor.execute(query, params)
        return self
    
    def executemany(self, query: str, params: Iterable[Sequence[object]]) -> Self:
        _ = self.cursor.executemany(query, params)
        return self
    
    def fetchone(self):
        return cast(T | None, self.cursor.fetchone())
    
    def fetchall(self) -> list[T]:
        return self.cursor.fetchall()
    
    @property
    def lastrowid(self):
        return self.cursor.lastrowid

type scalar_t = int | float | str | bytes | bool | None

class Database:
    '''Type-safe(r) database interface.'''

    def __init__(self, conn: sqlite3.Connection):
        self.conn: sqlite3.Connection = conn

    @overload
    def cursor(self) -> Cursor[nonempty_tuple[scalar_t]]: ...
    @overload
    def cursor[T](self, factory: Callable[..., T]) -> Cursor[T]: ...
    @overload
    def cursor[T](self, *, return_type: type[T]) -> Cursor[T]: ...

    def cursor[T](self, factory: Callable[..., T] | None=None, *, return_type: type[T] | None = None):
        cur = self.conn.cursor()
        if factory:
            cur.row_factory = lambda cur, row: factory(**{
                desc[0]: row[i] for i, desc in enumerate(cur.description)
            })
        return Cursor[T](cur)
    
    def commit(self): self.conn.commit()
    def rollback(self): self.conn.rollback()
    def close(self): self.conn.close()

class DatabaseRO(Database):
    '''
    Provides read-only access to the database. This is used to prevent
    accidental mutations to the database when only read access is needed.
    '''
    
    @contextmanager
    def transaction(self):
        '''
        Context manager for a transaction. This will automatically commit or
        rollback the transaction depending on whether an exception was raised.
        '''
        try:
            dbrw = DatabaseRW.__new__(DatabaseRW)
            dbrw.__dict__ = self.__dict__
            yield dbrw
            self.commit()
        except CancelTransaction:
            self.rollback()
        except:
            self.rollback()
            raise
    
    def has_cid(self, cid: CIDv1) -> bool:
        '''Check if the database has a CID.'''
        cur = self.cursor()
        _ = cur.execute("""
            SELECT 1 FROM memories WHERE cid = ?
        """, (cid.buffer,))
        if cur.fetchone():
            return True
        
        _ = cur.execute("""
            SELECT 1 FROM acthreads WHERE cid = ?
        """, (cid.buffer,))

        return bool(cur.fetchone())
    
    @overload
    def select_memory(self, *, cid: CIDv1) -> MemoryRow | None: ...
    @overload
    def select_memory(self, *, rowid: int) -> AnyMemoryRow | None: ...
    
    def select_memory(self, *,
            cid: CIDv1 | None=None,
            rowid: int | None=None
        ) -> AnyMemoryRow | None:
        '''Lookup a memory object by CID or rowid.'''
        cur = self.cursor(BaseMemoryRow.factory)
        return cur.execute("""
            SELECT
                rowid, cid, timestamp, kind, JSON(data)
            FROM memories
            WHERE cid = ?
        """, (cid and cid.buffer, rowid)).fetchone()
    
    def select_memory_ipld(self, *, cid: CIDv1) -> Memory | None:
        '''
        Lookup a memory by CID, returning the complete Memory object.
        This is used to retrieve the memory data and edges.
        '''
        if mr := self.select_memory(cid=cid):
            return mr.to_memory(
                self.backward_edges(rowid=mr.rowid)
            )
    
    @overload
    def select_file(self, *, cid: CID) -> FileRow | None: ...
    @overload
    def select_file(self, *, rowid: int) -> FileRow | None: ...

    def select_file(self, *,
            cid: CID | None=None,
            rowid: int | None=None
        ) -> FileRow | None:
        '''
        Lookup an IPLD file by CID, returning its FileRow. The Database
        doesn't handle block storage so this is only useful for metadata.
        '''
        cur = self.cursor(FileRow)
        return cur.execute("""
            SELECT cid, filename, mimetype, filesize, overhead
            FROM ipfs_files WHERE cid = ? OR rowid = ?
        """, (cid and cid.buffer, rowid)).fetchone()

    def select_embedding(self, memory_id: int) -> ndarray | None:
        cur = self.cursor()
        row = cur.execute("""
            SELECT embedding FROM memory_vss WHERE memory_id = ?
        """, (memory_id,)).fetchone()
        if row is None:
            return None
        return numpy.frombuffer(cast(Buffer, row[0]), dtype=numpy.float32)
    
    @overload
    def backward_edges(self, *, rowid: int) -> Iterable[Edge[CIDv1]]: ...
    @overload
    def backward_edges(self, *, cid: CIDv1) -> Iterable[Edge[CIDv1]]: ...

    def backward_edges(self, *,
            rowid: int | None=None,
            cid: CIDv1 | None=None
        ) -> Iterable[Edge[CIDv1]]:
        '''Get all edges leading from the given memory.'''
        cur = self.cursor(Edge[CIDv1])
        return cur.execute("""
            SELECT dst.cid AS target, e.weight AS weight
            FROM edges e
                JOIN memories dst ON e.dst_id = dst.rowid
                LEFT JOIN memories src ON e.src_id = src.rowid
            WHERE src.cid = ? OR src.rowid = ?
        """, (cid and cid.buffer, rowid))

    @overload
    def dependencies(self, *, rowid: int) -> Iterable[Edge[MemoryRow]]: ...
    @overload
    def dependencies(self, *, cid: CIDv1) -> Iterable[Edge[MemoryRow]]: ...

    def dependencies(self, *,
            rowid: int | None=None,
            cid: CIDv1 | None=None
        ) -> Iterable[Edge[MemoryRow]]:
        '''
        Get all edges leading to the given memory, returning the source id
        and weight of the edge.
        '''
        cur = self.cursor(return_type=tuple[
            int, bytes, int, JSONB, JSONB | None, float
        ])
        _ = cur.execute("""
            SELECT
                dst.rowid, dst.cid, dst.timestamp,
                JSON(dst.data), JSON(dst.metadata),
                e.weight
            FROM edges e
                JOIN memories dst ON e.dst_id = dst.rowid
                LEFT JOIN memories src ON e.src_d = src.rowid
            WHERE src.cid = ? OR src.rowid = ?
        """, (cid and cid.buffer, rowid))
        
        for rowid, mcid, ts, data, md, weight in cur:
            yield Edge(
                target=MemoryRow(
                    rowid=rowid,
                    cid=mcid,
                    timestamp=ts,
                    data=data,
                    metadata=md
                ),
                weight=weight
            )
    
    def references(self, *, rowid: int) -> Iterable[Edge[AnyMemoryRow]]:
        '''
        Get all edges leading from the given memory, returning the destination id
        and weight of the edge.
        '''
        cur = self.cursor(return_type=tuple[
            int, bytes | None, int, JSONB,
            JSONB | None,
            float
        ])
        _ = cur.execute("""
            SELECT
                m.rowid, m.cid, m.timestamp, JSON(m.data),
                JSON(m.metadata),
                e.weight
            FROM edges e JOIN memories m ON m.rowid = e.src_id
            WHERE e.dst_id = ?
        """, (rowid,))
        
        for rowid, mcid, ts, data, md, weight in cur:
            yield Edge(
                weight=weight,
                target=BaseMemoryRow.factory(
                    rowid=rowid,
                    cid=mcid,
                    timestamp=ts,
                    data=data,
                    metadata=md
                )
            )
    
    def list_memories(self, page: int, perpage: int) -> Iterable[MemoryRow]:
        '''List messages in the database, paginated.'''
        cur = self.cursor(MemoryRow)
        return cur.execute("""
            SELECT
                rowid, cid, timestamp, kind, JSON(data) AS data
            FROM memories
            ORDER BY rowid DESC
            LIMIT ? OFFSET ?
        """, (perpage, (page - 1) * perpage))

class DatabaseRW(DatabaseRO):
    '''Provides mutating operations for the database.'''
    
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
        cur = self.cursor(return_type=tuple[int])
        _ = cur.execute("""
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
        
        cur = cur.executemany("""
            UPDATE memories SET cid = ?
            WHERE rowid = ?
        """, (
            (cid, rowid)
            for (rowid,) in mids
                if (m := self.select_memory(rowid=rowid))
                    if (cid := m.cid)
        ))
        cur = cur.executemany("""
            DELETE FROM invalid_memories WHERE memory_id = ?
        """, mids)

        return True
    
    def recall(self,
            index: list[str] | None=None,
            timestamp: int | None=None,
            config: RecallConfig | None=None
        ) -> Iterable[tuple[MemoryRow, float]]:
        config = config or RecallConfig()
        
        if index:
            e = numpy.mean(list(nomic_text.embed(index)))
        else:
            e = None
        
        cur = self.cursor(return_type=tuple[
            int, bytes, int | None, JSONB, JSONB | None,
            float
        ])
        _ = cur.execute("""
            WITH
                fts AS (
                    SELECT rowid, bm25(memory_fts) AS score
                    FROM memory_fts
                    WHERE :index AND memory_fts MATCH :index
                    LIMIT :k
                ),
                vss AS (
                    SELECT memory_id, distance
                    FROM memory_vss
                    WHERE embedding MATCH :vss_e AND k = :k
                )
            SELECT
                m.rowid, m.cid, m.timestamp, JSON(m.data),
                JSON(m.metadata),
                (
                    + IFNULL(:w_recency * POWER(
                        :timestamp - m.timestamp, -:decay
                    ), 0)
                    + IFNULL(:w_fts *
                        (fts.score - MIN(fts.score) OVER())
                        / (MAX(fts.score) OVER() - MIN(fts.score) OVER()), 0)
                    + IFNULL(:w_vss / (1 + mvss.distance), 0)
                ) / (:w_recency + :w_fts + :w_vss) AS score
            FROM memories m
                LEFT JOIN sona_memories sm ON sm.memory_id = m.rowid
                LEFT JOIN fts ON fts.rowid = m.rowid
                LEFT JOIN vss ON vss.memory_id = m.rowid
            WHERE
                m.cid IS NOT NULL AND ( -- Don't recall incomplete memories
                    :timestamp IS NULL OR
                    m.timestamp <= :timestamp -- Don't recall future memories
                )
            ORDER BY score DESC
            LIMIT :k
        """, {
            "index": index,
            "vss_e": e and e.astype(numpy.float32),
            "timestamp": timestamp,
            "w_recency": finite(config.recency),
            "w_fts": finite(config.fts),
            "w_vss": finite(config.vss),
            "w_sona": finite(config.sona),
            "k": int(config.k),
            "decay": finite(config.decay)
        })
        for rowid, cid, ts, data, md, score in cur:
            m = MemoryRow(
                rowid=rowid,
                cid=cid,
                timestamp=ts,
                data=data,
                metadata=md
            )
            yield m, score
    
    def finalize_memory(self, rowid: int) -> Memory:
        '''
        Finalize a memory by setting its CID and returning the memory object.
        This is used after all edges have been linked to the memory.
        '''
        if (mr := self.select_memory(rowid=rowid)) is None:
            raise ValueError(f"Memory with rowid {rowid} does not exist.")
        
        if mr.cid is not None:
            raise ValueError(f"Memory with rowid {rowid} already has a CID: {mr.cid}")
        
        # Build the CID from the memory data
        m = Memory(
            data=MemoryDataAdapter.validate_python(mr.data),
            edges=list(self.backward_edges(rowid=rowid))
        )
        cur = self.cursor()
        _ = cur.execute("""
            UPDATE memories SET cid = ? WHERE rowid = ?
        """, (m.cid.buffer, rowid))
        return m
    
    def insert_text_embedding(self, memory_id: int, index: str):
        '''Insert a text embedding for a memory.'''
        e, = nomic_text.embed(index)
        cur = self.cursor()
        # Deduplicate because sqlite-vec can't.
        _ = cur.execute("""
            SELECT rowid FROM memory_vss
            WHERE embedding = ?
        """, (e.astype(numpy.float32),))
        if cur.fetchone():
            return
        # Insert the embedding
        _ = cur.execute("""
            INSERT INTO memory_vss (memory_id, embedding)
            VALUES (?, ?)
        """, (memory_id, e.astype(numpy.float32)))

    def insert_text_fts(self, memory_id: int, index: str):
        '''Index a memory by inserting it into the full-text search index.'''
        cur = self.cursor()
        # Ignore duplicate memory_id
        _ = cur.execute("""
            INSERT OR IGNORE INTO memory_fts (rowid, content)
            VALUES (?, ?)
        """, (memory_id, index))

    def insert_memory(self,
            memory: Memory,
            index: list[str] | None=None,
            timestamp: int | None = None
        ) -> int:
        cid = memory.cid
        cid = cid and cid.buffer
        cur = self.cursor(return_type=tuple[int])
        _ = cur.execute("""
            INSERT OR IGNORE INTO memories
                (cid, timestamp, kind, data)
                    VALUES (?, ?, ?, JSONB(?), ?)
        """, (
            cid,
            timestamp,
            memory.data.kind,
            memory.data.model_dump_json(exclude={"kind"})
        ))
        
        if rowid := cur.lastrowid:
            # Continue insertion
            self.link_memory_edges(rowid, memory.edges or [])

            for doc in index or []:
                self.insert_text_embedding(rowid, doc)
                self.insert_text_fts(rowid, doc)
        # 0 (ignored) or None (error)
        else:
            # Memory already exists
            row = cur.execute("""
                SELECT rowid FROM memories WHERE cid = ?
            """, (cid,)).fetchone()
            if row is None:
                raise RuntimeError(
                    "Failed to either insert memory or lookup by CID."
                )
            rowid = row[0]
        
        return rowid
    
    def register_file(self, 
            cid: CID,
            filename: str, 
            mimetype: str,
            filesize: int,
            overhead: int
        ) -> int:
        '''
        Insert a file into the database. This is used for files that are not
        linked to any memory, such as images or other media.
        
        Returns the rowid of the inserted file.
        '''
        cur = self.cursor()
        _ = cur.execute("""
            INSERT OR IGNORE INTO ipfs_files (
                cid, filename, mimetype, filesize, overhead
            ) VALUES (?, ?, ?, ?, ?)
        """, (cid, filename, mimetype, filesize, overhead))
        
        if rowid := cur.lastrowid:
            return rowid
        raise RuntimeError("Failed to insert file, it may already exist.")

    def update_memory_data(self, rowid: int, data: str):
        '''
        Insert the data for a memory. This is used for file memories and other
        large data blobs.
        '''
        cur = self.cursor()
        _ = cur.execute("""
            UPDATE memories SET data = JSONB(?)
            WHERE rowid = ?
        """, (data, rowid))

    def link_memory_edges(self, rowid: int, edges: list[Edge[CIDv1]]):
        '''
        Link the edges of a memory to the database. This is used when inserting
        a memory with edges that are already in the database.
        '''
        assert all(0 <= e.weight <= 1 for e in edges)
        cur = self.cursor()
        cur = cur.executemany("""
            INSERT OR IGNORE INTO edges (src_id, dst_id, weight)
            SELECT ?, rowid, ?
            FROM memories m WHERE cid = ?
        """, (
            (rowid, e.weight, e.target.buffer)
                for e in edges
        ))
