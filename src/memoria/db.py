from contextlib import contextmanager
from typing import Callable, Self, cast, overload
from collections.abc import Iterable, Iterator, Sequence
import sqlite3
import os
from uuid import UUID

from pydantic import BaseModel

from cid import CIDv1

from .memory import Memory, MemoryDataAdapter, PartialMemory
from .util import nonempty_tuple

__all__ = (
    'MemoryRow', 'EdgeRow',
    'CancelTransaction',
    'database', "DatabaseRO", "DatabaseRW"
)

with open(os.path.join(os.path.dirname(__file__), "schema.sql"), "r") as f:
    SCHEMA = f.read()

type JSONB = str
'''Alias for JSONB type in SQLite, which is selected as a string.'''

## Rows are represented as the raw output from a select query with no processing
## - this allows us to avoid processing columns we don't need
## PrimaryKey aliases also give us little semantic type hints for the linter

class MemoryRow(BaseModel):
    '''A completed memory, has a CID.'''
    
    type PrimaryKey = int
    
    rowid: PrimaryKey
    cid: bytes
    uuid: bytes | None
    data: JSONB

    def to_partial(self, edges: Iterable[CIDv1]=()) -> PartialMemory:
        u = self.uuid
        return PartialMemory(
            uuid=None if u is None else UUID(bytes=u),
            cid=CIDv1(self.cid),
            data=MemoryDataAdapter.validate_json(self.data),
            edges=set(edges)
        )

class EdgeRow(BaseModel):
    src_id: MemoryRow.PrimaryKey
    dst_id: MemoryRow.PrimaryKey

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

    def cursor[T](self, factory: Callable[..., T] | None=None, *, return_type: type[T] | None = None): # pyright: ignore [reportUnusedParameter]
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
    def select_memory(self, *, rowid: int) -> MemoryRow | None: ...
    
    def select_memory(self, *,
            cid: CIDv1 | None=None,
            rowid: int | None=None
        ) -> MemoryRow | None:
        '''Lookup a memory object by CID or rowid.'''
        return self.cursor(MemoryRow).execute("""
            SELECT rowid, cid, uuid, JSON(data) as data
            FROM memories
            WHERE cid = ? OR rowid = ?
        """, (cid and cid.buffer, rowid)).fetchone()
    
    def select_memories(self, cids: Iterable[CIDv1]) -> Iterable[MemoryRow]:
        '''Lookup a memory object by CID or rowid.'''
        return self.cursor(MemoryRow).executemany("""
            SELECT rowid, cid, uuid, JSON(data) as data
            FROM memories
            WHERE cid = ? OR rowid = ?
        """, ((cid.buffer,) for cid in cids))
    
    def select_memory_ipld(self, *, cid: CIDv1) -> PartialMemory | None:
        '''
        Lookup a memory by CID, returning the complete Memory object.
        This is used to retrieve the memory data and edges.
        '''
        if mr := self.select_memory(cid=cid):
            return mr.to_partial(self.backward_edges(rowid=mr.rowid))
    
    @overload
    def backward_edges(self, *, rowid: int) -> Iterable[CIDv1]: ...
    @overload
    def backward_edges(self, *, cid: CIDv1) -> Iterable[CIDv1]: ...

    def backward_edges(self, *,
            rowid: int | None=None,
            cid: CIDv1 | None=None
        ) -> Iterable[CIDv1]:
        '''Get all edges leading from the given memory.'''
        cur = self.cursor(return_type=tuple[bytes])
        _ = cur.execute("""
            SELECT dst.cid
            FROM edges e
                JOIN memories dst ON e.dst_id = dst.rowid
                LEFT JOIN memories src ON e.src_id = src.rowid
            WHERE src.cid = ? OR src.rowid = ?
        """, (cid and cid.buffer, rowid))
        for mcid, in cur:
            yield CIDv1(mcid)

    @overload
    def dependencies(self, *, rowid: int) -> Iterable[MemoryRow]: ...
    @overload
    def dependencies(self, *, cid: CIDv1) -> Iterable[MemoryRow]: ...

    def dependencies(self, *,
            rowid: int | None=None,
            cid: CIDv1 | None=None
        ) -> Iterable[MemoryRow]:
        '''
        Get all edges leading to the given memory, returning the source id
        and weight of the edge.
        '''
        return self.cursor(MemoryRow).execute("""
            SELECT
                dst.rowid as rowid, dst.cid as cid,
                dst.uuid as uuid, JSON(dst.data) as data
            FROM edges e
                JOIN memories dst ON e.dst_id = dst.rowid
                LEFT JOIN memories src ON e.src_d = src.rowid
            WHERE src.cid = ? OR src.rowid = ?
        """, (cid and cid.buffer, rowid))
    
    @overload
    def references(self, *, cid: CIDv1) -> Iterable[MemoryRow]: ...
    @overload
    def references(self, *, rowid: int) -> Iterable[MemoryRow]: ...
    
    def references(self, *,
        rowid: int | None=None,
        cid: CIDv1 | None=None
    ) -> Iterable[MemoryRow]:
        '''
        Get all edges leading from the given memory, returning the destination id
        and weight of the edge.
        '''
        return self.cursor(MemoryRow).execute("""
            SELECT
                m.rowid as rowid, m.cid as cid,
                m.uuid as uuid, JSON(m.data) as data
            FROM edges e
                JOIN memories src ON e.src_id = src.rowid
                JOIN memories dst ON e.dst_id = dst.rowid
            WHERE dst.cid = ? OR dst.rowid = ?
        """, (cid and cid.buffer, rowid))
    
    def all_memories(self) -> Iterable[MemoryRow]:
        return self.cursor(MemoryRow).execute("""
            SELECT rowid, cid, uuid, JSON(data) as data
            FROM memories
            ORDER BY uuid DESC
        """)
    
    def list_memories(self, page: int, perpage: int) -> Iterable[MemoryRow]:
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
    
    def insert_memory(self, memory: Memory) -> int:
        cid = memory.cid.buffer
        uid = memory.uuid
        cur = self.cursor(return_type=tuple[int])
        _ = cur.execute("""
            INSERT OR IGNORE INTO memories (cid, uuid, data)
            VALUES (?, ?, JSONB(?))
        """, (
            cid,
            None if uid is None else uid,
            memory.data.model_dump_json()
        ))
        
        if rowid := cur.lastrowid:
            # Continue insertion
            _ = cur.executemany("""
                INSERT OR IGNORE INTO edges (src_id, dst_id)
                SELECT ?, rowid
                FROM memories m WHERE cid = ?
            """, ((rowid, e.buffer) for e in memory.edges))
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
            rowid, = row
        
        return rowid
