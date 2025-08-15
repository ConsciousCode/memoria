from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Optional, Self, Sequence, cast, overload
import sqlite3
from uuid import UUID
import os
import json

from pydantic import BaseModel
from numpy import ndarray
import numpy
import sqlite_vec
from fastembed import TextEmbedding
from uuid_extension import uuid7

from cid import CID, CIDv1

from .memory import ACThread, AnyMemory, Edge, IncompleteACThread, DraftMemory, Memory, MemoryDataAdapter, PartialMemory
from .config import RecallConfig
from .util import finite

__all__ = (
    'FileRow',
    'MemoryRow', 'IncompleteMemoryRow', 'AnyMemoryRow',
    'EdgeRow',
    'SonaRow',
    'ACThreadRow', 'IncompleteACThreadRow', 'AnyACThreadRow',
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
    filename: Optional[str]
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
    timestamp: Optional[int]
    data: JSONB
    metadata: Optional[JSONB]

    @classmethod
    def factory(cls, *,
            cid: Optional[bytes],
            rowid: PrimaryKey,
            timestamp: Optional[int],
            data: JSONB,
            metadata: Optional[JSONB]
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

class SonaRow(BaseModel):
    type PrimaryKey = int

    rowid: PrimaryKey
    uuid: bytes
    active_id: Optional['ACThreadRow.PrimaryKey']
    pending_id: Optional['ACThreadRow.PrimaryKey']

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
    def factory(cls, **row) -> 'AnyACThreadRow':
        '''Create an ACThreadRow from a raw database row.'''
        if row['cid']:
            return ACThreadRow(**row)
        else:
            return IncompleteACThreadRow(**row)

class ACThreadRow(BaseACThreadRow):
    '''A completed act thread, has a CID.'''
    cid: bytes

class IncompleteACThreadRow(BaseACThreadRow):
    '''An incomplete act thread, does not have a CID.'''
    cid: None = None

type AnyACThreadRow = ACThreadRow | IncompleteACThreadRow

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

    conn.executescript(SCHEMA)
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
        self.cursor = cursor
    
    def __iter__(self) -> Iterator[T]:
        return iter(self.cursor)

    def execute(self, query: str, params: dict[str, object]|Sequence[object]=()) -> Self:
        self.cursor.execute(query, params)
        return self
    
    def executemany(self, query: str, params: Iterable[Sequence[object]]) -> Self:
        self.cursor.executemany(query, params)
        return self
    
    def fetchone(self) -> Optional[T]:
        return self.cursor.fetchone()
    
    def fetchall(self) -> list[T]:
        return self.cursor.fetchall()
    
    @property
    def lastrowid(self):
        return self.cursor.lastrowid

class Database:
    '''Type-safe(r) database interface.'''

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    @overload
    def cursor(self) -> Cursor[tuple[Any, *tuple[Any, ...]]]: ...
    @overload
    def cursor[T](self, factory: Callable[..., T]) -> Cursor[T]: ...
    @overload
    def cursor[T](self, *, return_type: type[T]) -> Cursor[T]: ...

    def cursor[T](self, factory: Optional[Callable[..., T]]=None, *, return_type: Optional[type[T]] = None):
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

        return bool(cur.fetchone())
    
    @overload
    def select_memory(self, *, cid: CIDv1) -> Optional[MemoryRow]: ...
    @overload
    def select_memory(self, *, rowid: int) -> Optional[AnyMemoryRow]: ...
    
    def select_memory(self, *,
            cid: Optional[CIDv1]=None,
            rowid: Optional[int]=None
        ) -> Optional[AnyMemoryRow]:
        '''Lookup a memory object by CID or rowid.'''
        cur = self.cursor(BaseMemoryRow.factory)
        return cur.execute("""
            SELECT
                rowid, cid, timestamp, kind, JSON(data)
            FROM memories
            WHERE cid = ?
        """, (cid and cid.buffer, rowid)).fetchone()
    
    def select_memory_ipld(self, *, cid: CIDv1) -> Optional[Memory]:
        '''
        Lookup a memory by CID, returning the complete Memory object.
        This is used to retrieve the memory data and edges.
        '''
        if mr := self.select_memory(cid=cid):
            return mr.to_memory(
                self.backward_edges(rowid=mr.rowid)
            )
    
    @overload
    def select_act(self, *, cid: CIDv1) -> Optional[ACThreadRow]: ...
    @overload
    def select_act(self, *, rowid: int) -> Optional[AnyACThreadRow]: ...

    def select_act(self, *, 
            cid: Optional[CIDv1]=None,
            rowid: Optional[int]=None
        ) -> Optional[AnyACThreadRow]:
        '''Lookup an ACT by CID or rowid.'''
        cur = self.cursor(BaseACThreadRow.factory)
        return cur.execute("""
            SELECT rowid, cid, sona_id, memory_id, prev_id
            FROM acthreads
            WHERE cid = ? OR rowid = ?
        """, (cid and cid.buffer, rowid)).fetchone()
    
    def select_act_ipld(self, *, cid: CID) -> Optional[ACThread]:
        '''
        Lookup an ACT by CID, returning the complete ACThread object.
        '''
        cur = self.cursor(ACThread)
        return cur.execute("""
            SELECT s.uuid AS sona, m.cid AS memory, prev.cid AS prev,
                JOIN sonas s ON acthreads.sona_id = sonas.rowid
                JOIN memories m ON acthreads.memory_id = m.rowid
                JOIN acthreads prev ON acthreads.prev_id = prev.rowid
            FROM acthreads act WHERE cid = ?
        """, (cid.buffer,)).fetchone()

    @overload
    def select_file(self, *, cid: CID) -> Optional[FileRow]: ...
    @overload
    def select_file(self, *, rowid: int) -> Optional[FileRow]: ...

    def select_file(self, *,
            cid: Optional[CID]=None,
            rowid: Optional[int]=None
        ) -> Optional[FileRow]:
        '''
        Lookup an IPLD file by CID, returning its FileRow. The Database
        doesn't handle block storage so this is only useful for metadata.
        '''
        cur = self.cursor(FileRow)
        return cur.execute("""
            SELECT cid, filename, mimetype, filesize, overhead
            FROM ipfs_files WHERE cid = ? OR rowid = ?
        """, (cid and cid.buffer, rowid)).fetchone()

    def select_sona_aliases(self, rowid: int) -> list[str]:
        '''Select all aliases for a sona by its rowid.'''
        cur = self.cursor(return_type=tuple[str])
        rows = cur.execute("""
            SELECT name FROM sona_aliases WHERE sona_id = ?
        """, (rowid,))
        return [name for (name,) in rows]

    def get_act_active(self, sona_id: int) -> Optional[ACThreadRow]:
        '''Get the sona's active thread node currently receiving updates.'''
        cur = self.cursor(ACThreadRow)
        return cur.execute("""
            SELECT act.rowid, cid, sona_id, memory_id, prev_id
            FROM sonas s
                JOIN acthreads act ON s.active_id = act.rowid
            WHERE s.rowid = ?
        """, (sona_id,)).fetchone()

    def get_act_pending(self, sona_id: int) -> Optional[ACThreadRow]:
        '''Get the sona's pending thread node which is aggregating requests.'''
        cur = self.cursor(ACThreadRow)
        return cur.execute("""
            SELECT act.rowid, cid, sona_id, memory_id, prev_id
            FROM sonas s
                JOIN acthreads act ON s.pending_id = act.rowid
            WHERE s.rowid = ?
        """, (sona_id,)).fetchone()
    
    @overload
    def select_sona(self, *, rowid: int) -> Optional[SonaRow]: ...
    @overload
    def select_sona(self, *, uuid: UUID) -> Optional[SonaRow]: ...

    def select_sona(self, *,
            rowid: Optional[int]=None,
            uuid: Optional[UUID]=None
        ) -> Optional[SonaRow]:
        '''Select a sona by its rowid.'''
        cur = self.cursor(SonaRow)
        return cur.execute("""
            SELECT rowid, uuid, active_id, pending_id
            FROM sonas
            WHERE rowid = ? OR uuid = ?
        """, (rowid, uuid and uuid.bytes)).fetchone()
    
    def get_incomplete_act(self, rowid: int) -> Optional[IncompleteACThread]:
        '''Get the act thread for a sona and memory.'''
        cur = self.cursor(
            return_type=tuple[
                int, Optional[bytes],
                JSONB,
                Optional[bytes], Optional[bytes]
            ]
        )
        row = cur.execute("""
            SELECT
                a1.rowid, a1.cid, -- IncompleteACThread
                m.data, -- IncompleteMemory
                s.uuid, a2.cid -- Sona, Previous CID
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
            m_data, # IncompleteMemory
            s_cid, a2_cid # Sona, Previous CID
        ) = row
        # Just to double-check that this is actually an incomplete ACT
        if a1_cid is not None:
            raise ValueError(f"ACT with rowid {a1_rowid} already has a CID: {a1_cid}")
        
        return IncompleteACThread(
            sona=UUID(bytes=s_cid),
            memory=DraftMemory(
                data=MemoryDataAdapter.validate_json(m_data),
                edges=[
                    Edge(target=CIDv1(e.target.cid), weight=e.weight)
                        for e in self.dependencies(rowid=rowid)
                ]
            ),
            prev=None if a2_cid is None else CIDv1(a2_cid)
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
        row = cur.execute("""
            SELECT embedding FROM memory_vss WHERE memory_id = ?
        """, (memory_id,)).fetchone()
        if row is None:
            return None
        return numpy.frombuffer(row[0], dtype=numpy.float32)
    
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

    @overload
    def backward_edges(self, *, rowid: int) -> Iterable[Edge[CIDv1]]: ...
    @overload
    def backward_edges(self, *, cid: CIDv1) -> Iterable[Edge[CIDv1]]: ...

    def backward_edges(self, *,
            rowid: Optional[int]=None,
            cid: Optional[CIDv1]=None
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
            rowid: Optional[int]=None,
            cid: Optional[CIDv1]=None
        ) -> Iterable[Edge[MemoryRow]]:
        '''
        Get all edges leading to the given memory, returning the source id
        and weight of the edge.
        '''
        cur = self.cursor(return_type=tuple[
            int, bytes, int, JSONB, Optional[JSONB], float
        ])
        cur.execute("""
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
            int, Optional[bytes], int, JSONB,
            Optional[JSONB],
            float
        ])
        cur.execute("""
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
    
    def list_sonas(self, page: int, perpage: int) -> Iterable[SonaRow]:
        '''List sonas in the database, paginated.'''
        cur = self.cursor(SonaRow)
        return cur.execute("""
            SELECT rowid, uuid, active_id, pending_id
            FROM sonas
            LIMIT ? OFFSET ?
        """, (perpage, (page - 1) * perpage))

class DatabaseRW(DatabaseRO):
    '''Provides mutating operations for the database.'''

    @classmethod
    def _construct(cls, db: DatabaseRO):
        self = DatabaseRW.__new__(DatabaseRW)
        self.__dict__ = db.__dict__
        return self
    
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
            (cid, rowid)
            for (rowid,) in mids
                if (m := self.select_memory(rowid=rowid))
                    if (cid := m.cid)
        ))
        cur.executemany("""
            DELETE FROM invalid_memories WHERE memory_id = ?
        """, mids)

        return True
    
    def find_sona_embedding(self, sona: Optional[UUID|str]) -> Optional[ndarray]:
        '''Select the embedding for a sona by its UUID.'''
        if sona is None:
            return None
        
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
            sona: Optional[UUID|str],
            prompt: AnyMemory,
            index: Optional[list[str]]=None,
            timestamp: Optional[int]=None,
            config: Optional[RecallConfig]=None
        ) -> Iterable[tuple[MemoryRow, float]]:
        config = config or RecallConfig()
        
        if index:
            e = numpy.mean(list(nomic_text.embed(index)))
        else:
            e = None
        
        if se := self.find_sona_embedding(sona):
            s = se.astype(numpy.float32)
        else:
            s = None
        
        cur = self.cursor(return_type=tuple[
            int, bytes, Optional[int], JSONB, Optional[JSONB],
            float
        ])
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
                    -- If the sona vss fails, treat it like a match
                    + IFNULL(:w_sona / (1 + svss.distance), 1)
                ) / (:w_recency + :w_fts + :w_vss + :w_sona) AS score
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
            "vss_e": e and e.astype(numpy.float32),
            "sona_e": s and s.astype(numpy.float32),
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
        cur.execute("""
            UPDATE memories SET cid = ? WHERE rowid = ?
        """, (m.cid.buffer, rowid))
        return m
    
    def finalize_act(self, rowid: int) -> ACThread:
        '''Finalize an ACT by setting its CID and returning it.'''

        cur = self.cursor(ACThread)
        cur.execute("""
            SELECT s.cid AS sona, m.cid AS memory, p.cid AS prev
            FROM acthreads act
                JOIN sonas s ON act.sona_id = s.rowid
                JOIN memories m ON act.memory_id = m.rowid
                LEFT JOIN acthreads p ON act.prev_id = p.rowid
            WHERE act.rowid = ?
        """, (rowid,))
        if (act := cur.fetchone()) is None:
            raise ValueError(f"ACT with rowid {rowid} does not exist.")

        cur.execute("""
            UPDATE acthreads SET cid = ?
            WHERE rowid = ?
        """, (act.cid.buffer, rowid))

        return act

    @overload
    def find_sona(self, name: UUID) -> Optional[SonaRow]: ...
    @overload
    def find_sona(self, name: str) -> SonaRow: ...

    def find_sona(self, name: UUID|str):
        '''Find or create the sona closest to the given name.'''

        if isinstance(name, UUID):
            return self.select_sona(uuid=name)

        cur = self.cursor(SonaRow)        
        cur.execute("""
            SELECT rowid, uuid, active_id, pending_id
            FROM sona_aliases
                JOIN sonas ON sonas.rowid = sona_aliases.sona_id
            WHERE name = ?
        """, (name,))
        
        if row := cur.fetchone():
            return row

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
            """, (row.rowid, name))
            return row
        
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
        
        return SonaRow(
            rowid=rowid,
            uuid=u.bytes,
            active_id=None,
            pending_id=None
        )

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

    def insert_memory(self,
            memory: Memory,
            index: Optional[list[str]]=None,
            timestamp: Optional[int] = None
        ) -> int:
        cid = memory.cid
        cid = cid and cid.buffer
        cur = self.cursor()
        cur.execute("""
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
        '''
        cur = self.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO ipfs_files (
                cid, filename, mimetype, filesize, overhead
            ) VALUES (?, ?, ?, ?, ?)
        """, (cid, filename, mimetype, filesize, overhead))
        
        if rowid := cur.lastrowid:
            return rowid
        raise RuntimeError("Failed to insert file, it may already exist.")

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