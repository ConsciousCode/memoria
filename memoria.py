from dataclasses import dataclass
import sqlite3
from typing import Any, Iterable, Literal, Optional, Self, overload
import json
from datetime import datetime
import re

from pydantic import BaseModel, Field
import sqlite_vec
from fastembed import TextEmbedding, ImageEmbedding
from fastembed.common.model_description import PoolingType, ModelSource
from fastembed.common.types import ImageInput

from pydantic_ai import Agent, RunContext

SCHEMA = '''
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
    rowid INTEGER PRIMARY KEY,
    multihash TEXT NOT NULL,
    filename TEXT, /* filename at time of upload */
    mimetype TEXT NOT NULL,
    metadata JSONB,
    size INTEGER NOT NULL,
    content BLOB, /* actual file content - NULL = external storage */

    UNIQUE(multihash)
);

CREATE TABLE IF NOT EXISTS memories (
    rowid INTEGER PRIMARY KEY,
    timestamp REAL,
    kind TEXT NOT NULL,
    data JSONB NOT NULL,
    importance REAL
);

CREATE TABLE IF NOT EXISTS edges (
    src_id INTEGER NOT NULL,
    dst_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    weight REAL NOT NULL,

    PRIMARY KEY (src_id, label),
    FOREIGN KEY (src_id) REFERENCES memories(rowid),
    FOREIGN KEY (dst_id) REFERENCES memories(rowid)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(content);

CREATE VIRTUAL TABLE IF NOT EXISTS vss_nomic_v1_5_index USING vec0 (
    memory_id INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);
'''

nomic_text = TextEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
)

ImageEmbedding.add_custom_model(
    model="nomic-ai/nomic-embed-vision-v1.5",
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf="nomic-ai/nomic-embed-vision-v1.5"),
    dim=768
)
nomic_image = ImageEmbedding(
    model_name="nomic-ai/nomic-embed-vision-v1.5"
)

type json_t = dict[str, json_t]|list[json_t]|str|int|float|bool|None

def _fattr(x: json_t) -> str:
    if isinstance(x, str) and re.match(r"""[\s'"=</>&;]""", x):
        return json.dumps(x)
    else:
        return str(x)

def _fattrs(k: str, v: json_t) -> Iterable[str]:
    match v:
        case None|False: pass
        case True: yield k
        case list():
            for x in v:
                yield from _fattrs(k, x)
        case _:
            yield f'{k}={_fattr(v)}'

_xents = str.maketrans({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&apos;",
    '"': "&quot;"
})

def _X(tag: str, **props: json_t):
    def X_Content(content: str):
        pv = []
        for k, v in props.items():
            pv.extend(_fattrs(k, v))
        
        p = f' {" ".join(pv)}' if pv else ""
        return f"<{tag}{p}>{content.translate(_xents)}</{tag}>"
    
    return X_Content

type MemoryKind = Literal["text", "image", "file", "entity"]

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
        return cls(
            rowid, datetime.fromtimestamp(ts),
            kind, json.loads(data), importance
        )

    def format(self, **props) -> str:
        p = {
            "id": hex(self.rowid),
            "datetime": self.timestamp and self.timestamp.isoformat(),
            "importance": self.importance
        }
        
        match self.kind:
            case _:
                return _X("memory", **p, **props)(json.dumps(self.data))

def todo_iter[T](todo: list[T]) -> Iterable[T]:
    '''
    Iterate over a stack, removing items as they are yielded. This can be
    appended to during iteration.
    '''
    while todo:
        yield todo.pop(0)

class Graph[K, E, V]:
    adj: dict[K, tuple[V, list[tuple[E, K]]]]

    def __init__(self, keys: dict[K, V]|None = None):
        super().__init__()
        self.adj = {k: (v, []) for k, v in (keys or {}).items()}
    
    def __contains__(self, key: K) -> bool:
        return key in self.adj

    def insert(self, key: K, value: V):
        if key not in self.adj:
            self.adj[key] = (value, [])
    
    def __iter__(self):
        return iter(self.adj)

    def add_edge(self, src: K, dst: K, edge: E):
        if src not in self.adj:
            raise KeyError(f"Source {src} not found")
        if dst not in self.adj:
            raise KeyError(f"Destination {dst} not found")
        self.adj[src][1].append((edge, dst))
    
    def __getitem__(self, key: K) -> tuple[V, list[tuple[E, K]]]:
        return self.adj[key]
    
    def edges(self, k: Optional[K] = None) -> Iterable[tuple[E, K]]:
        if k is None:
            for _, edges in self.adj.values():
                yield from edges
        else:
            _, edges = self.adj[k]
            yield from edges
    
    def toposort(self) -> Iterable[V]:
        '''
        BFS topological sort of a directed graph.
        '''
        for u in todo_iter(unseen := list(self.adj.keys())):
            for k in todo_iter(todo := [u]):
                m, edges = self.adj[k]
                yield m

                for _, src_id in edges:
                    if src_id in unseen:
                        unseen.remove(src_id)
                        todo.append(src_id)

@dataclass
class Database:
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
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
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
    
    def insert_image(self, memory_id: int, image: ImageInput) -> int:
        e, = nomic_image.embed(image)
        rowid = self.insert_memory("image", e, None)
        cur = self.cursor()
        cur.execute("""
            INSERT INTO vss_nomic_v1_5_index
                (memory_id, embedding) VALUES (?, ?)
        """, (memory_id, e.tobytes()))

        return rowid

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
            SELECT m.rowid, m.timestamp, m.kind, m.data,
                0.30 * m.importance +
                0.30 * power(0.995, ? - m.timestamp) +
                0.15 * bm25(memory_fts) +
                0.25 / (1 + vss_distance)) AS score
            FROM memories m
                JOIN memory_fts ON memory_fts.rowid = m.rowid
                JOIN vss_nomic_v1_5_index
                    ON vss_nomic_v1_5_index.memory_id = m.rowid
            WHERE memory_fts MATCH ?
                AND vss_search(
                    vss_nomic_v1_5_index.embedding, 
                    vss_search_params(?, 10)
                )
            ORDER BY score DESC
        """, (datetime.now().timestamp(), prompt, e.tobytes()))

        budget = 10 # placeholder
        forward = 0
        
        # Not fetchall so we can stop early
        for row in cur:
            rowid, score = row[0], row[-1]
            g.insert(rowid, MemoryRow.from_row(row[:5]))

            forward += score
            if forward >= budget:
                break
        
        backward = [(budget*e[1], k) for e, k in g.edges()]
        forward = backward.copy()

        # Note: These feel so similar, maybe there's a way to reduce boilerplate?

        # Search backwards for supporting memories using their edge weight
        #  to determine how relevant they are to the current memory
        for budget, dst_id in todo_iter(backward):
            b = 0
            cur.execute("""
                SELECT
                    e.src_id, m.timestamp, m.kind, m.data, m.importance,
                    e.label, e.weight
                FROM edges e JOIN memories m ON m.rowid = e.dst_id
                WHERE e.dst_id = ?
                ORDER BY e.weight DESC
            """, (dst_id,))
            for row in cur:
                src_id, node, edge = row[0], row[:5], row[5:]
                weight = edge[-1]
                g.insert(src_id, MemoryRow.from_row(node))
                g.add_edge(src_id, dst_id, edge)

                backward.append((budget*weight, src_id))

                b += weight
                if b >= budget:
                    break
        
        # Search forwards for response memories using their annotated
        #  importance - important conclusions are more relevant
        for budget, src_id in todo_iter(forward):
            f = 0
            cur.execute("""
                SELECT
                    e.dst_id, m.timestamp, m.kind, m.data, m.importance,
                    e.label, e.weight
                FROM edges e JOIN memories m ON m.rowid = e.src_id
                WHERE e.src_id = ?
                ORDER BY m.importance DESC
            """, (src_id,))
            for row in cur:
                dst_id, node, edge = row[0], row[:5], row[5:]
                importance = node[-1]
                g.insert(dst_id, MemoryRow.from_row(node))
                g.add_edge(src_id, dst_id, edge)

                forward.append((budget*importance, dst_id))

                f += importance
                if f >= budget:
                    break

        return g

class System1ResponseModel(BaseModel):
    response: str = Field(description="My response to the prompt.")
    importance: int = Field(description="The importance of the prompt on a scale of 1-10.", ge=1, le=10)
    weights: dict[str, int] = Field(description="For each memory, mapping their hex id to how important it was to my response on a scale of 1-10. I won't list them at all if they weren't useful.", ge=1, le=10)

class System1Deps(BaseModel):
    prev: Optional[int]
    db: Database

system1 = Agent(
    name="ezra.system1",
    output_type=System1ResponseModel,
    deps_type=System1Deps,
    system_prompt="""My name is Ezra, AI assistant and experiment in artificial phenomenology and personhood. I am embodied by my memories and simulated by one or more LLMs.""",
    instructions="""I need to use my memories to respond to the prompt. The prompt and my response will be added to my memory."""
)

@system1.system_prompt
async def system1_system_prompt(ctx: RunContext[System1Deps]) -> str:
    match ctx.prompt:
        case str(prompt): pass
        case [*it]: prompt = ''.join(map(str, it))
        case None: raise RuntimeError("No prompt provided")
    
    memories: list[str] = []
    g = ctx.deps.db.recall(ctx.deps.prev, prompt)
    for row in g.toposort():
        props: dict = {
            "role": "prev" if row.rowid == ctx.deps.prev else None
        }
        for label, src_id in g.edges(row.rowid):
            xid = hex(src_id)
            if (p := props.get(label)) is not None:
                if isinstance(p, list):
                    p.append(xid)
                else:
                    props[label] = [p, xid]
            else:
                props[label] = xid
        memories.append(row.format(**props))

    if not memories:
        return "I remember... Nothing! I have no memories."
    return f"I remember... <memories>\n{'\n'.join(memories)}\n</memories>"

class Memoria:
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
    
    async def process(self, prev: Optional[int], prompt: str):
        ts = datetime.now()
        result = await system1.run(
            prompt,
            deps=System1Deps(
                prev=prev,
                db=self.db
            ),
            output_type=System1ResponseModel
        )
        output = result.output

        prompt_id = self.db.insert_text(prompt,
            importance=output.importance / 10,
            timestamp=ts
        )
        # Do I need the importance of the response?
        response_id = self.db.insert_text(output.response)
        
        self.db.link("prompt", 1.0, response_id, prompt_id)
        
        # Connect referenced memories to response with weights
        self.db.link_many(
            ("ref", weight / 10, response_id, int(xid, 16))
                for xid, weight in output.weights.items()
        )
        
        # If there was a previous message, link to it
        if prev is not None:
            self.db.link("prev", 1.0, response_id, prev)
        
        self.db.commit()
        
        return {
            "id": response_id,
            "response": output.response,
            "importance": output.importance,
            "weights": output.weights
        }
