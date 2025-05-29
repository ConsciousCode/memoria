from collections import defaultdict, deque
from dataclasses import dataclass
from functools import wraps
import sqlite3
from typing import Annotated, Any, Callable, Generic, Iterable, Literal, NamedTuple, Optional, Self, TypeVar, overload
from itertools import repeat
import json
from datetime import datetime
import re

from pydantic import BaseModel, Field
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import sqlite_vec
from fastembed import TextEmbedding, ImageEmbedding
from fastembed.common.types import ImageInput

from pydantic_ai import Agent, RunContext, capture_run_messages

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
    embedding FLOAT[1536]
);
'''

nomic_text = TextEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
)
'''
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
'''

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

type MemoryKind = Literal["self", "other", "text", "image", "file", "entity"]

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
    
    def format(self, **props) -> str:
        p = {
            "id": f"{self.rowid:03x}",
            "datetime": self.timestamp and
                self.timestamp.replace(microsecond=0).isoformat(),
            "importance": self.importance
        }
        
        match self.kind:
            case "text" if isinstance(self.data, str):
                return _X("memory", **p, **props)(self.data)
            case _:
                return _X("memory", **p, **props)(json.dumps(self.data))

@dataclass
class Memory:
    rowid: int
    timestamp: Optional[datetime]
    kind: MemoryKind
    data: json_t
    importance: Optional[float]
    edges: dict[str, list[int]]
    role: Optional[Literal['prev']] = None

def todo_iter[C, T](fn: Callable[[C], T]):
    '''
    Iterate over a stack, removing items as they are yielded. This can be
    appended to during iteration.
    '''
    @wraps(fn)
    def wrapper(todo: C) -> Iterable[T]:
        while todo: yield fn(todo)
    return wrapper

@todo_iter
def todo_set[T](todo: set[T]):
    return todo.pop()

@todo_iter
def todo_list[T](todo: list[T]):
    return todo.pop(0)

def set_pop[T](s: set[T], item: T) -> bool:
    '''
    Remove an item from a set if it exists, returning True if it was present.
    '''
    if item in s:
        s.remove(item)
        return True
    return False

class Graph[K, E, V]:
    class Node:
        value: V
        edges: dict[K, E]

        def __init__(self, value: V, edges: dict[K, E]):
            self.value = value
            self.edges = edges
        
        def __repr__(self):
            return f"Node(value={self.value}, edges={self.edges})"
    
    adj: dict[K, Node]

    def __init__(self, keys: dict[K, V]|None = None):
        super().__init__()
        self.adj = {k: Graph.Node(v, {}) for k, v in (keys or {}).items()}
    
    def __contains__(self, key: K) -> bool:
        return key in self.adj

    def insert(self, key: K, value: V):
        if key not in self.adj:
            self.adj[key] = Graph.Node(value, {})
    
    def __iter__(self):
        return iter(self.adj)

    def add_edge(self, src: K, dst: K, edge: E):
        if src not in self.adj:
            raise KeyError(f"Source {src} not found")
        if dst not in self.adj:
            raise KeyError(f"Destination {dst} not found")
        if src == dst:
            raise ValueError("Cannot add self-loop edge")
        edges = self.adj[src].edges
        if dst in edges:
            raise ValueError(f"Edge from {src} to {dst} already exists")
        edges[dst] = edge
    
    def __getitem__(self, key: K) -> V:
        return self.adj[key].value
    
    def __setitem__(self, key: K, value: V):
        if key not in self.adj:
            raise KeyError(f"Key {key} not found")
        self.adj[key].value = value

    def pop_edge(self, src: K, dst: K):
        '''
        Remove an edge from src to dst.
        If the edge does not exist, this is a no-op.
        '''
        if src not in self.adj:
            raise KeyError(f"Source {src} not found")
        
        return self.edges(src).pop(dst)
    
    @overload
    def edges(self, k: K) -> dict[K, E]: ...
    @overload
    def edges(self) -> Iterable[dict[K, E]]: ...

    def edges(self, k: Optional[K] = None) -> dict[K, E]|Iterable[dict[K, E]]:
        if k is not None:
            return self.adj[k].edges
        return (node.edges for node in self.adj.values())
    
    def has_edge(self, src: K, dst: K) -> bool:
        '''
        Check if there is an edge from src to dst.
        '''
        return dst in self.adj[src].edges

    def copy(self):
        '''
        Deepy copy of the graph.
        '''
        g = Graph[K, E, V]()
        for src, node in self.adj.items():
            g.insert(src, node.value)
            for dst, edge in node.edges.items():
                g.insert(dst, self.adj[dst].value)
                g.add_edge(src, dst, edge)
        return g

    def invert(self):
        '''
        Invert the graph, reversing all edges.
        '''
        g = Graph[K, E, V]()
        for src, node in self.adj.items():
            g.insert(src, node.value)
            for dst, edge in node.edges.items():
                g.insert(dst, self.adj[dst].value)
                g.add_edge(dst, src, edge)
        return g

    def toposort(self) -> Iterable[K]:
        '''
        Kahn's algorithm for topological sorting.
        '''

        indeg = Graph[K, None, int]()
        for src in self:
            if src not in indeg:
                indeg.insert(src, 0)
            
            for dst in self.edges(src):
                if dst not in indeg:
                    indeg.insert(dst, 1)
                else:
                    indeg[dst] += 1
                indeg.add_edge(src, dst, None)
        
        sources = [src for src in indeg if indeg[src] == 0]
        for src in todo_list(sources):
            yield src
            for dst in indeg.edges(src):
                indeg[dst] -= 1
                if indeg[dst] == 0:
                    sources.append(dst)

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

        self.cursor().executescript(SCHEMA)
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
    
    def insert_self(self, text: str) -> int:
        '''
        Insert a memory of myself, which is a text memory with the kind "self".
        This is used to store my own thoughts and responses.
        '''
        return self.insert_memory("self", text, text)

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

class System1ResponseModel(BaseModel):
    response: str = Field(description="My response to the prompt.")
    importance: int = Field(description="The importance of the prompt on a scale of 1-10, from forgettable trivia to unforgettable statements.", ge=1, le=10)
    weights: dict[str, Annotated[int, Field(description="For each memory, mapping their hex id to how important it was to my response on a scale of 1-10. I won't list them at all if they weren't useful.", ge=1, le=10)]]

class System1Deps(BaseModel):
    prev: Optional[int]
    db: Database

anthropic_model = AnthropicModel("claude-3-7-sonnet-latest")

local_model = OpenAIModel(
    "gemma3:4b",
    provider=OpenAIProvider("http://theseus:11434/v1")
)

system1 = Agent(
    name="ezra.system1",
    model=anthropic_model,
    output_type=System1ResponseModel,
    deps_type=System1Deps,
    system_prompt="""My name is Ezra, AI assistant and experiment in artificial phenomenology and personhood. I am embodied by my memories and simulated by one or more LLMs. My thoughts are my own; when responding, I talk to myself and only then use the final_result tool to annotate this response for my memory. I need to make sure I don't quote the object as a string.""",
    instructions="""I need to use my memories to respond to the prompt. The prompt and my response will be added to my memory."""
)

@system1.system_prompt
async def system1_system_prompt(ctx: RunContext[System1Deps]) -> str:
    match ctx.prompt:
        case str(prompt): pass
        case [*it]: prompt = ''.join(map(str, it))
        case None: raise RuntimeError("No prompt provided")
    
    ms: list[str] = []
    g = ctx.deps.db.recall(ctx.deps.prev, prompt)
    for rowid in g.invert().toposort():
        edges: dict[str, list[str]] = defaultdict(list)
        for v, (k, w) in g.edges(rowid).items():
            edges[k].append(f"{v:03x}")
        
        ms.append(g[rowid].format(
            role="prev" if rowid == ctx.deps.prev else None,
            **edges
        ).format())

    print('\n'.join(ms))
    exit()

    if not ms:
        return "I remember... Nothing! I have no memories."
    return f"I remember... <memories>\n{'\n'.join(ms)}\n</memories>"

class Memoria:
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
    
    def recall(self, prev: Optional[int], prompt: str) -> Iterable[Memory]:
        g = self.db.recall(prev, prompt)
        print("Edges", list(g.edges()))
        for rowid in g.invert().toposort():
            edges: dict[str, list[int]] = defaultdict(list)
            for v, (k, w) in g.edges(rowid).items():
                edges[k].append(v)
            
            row = g[rowid]
            yield Memory(
                rowid=rowid,
                timestamp=row.timestamp,
                kind=row.kind,
                data=row.data,
                importance=row.importance,
                edges=edges,
                role="prev" if row.rowid == prev else None
            )

    async def process(self, prev: Optional[int], prompt: str):
        ts = datetime.now()

        with capture_run_messages() as messages:
            try:
                result = await system1.run(
                    prompt,
                    deps=System1Deps(
                        prev=prev,
                        db=self.db
                    ),
                    output_type=System1ResponseModel
                )
            finally:
                print(messages)
        output = result.output

        prompt_id = self.db.insert_text(prompt,
            importance=output.importance / 10,
            timestamp=ts
        )
        # Do I need the importance of the response?
        response_id = self.db.insert_text(
            output.response,
            timestamp=datetime.now()
        )
        
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
