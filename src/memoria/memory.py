'''
The point of memory data, including metadata, isn't to create a convenient
index for accessing nodes. It's to make the construction of such indexes
*feasible*. They should always add information which, were they missing, could
not be reconstructed without external information.

We can put memories on axes of trusted (non-malicious) and verified (accurate).
For now we just have Other (neither trusted nor verified) and Self (trusted
but not verified).
'''

from heapq import heapify, heappop, heappush
from typing import Annotated, Callable, Literal, cast, overload
from collections.abc import Iterable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from memoria.util import Least, Lexicographic, ReverseCmp, json_t

__all__ = (
    'TextPart', 'FilePart', 'ThinkPart', 'ToolPart',
    'SelfMemory', 'OtherMemory', 'Memory', 'MemoryAdapter',
    'MemoryDAG'
)

class TextPart(BaseModel):
    '''External text memory.'''
    kind: Literal["text"] = "text"
    content: str

class FilePart(BaseModel):
    '''A file part containing a UUID and metadata.'''
    kind: Literal["file"] = "file"
    file: UUID
    mimetype: str

class ThinkPart(BaseModel):
    '''Opaque provider-specific thoughts by the model.'''
    kind: Literal["think"] = "think"
    content: str
    think_id: str | None = None
    signature: str | None = None

class ToolPart(BaseModel):
    '''Tool call paired with its results.'''
    kind: Literal["tool"] = "tool"
    name: str
    args: dict[str, json_t]
    result: json_t
    call_id: str

class BaseMemory(BaseModel):
    '''An immutable memory in the DAG.'''

    model_config = ConfigDict(frozen=True)
    __pydantic_extra__: dict[str, json_t] = Field(init=False) # pyright: ignore[reportIncompatibleVariableOverride]

    uuid: UUID
    '''Unique identifier for this memory.'''
    metadata: dict[str, json_t] = Field(default_factory=dict)
    '''Auxiliary metadata.'''
    edges: set[UUID]
    '''Edges to other memories.'''

class SelfMemory(BaseMemory):
    '''The agent's own inner monologue. Not verified but trusted.'''

    type Part = Annotated[
        TextPart | FilePart | ThinkPart | ToolPart,
        Field(discriminator="kind")
    ]

    kind: Literal["self"] = "self"
    parts: list[Part]

class OtherMemory(BaseMemory):
    '''Data from another agent. Neither verified nor trusted.'''

    type Part = Annotated[
        TextPart | FilePart,
        Field(discriminator="kind")
    ]

    kind: Literal["other"] = "other"
    parts: list[Part]

type Memory = Annotated[
    SelfMemory | OtherMemory,
    Field(discriminator="kind")
]

MemoryAdapter = TypeAdapter[Memory](Memory)

_default = object()

class _SimpleNode:
    def __init__(self, value: int):
        self.value: int = value
        self.edges: set[UUID] = set()

class MemoryDAG:
    '''Subgraph of the memory DAG.'''

    adj: dict[UUID, Memory]

    def __init__(self, adj: dict[UUID, Memory] | None = None):
        super().__init__()
        self.adj = {} if adj is None else adj

    def __contains__(self, key: UUID) -> bool:
        return key in self.adj

    def __bool__(self) -> bool:
        return bool(self.adj)

    def __len__(self) -> int:
        return len(self.adj)

    def __setitem__(self, key: UUID, value: Memory):
        self.adj[key] = value

    def __getitem__(self, key: UUID) -> Memory:
        '''Get the value of a node.'''
        return self.get(key)

    def __iter__(self):
        return iter(self.adj)

    @overload
    def get(self, key: UUID, /) -> Memory: ...
    @overload
    def get[D](self, key: UUID, /, default: D) -> Memory | D: ...

    def get[D](self, key: UUID, /, default: D = None) -> Memory | D:
        '''Get the value of a node, or return default if not found.'''
        return self.adj.get(key, default)

    def insert(self, key: UUID, value: Memory):
        if key not in self.adj:
            self.adj[key] = value

    @overload
    def edges(self, node: UUID, /) -> Iterable[UUID]: ...
    @overload
    def edges[D](self, node: UUID, /, default: D) -> Iterable[UUID] | D: ...

    def edges[D](self, node: UUID, /, default: D = _default) -> Iterable[UUID] | D:
        '''Iterate over the edges of a node.'''
        n = self.adj.get(node, default)
        if n is _default:
            if default is _default:
                raise KeyError(node)
            return default
        return cast(Memory, n).edges

    def add_edge(self, src: UUID, dst: UUID):
        if src not in self.adj:
            raise KeyError(f"Source {src} not found")
        if dst not in self.adj:
            raise KeyError(f"Destination {dst} not found")
        if src == dst:
            raise ValueError("Cannot add self-loop edge")

        self.adj[src].edges.add(dst)

    def keys(self) -> Iterable[UUID]:
        return self.adj.keys()

    def values(self) -> Iterable[Memory]:
        return self.adj.values()

    def items(self) -> Iterable[tuple[UUID, Memory]]:
        return self.adj.items()

    def has_edge(self, src: UUID, dst: UUID) -> bool:
        '''Check if there is an edge from src to dst.'''
        return dst in self.edges(src, ())

    def clear(self):
        self.adj = {}

    def update(self, other: 'MemoryDAG'):
        self.adj.update(other.adj)

    def copy(self):
        '''Deep copy of the graph.'''
        g = type(self)()
        for src, node in self.adj.items():
            g.insert(src, node)
            for dst in node.edges:
                g.insert(dst, self.adj[dst])
                g.add_edge(src, dst)
        return g

    def invert(self):
        '''Invert the graph, reversing all edges.'''
        g = type(self)()
        for src, node in self.adj.items():
            g.insert(src, node)
            for dst in node.edges:
                g.insert(dst, self.adj[dst])
                g.add_edge(dst, src)
        return g

    def toposort(self, key: Callable[[Memory], Lexicographic | None] | None=None, reverse: bool=False) -> Iterable[UUID]:
        '''
        Kahn's algorithm for topological sorting.

        :param key: Optional function to determine the lexicographical order of nodes.
        :param reverse: Reverse the order of the sort.
        '''

        if key is None:
            key = lambda _: Least
        else:
            # Replace None with Least in the key function
            _oldkey1 = key
            def _key1(v: Memory) -> Lexicographic:
                ok = _oldkey1(v)
                return Least if ok is None else ok
            key = _key1

        if reverse:
            _oldkey2 = key
            def _key2(v: Memory) -> Lexicographic:
                return ReverseCmp(_oldkey2(v))
            key = _key2

        # Build the in-degree graph
        indeg = dict[UUID, _SimpleNode]()
        for src in self:
            if src not in indeg:
                indeg[src] = _SimpleNode(0)

            for dst in self.edges(src):
                if dst in indeg:
                    indeg[dst].value += 1
                else:
                    indeg[dst] = _SimpleNode(1)
                indeg[src].edges.add(dst)

        # Initialize the source heap with key as its tie-breaker
        sources = [
            (key(self[src]), src)
                for src, deg in indeg.items()
                    if deg.value == 0
        ]
        heapify(sources)
        # Pop source nodes until the heap is empty
        while sources:
            _, src = heappop(sources)
            yield src
            # Check for new sources
            for dst in indeg[src].edges:
                indeg[dst].value -= 1
                if indeg[dst].value == 0:
                    heappush(sources, (key(self[dst]), dst))
