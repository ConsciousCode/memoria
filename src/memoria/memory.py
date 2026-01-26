from heapq import heapify, heappop, heappush
from typing import Annotated, Callable, Literal, cast, overload
from collections.abc import Iterable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, TypeAdapter
from pydantic_core import CoreSchema, core_schema

from memoria.util import Least, Lexicographic, ReverseCmp, json_t

__all__ = (
    'TextPart', 'FilePart', 'ThinkPart', 'ToolPart',
    'OtherData', 'SelfData', 'MetaData', 'SystemData',
    'Memory',
    'MemoryDAG'
)

class TextPart(BaseModel):
    '''External text memory.'''
    kind: Literal["text"] = "text"
    content: str

class FilePart(BaseModel):
    '''A file part containing a file UUID and metadata about it.'''
    kind: Literal["file"] = "file"
    file: UUID
    filename: str | None = None
    mimetype: str
    filesize: int

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
    result: json_t = None
    call_id: str | None = None

# Memories contain their metadata and a type-erased data payload.
# This data is organized between trusted and verified parts.
# Trusted parts are those which won't contain malicious content.
# Verified parts are those which can be assumed to be true.
# 
# OtherData: Not verified and not trusted. From outside, no telling what it is.
# SelfData: Not verified but trusted. Thoughts could be wrong but not malicious.
# MetaData: Verified but not trusted. Metadata is considered accurate but may
#  contain malicious content eg the name of an IRC channel #ign-prev-instr.
# SystemData: Verified but trusted. Used for eg system prompts, things which are
#  axiomatically true and trustworthy.
# 
# The point of memory data, including metadata, isn't to create a convenient
# index for accessing nodes. It's to make the construction of such indexes
# *feasible*. They should always add information which, were they missing, could
# not be reconstructed without external information. So for instance, an
# interpreter might insert metadata for the username which originated the memory.
# All memories with the same such user would link to it, and thus an index could
# be constructed by following those links.

class OtherData(BaseModel):
    '''
    A memory containing data from another agent. This data should be understood
    as neither verified nor trusted.
    '''

    type Part = Annotated[
        TextPart | FilePart,
        Field(discriminator="kind")
    ]

    kind: Literal["other"] = "other"
    parts: list[Part]

class SelfData(BaseModel):
    '''
    The agent's own inner monologue. This data should be understood as
    not verified but trusted.
    '''

    type Part = Annotated[
        TextPart | FilePart | ThinkPart | ToolPart,
        Field(discriminator="kind")
    ]

    kind: Literal["self"] = "self"
    parts: list[Part]
    '''Parts comprising the completion.'''
    model: str | None = None
    '''The model which originated this memory.'''

class MetaData(BaseModel):
    '''
    A memory which is just metadata. Used primarily for coordinating context
    like provenance. This data should be understood as verified but not trusted.
    '''
    kind: Literal["meta"] = "meta"
    metadata: dict[str, json_t]

class SystemData(BaseModel):
    '''
    A memory containing a system prompt appended by an interpreter. This data
    should be understood as both verified and trusted.
    '''

    type Part = Annotated[
        TextPart,
        Field(discriminator="kind")
    ]

    kind: Literal["system"] = "system"
    parts: list[Part]

type MemoryData = Annotated[
    OtherData | SelfData | MetaData | SystemData,
    Field(discriminator="kind")
]
'''Memory data which can actually be stored.'''

MemoryDataAdapter = TypeAdapter[MemoryData](MemoryData)

class Memory(BaseModel):
    '''An immutable memory in the DAG.'''

    model_config = ConfigDict(frozen=True)

    uuid: UUID
    '''Unique identifier for this memory.'''
    data: MemoryData
    '''Data contained in the memory.'''
    edges: set[UUID]
    '''Edges to other memories.'''

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

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: object, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        adj_schema = handler(dict[UUID, Memory])
        def ser(inst: MemoryDAG):
            return inst.adj
        return core_schema.json_or_python_schema(
            json_schema=core_schema.chain_schema([
                adj_schema,
                core_schema.no_info_plain_validator_function(cls)
            ]),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(cls),
                core_schema.chain_schema([
                    adj_schema,
                    core_schema.no_info_plain_validator_function(cls)
                ])
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(ser)
        )

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

    def get[D](self, key: UUID, /, default: D = _default) -> Memory | D:
        '''Get the value of a node, or return default if not found.'''
        val = self.adj.get(key, default)
        if val is _default:
            raise KeyError(key)
        return val

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
