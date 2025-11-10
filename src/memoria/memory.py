from functools import cached_property
from heapq import heapify, heappop, heappush
from typing import Annotated, Callable, Literal, cast, overload, override
from collections.abc import Iterable
from typing_extensions import ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, TypeAdapter
from pydantic_core import CoreSchema, core_schema

from ipld import dagcbor, IPLData
from cid import CID, CIDv1

from memoria.util import Least, Lexicographic, json_t

__all__ = (
    'IPLDModel', 'IPLDRoot',
    'BaseMemory', 'PartialMemory', 'Memory',
    'MemoryDAG'
)

class IPLDModel(BaseModel):
    '''Base model for IPLD objects.'''

    def ipld_model(self) -> IPLData:
        '''Return the object as an IPLD model.'''
        return self.model_dump()

class IPLDRoot(IPLDModel):
    '''Base model for IPLD objects which can get a CID.'''

    def ipld_block(self) -> bytes:
        '''Return the object as an IPLD block.'''
        return dagcbor.marshal(self.ipld_model())

    @cached_property
    def cid(self):
        return CIDv1.hash(self.ipld_block())

class TextPart(IPLDModel):
    '''External text memory.'''
    kind: Literal["text"] = "text"
    content: str
    
class FilePart(IPLDModel):
    '''A file part containing a file CID and metadata about it.'''
    kind: Literal["file"] = "file"
    file: CID
    filename: str | None = None
    mimetype: str
    filesize: int

class ThinkPart(IPLDModel):
    '''Opaque provider-specific thoughts by the model.'''
    kind: Literal["think"] = "think"
    content: str
    think_id: str | None = None
    signature: str | None = None

    @override
    def ipld_model(self) -> IPLData:
        data = {
            "kind": self.kind,
            "content": self.content
        }
        if self.think_id: data["think_id"] = self.think_id
        if self.signature: data["signature"] = self.signature

        return data

class ToolPart(IPLDModel):
    '''Tool call paired with its results.'''
    kind: Literal["tool"] = "tool"
    name: str
    args: dict[str, json_t]
    result: json_t = None
    call_id: str | None = None

    @override
    def ipld_model(self) -> IPLData:
        data = {
            "kind": self.kind,
            "name": self.name,
            "args": self.args,
            "result": self.result
        }
        if self.call_id: data["call_id"] = self.call_id

        return data

# Memories contain their metadata and a type-erased data payload.
# This data is organized between trusted and verified parts.
# Trusted parts are those which won't contain malicious content.
# Verified parts are those which can be assumed to be true.
# 
# OtherData: Not verified and not trusted. From outside, no telling what it is.
# SelfData: Not verified but trusted. Thoughts could be wrong but not malicious.
# MetaData: Verified but not trusted. Metadata is considered accurate but may contain malicious content eg the name of an IRC channel.
# SystemData: Verified but trusted. Used for eg system prompts, things which are axiomatically true and trustworthy.
# 
# The point of memory data, including metadata, isn't to create a convenient
# index for accessing nodes. It's to make the construction of such indexes
# *feasible*. They should always add information which, were they missing, could
# not be reconstructed without external information. So for instance, an
# interpreter might insert metadata for the username which originated the memory.
# All memories with the same such user would link to it, and thus an index could
# be constructed by following those links.

class OtherData(IPLDModel):
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

class SelfData(IPLDModel):
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

    @override
    def ipld_model(self):
        '''Return the object as an IPLD model.'''
        return {
            "kind": self.kind,
            "parts": [part.ipld_model() for part in self.parts],
            "model": self.model
        }

class MetaData(IPLDModel):
    '''
    A memory which is just metadata. Used primarily for coordinating context
    like provenance. This data should be understood as verified but not trusted.
    '''
    kind: Literal["meta"] = "meta"
    metadata: dict[str, json_t]

class SystemData(IPLDModel):
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

class BaseMemory[D: MemoryData=MemoryData](BaseModel):
    '''Base memory model.'''

    uuid: UUID | None = None
    '''
    UUID to act as a proxy for recency. It should only be None for metadata
    and system prompts. That allows them to be linked by CID without knowing
    the exact UUID.
    '''
    data: D
    '''Data contained in the memory.'''
    edges: set[CIDv1]
    '''Edges to other memories.'''

class PartialMemory[D: MemoryData=MemoryData](BaseMemory[D]):
    '''
    A memory which is complete but does not have its full contents. The
    CID can't be calculated and must be provided explicitly.
    '''
    cid: CIDv1

    def partial(self):
        return self
    
    def complete(self) -> 'Memory[D] | None':
        '''
        Attempt to coerce the memory into a complete memory.
        Returns None if the edges are incomplete.
        '''
        m = Memory(uuid=self.uuid, data=self.data, edges=self.edges)
        return m if self.cid == m.cid else None

class Memory[D: MemoryData=MemoryData](BaseMemory[D], IPLDRoot):
    '''A completed memory which can be referred to by CID.'''

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    @override
    def ipld_model(self) -> IPLData:
        # Edges must be sorted by target CID to ensure deterministic ordering
        model = {
            "data": self.data.ipld_model(),
            "edges": sorted(self.edges),
        }
        if (u := self.uuid) is not None:
            model["uuid"] = u.bytes
        return model

    def partial(self, edge_filter: Callable[[CIDv1], bool] | None) -> PartialMemory[D]:
        '''Return a PartialMemory with the same data and edges.'''
        return PartialMemory(
            cid=self.cid,
            uuid=self.uuid,
            data=self.data,
            edges=set(filter(edge_filter, self.edges))
        )

_default = object()

class _SimpleNode:
    def __init__(self, value: int):
        self.value: int = value
        self.edges: set[CIDv1] = set()

class MemoryDAG:
    '''Subgraph of the memory DAG.'''

    adj: dict[CIDv1, PartialMemory]
    
    def __init__(self, adj: dict[CIDv1, PartialMemory] | None = None):
        super().__init__()
        self.adj = {} if adj is None else adj

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: object, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        adj_schema = handler(dict[CIDv1, PartialMemory])
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
    
    def __contains__(self, key: CIDv1) -> bool:
        return key in self.adj
    
    def __bool__(self) -> bool:
        return bool(self.adj)
    
    def __len__(self) -> int:
        return len(self.adj)
    
    def __setitem__(self, key: CIDv1, value: PartialMemory):
        self.adj[key] = value

    def __getitem__(self, key: CIDv1) -> PartialMemory:
        '''Get the value of a node.'''
        return self.get(key)
    
    def __iter__(self):
        return iter(self.adj)

    @overload
    def get(self, key: CIDv1, /) -> PartialMemory: ...
    @overload
    def get[D](self, key: CIDv1, /, default: D) -> PartialMemory | D: ...

    def get[D](self, key: CIDv1, /, default: D = _default) -> PartialMemory | D:
        '''Get the value of a node, or return default if not found.'''
        val = self.adj.get(key, default)
        if val is _default:
            raise KeyError(key)
        return val

    def insert(self, key: CIDv1, value: PartialMemory):
        if key not in self.adj:
            self.adj[key] = value

    @overload
    def edges(self, node: CIDv1, /) -> Iterable[CIDv1]: ...
    @overload
    def edges[D](self, node: CIDv1, /, default: D) -> Iterable[CIDv1] | D: ...
    
    def edges[D](self, node: CIDv1, /, default: D = _default) -> Iterable[CIDv1] | D:
        '''Iterate over the edges of a node.'''
        n = self.adj.get(node, default)
        if n is _default:
            if default is _default:
                raise KeyError(node)
            return default
        return cast(PartialMemory, n).edges

    def add_edge(self, src: CIDv1, dst: CIDv1):
        if src not in self.adj:
            raise KeyError(f"Source {src} not found")
        if dst not in self.adj:
            raise KeyError(f"Destination {dst} not found")
        if src == dst:
            raise ValueError("Cannot add self-loop edge")
        
        self.adj[src].edges.add(dst)

    def keys(self) -> Iterable[CIDv1]:
        return self.adj.keys()
    
    def values(self) -> Iterable[PartialMemory]:
        return self.adj.values()

    def items(self) -> Iterable[tuple[CIDv1, PartialMemory]]:
        return self.adj.items()

    def has_edge(self, src: CIDv1, dst: CIDv1) -> bool:
        '''Check if there is an edge from src to dst.'''
        return dst in self.edges(src, ())

    def copy(self):
        '''Deepy copy of the graph.'''
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

    def toposort(self, key: Callable[[PartialMemory], Lexicographic | None] | None=None) -> Iterable[CIDv1]:
        '''
        Kahn's algorithm for topological sorting.

        :param key: Optional function to determine the lexicographical order of nodes.
        '''

        if key is None:
            key = lambda v: Least
        else:
            # Replace None with Least in the key function
            _oldkey = key
            def _key(v: PartialMemory) -> Lexicographic:
                ok = _oldkey(v)
                return Least if ok is None else ok
            key = _key
        
        # Build the in-degree graph
        indeg = dict[CIDv1, _SimpleNode]()
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
