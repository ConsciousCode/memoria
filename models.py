from functools import cached_property
from typing import Annotated, Iterable, Literal, Optional, overload, override
from uuid import UUID

from pydantic import BaseModel, Field, PlainSerializer, TypeAdapter

from graph import IGraph
from ipld import ipld
from ipld.cid import CIDv1, cidhash

type MemoryKind = Literal["self", "other", "text", "image", "file"]
type UUIDCID = Annotated[CIDv1,
    PlainSerializer(lambda u: cidhash(u.bytes, codec='raw'))
]
type StopReason = Literal["end", "error", "cancel"]

class RecallConfig(BaseModel):
    '''Configuration for how to weight memory recall.'''
    importance: Annotated[
        Optional[float],
        Field(description="Weight of memory importance.")
    ]=None
    recency: Annotated[
        Optional[float],
        Field(description="Weight of the recency of the memory.")
    ]=None
    sona: Annotated[
        Optional[float],
        Field(description="Weight of the sona relevance.")
    ]=None
    fts: Annotated[
        Optional[float],
        Field(description="Weight of the ull-text search relevance.")
    ]=None
    vss: Annotated[
        Optional[float],
        Field(description="Weight of the vector similarity.")
    ]=None
    k: Annotated[
        Optional[int],
        Field(description="Number of memories to return. 20 by default.")
    ]=None

class IPLDModel(BaseModel):
    '''Base model for IPLD objects.'''
    @cached_property
    def cid(self):
        return cidhash(ipld.dagcbor_marshal(self.model_dump()))

class Edge[T](BaseModel):
    '''Edge from one memory to another.'''
    target: T
    weight: float

class BaseMemory(IPLDModel):
    timestamp: Optional[float] = None
    edges: list[Edge[CIDv1]] = Field(
        default_factory=list,
        description="Edges to other memories."
    )
    importance: Optional[float] = Field(exclude=True, default=None)

    def edge(self, target: CIDv1) -> Optional[Edge[CIDv1]]:
        '''Get the edge to the target memory, if it exists.'''
        
        for edge in self.edges:
            if edge.target == target:
                return edge
        return None
    
    def insert_edge(self, target: CIDv1, weight: float):
        '''Insert an edge to the target memory with the given weight.'''
        if self.edge(target) is not None:
            raise ValueError(f"Edge to {target} already exists")
        
        self.edges.append(Edge(
            target=target,
            weight=weight
        ))
    
    @cached_property
    @override
    def cid(self) -> CIDv1:
        # Edges must be sorted by target CID to ensure deterministic ordering
        self.edges.sort(key=lambda e: e.target)
        return super().cid

class SelfMemory(BaseMemory):
    class Data(BaseModel):
        class Part(BaseModel):
            content: str = ""
            model: Optional[str] = None
        
        name: Optional[str] = None
        parts: list[Part]
        stop_reason: Optional[StopReason] = None
    
    kind: Literal["self"] = "self"
    data: Data

class OtherMemory(BaseMemory):
    class Data(BaseModel):
        name: Optional[str] = None
        content: str
    
    kind: Literal["other"] = "other"
    data: Data

class TextMemory(BaseMemory):
    type Data = str
    kind: Literal["text"] = "text"
    data: Data

class FileMemory(BaseMemory):
    class Data(BaseModel):
        name: Annotated[Optional[str],
            Field(description="Name of the file at time of upload, if available.")
        ] = None
        content: Annotated[str,
            Field(description="Base64 encoded file contents.")
        ]
        mimeType: Optional[str] = None

    kind: Literal["file"] = "file"
    data: Data

type Memory = Annotated[
    SelfMemory | OtherMemory | TextMemory | FileMemory,
    Field(discriminator="kind")
]
type MemoryData = SelfMemory.Data | OtherMemory.Data | TextMemory.Data | FileMemory.Data
MemoryDataAdapter = TypeAdapter[MemoryData](MemoryData)

@overload
def build_memory(kind: Literal['self'], data: SelfMemory.Data, timestamp: Optional[float], edges: Optional[dict[CIDv1, float]]=None) -> SelfMemory: ...
@overload
def build_memory(kind: Literal['other'], data: OtherMemory.Data, timestamp: Optional[float], edges: Optional[dict[CIDv1, float]]=None) -> OtherMemory: ...
@overload
def build_memory(kind: Literal['text'], data: TextMemory.Data, timestamp: Optional[float], edges: Optional[dict[CIDv1, float]]=None) -> TextMemory: ...
@overload
def build_memory(kind: Literal['file'], data: FileMemory.Data, timestamp: Optional[float], edges: Optional[dict[CIDv1, float]]=None) -> FileMemory: ...
@overload
def build_memory(kind: MemoryKind, data: MemoryData, timestamp: Optional[float], edges: Optional[dict[CIDv1, float]]=None) -> Memory: ...

def build_memory(kind: MemoryKind, data: MemoryData, timestamp: Optional[float]=None, edges: Optional[dict[CIDv1, float]]=None) -> Memory:
    '''Build a memory object from the given data.'''
    args = {
        "data": data,
        "timestamp": timestamp
    }
    if edges: args['edges'] = [
        Edge(target=k, weight=v) for k, v in edges.items()
    ]
    match kind:
        case "self": return SelfMemory(**args)
        case "other": return OtherMemory(**args)
        case "text": return TextMemory(**args)
        case "file": return FileMemory(**args)
        case _: raise ValueError(f"Unknown memory kind: {kind}")

def memory_document(memory: Memory) -> str:
    '''Construct a document for FTS from a memory.'''
    match memory.kind:
        case "self": return "".join(part.content for part in memory.data.parts)
        case "other": return memory.data.content
        case "text": return memory.data
        case "file": return memory.data.content
        case _: raise ValueError(f"Unknown memory kind: {memory.kind}")

class IncompleteACThread(IPLDModel):
    '''A thread of memories in the agent's context.'''
    memory: Memory # Memory is incomplete so it can't be referenced by CID
    prev: Optional[CIDv1] = None

class Sona(BaseModel):
    '''
    Sona model for returning from memory queries. It is not immutable and
    is thus referred to by UUID rather than CID.
    '''
    uuid: UUID = Field(
        description="Unique identifier for the Sona."
    )
    aliases: list[str] = Field(
        description="List of aliases for the Sona."
    )
    active: Optional[IncompleteACThread] = Field(
        default=None,
        description="Active thread for the Sona, if any."
    )
    pending: Optional[IncompleteACThread] = Field(
        default=None,
        description="Pending thread for the Sona, if any."
    )

class ACThread(IPLDModel):
    '''A thread of memories in the agent's context.'''
    sona: UUIDCID
    memory: CIDv1
    prev: Optional[CIDv1] = None

class MemoryDAG(IGraph[CIDv1, float, Memory, Memory]):
    '''IPLD data model for memories implementing the IGraph interface.'''
    @override
    def _node(self, value: Memory) -> Memory:
        return value
    
    @override
    def _setvalue(self, node: Memory, value: Memory):
        # We're assigning the discriminant with the data together so despite
        #  not being technically correct, there is no observable type violations
        node.kind = value.kind # type: ignore
        node.data = value.data # type: ignore
        node.timestamp = value.timestamp
        node.edges = value.edges
    
    @override
    def _valueof(self, node: Memory) -> Memory:
        return node
    
    @override
    def _edges(self, node: Memory) -> Iterable[tuple[CIDv1, float]]:
        for edge in node.edges:
            yield edge.target, edge.weight
        return node.edges

    @override
    def _add_edge(self, src: Memory, dst: CIDv1, edge: float):
        src.edges.append(Edge(
            target=dst,
            weight=edge
        ))
    
    @override
    def _pop_edge(self, src: Memory, dst: CIDv1) -> Optional[float]:
        edges = src.edges
        for i, edge in enumerate(edges):
            if edge.target == dst:
                del edges[i]
                return edge.weight
        return None