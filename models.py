from collections import defaultdict
from functools import cached_property
from typing import Annotated, Any, Iterable, Literal, Optional, overload, override
from cid import CIDv1
from pydantic import BaseModel, Field, PlainSerializer, TypeAdapter

from graph import IGraph
import ipld

type MemoryKind = Literal["self", "other", "text", "image", "file"]
type UUIDCID = Annotated[CIDv1, PlainSerializer(lambda u: CIDv1("raw", u.bytes))]

def build_cid(data) -> CIDv1:
    return CIDv1("dag-cbor", ipld.multihash(ipld.dagcbor_marshal(data)))

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
    @cached_property
    def cid(self):
        return CIDv1("dag-cbor",
            ipld.multihash(ipld.dagcbor_marshal(self.model_dump()))
        )

class Edge(BaseModel):
    weight: float
    target: CIDv1

class BaseMemory(IPLDModel):
    timestamp: Optional[float] = None
    edges: dict[str, list[Edge]] = Field(
        default_factory=lambda: defaultdict(list)
    )  # label: [(edge, target), ...]
    importance: Optional[float] = Field(exclude=True, default=None)

class SelfMemory(BaseMemory):
    class Data(BaseModel):
        name: Optional[str] = None
        model: Optional[str] = None
        content: str
    
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

type Memory = SelfMemory | OtherMemory | TextMemory | FileMemory
type MemoryData = SelfMemory.Data | OtherMemory.Data | TextMemory.Data | FileMemory.Data
MemoryDataAdapter = TypeAdapter[MemoryData](MemoryData)

@overload
def build_memory(kind: Literal['self'], data: SelfMemory.Data, timestamp: Optional[float], edges: Optional[dict[str, list[Edge]]]=None) -> SelfMemory: ...
@overload
def build_memory(kind: Literal['other'], data: OtherMemory.Data, timestamp: Optional[float], edges: Optional[dict[str, list[Edge]]]=None) -> OtherMemory: ...
@overload
def build_memory(kind: Literal['text'], data: TextMemory.Data, timestamp: Optional[float], edges: Optional[dict[str, list[Edge]]]=None) -> TextMemory: ...
@overload
def build_memory(kind: Literal['file'], data: FileMemory.Data, timestamp: Optional[float], edges: Optional[dict[str, list[Edge]]]=None) -> FileMemory: ...
@overload
def build_memory(kind: MemoryKind, data: MemoryData, timestamp: Optional[float], edges: Optional[dict[str, list[Edge]]]=None) -> Memory: ...

def build_memory(kind: MemoryKind, data: MemoryData, timestamp: Optional[float]=None, edges: Optional[dict[str, list[Edge]]]=None) -> Memory:
    args = {
        "data": data,
        "timestamp": timestamp
    }
    if edges: args['edges'] = edges
    match kind:
        case "self":  return SelfMemory(**args)
        case "other": return OtherMemory(**args)
        case "text":  return TextMemory(**args)
        case "file":  return FileMemory(**args)
        case _: raise ValueError(f"Unknown memory kind: {kind}")

def model_dump(obj: Any) -> Any:
    try:
        return obj.model_dump()
    except AttributeError:
        return obj

class ACThread(IPLDModel):
    sona: UUIDCID
    memory: CIDv1
    prev: Optional[CIDv1] = None

class DAGEdge(BaseModel):
    '''Edge for use in graph operations.'''
    label: str
    weight: float

class MemoryDAG(IGraph[CIDv1, DAGEdge, Memory, Memory]):
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
    def _edges(self, node: Memory) -> Iterable[tuple[CIDv1, DAGEdge]]:
        for label, edges in node.edges.items():
            for edge in edges:
                yield edge.target, DAGEdge(label=label, weight=edge.weight)
        return node.edges

    @override
    def _add_edge(self, src: Memory, dst: CIDv1, edge: DAGEdge):
        src.edges[edge.label].append(Edge(
            weight=edge.weight,
            target=dst
        ))
    
    @override
    def _pop_edge(self, src: Memory, dst: CIDv1) -> Optional[DAGEdge]:
        for label, edges in src.edges.items():
            for i, edge in enumerate(edges):
                if edge.target == dst:
                    del edges[i]
                    return DAGEdge(label=label, weight=edge.weight)
        return None