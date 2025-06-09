from collections import defaultdict
from functools import cached_property
from typing import Annotated, Iterable, Literal, Optional, override
from uuid import UUID
from cid import CIDv1
from pydantic import BaseModel, Field, PlainSerializer

from graph import IGraph
import ipld
from util import json_t

type MemoryKind = Literal["self", "other", "text", "image", "file", "entity"]

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

class Edge(BaseModel):
    weight: float
    target: CIDv1

class SelfMemory(BaseModel):
    kind: Literal["self"] = "self"
    name: Optional[str] = None
    model: Optional[str] = None
    content: str

class OtherMemory(BaseModel):
    kind: Literal["other"] = "other"
    name: Optional[str] = None
    content: str

class TextMemory(BaseModel):
    kind: Literal["text"] = "text"
    content: str

class FileMemory(BaseModel):
    kind: Literal["file"] = "file"
    name: Annotated[Optional[str], Field(description="Name of the file at time of upload, if available.")] = None
    content: Annotated[str, Field(description="Base64 encoded file contents.")]
    mimeType: Optional[str] = None

class IPLDModel(BaseModel):
    @cached_property
    def cid(self):
        return CIDv1("dag-cbor",
            ipld.multihash(ipld.dagcbor_marshal(self.model_dump()))
        )

class Memory(IPLDModel):
    '''
    IPLD doesn't allow links as keys, so edges is label: [(edge, target), ...].
    This differs from Graph which uses the target as the key.
    '''
    kind: MemoryKind
    data: json_t
    timestamp: Optional[float]
    edges: dict[str, list[Edge]] = Field(
        default_factory=lambda: defaultdict(list)
    ) # label: [(edge, target), ...]

class ACThread(IPLDModel):
    sona: Annotated[UUID,
        PlainSerializer(lambda u: CIDv1("raw", u.bytes))
    ]
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
        node.kind = value.kind
        node.data = value.data
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