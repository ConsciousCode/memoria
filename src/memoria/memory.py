from datetime import datetime
from functools import cached_property
from typing import Annotated, Iterable, Literal, Optional, override
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, TypeAdapter
from pydantic_core import CoreSchema, core_schema

from ipld import dagcbor, IPLData
from cid import CID, CIDv1

from graph import IntrusiveGraph
from memoria.util import json_t

__all__ = (
    'MemoryKind',
    'StopReason',
    'IPLDModel', 'IPLDRoot',
    'Edge', 'BaseMemory',
    'DraftMemory', 'PartialMemory', 'Memory',
    'CompleteMemory', 'AnyMemory',
    'AnyACThread', 'IncompleteACThread', 'ACThread',
    'Sona', 'MemoryDAG'
)

type MemoryKind = Literal["self", "other", "text", "image", "file", "metadata"]
type StopReason = Literal["endTurn", "stopSequence", "maxTokens"] | str

class IPLDModel(BaseModel):
    '''Base model for IPLD objects.'''

    def ipld_model(self) -> IPLData:
        '''Return the object as an IPLD model.'''
        return self.model_dump()

class IPLDRoot(IPLDModel):
    '''Base model faor IPLD objects which can get a CID.'''

    def ipld_block(self) -> bytes:
        '''Return the object as an IPLD block.'''
        return dagcbor.marshal(self.ipld_model())

    @cached_property
    def cid(self):
        return CIDv1.hash(self.ipld_block())

class Edge[T](BaseModel):
    '''Edge from one memory to another.'''
    target: T
    weight: float

class SelfData(IPLDModel):
    '''The agent's own transparent inner monologue.'''
    class TextPart(IPLDModel):
        '''Transparent thoughts by the model.'''
        kind: Literal["text"] = "text"
        content: str
    
    class ThinkPart(IPLDModel):
        '''Opaque thoughts by the model.'''
        kind: Literal["think"] = "think"
        content: str
        think_id: Optional[str] = None
        signature: Optional[str] = None

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
        call_id: Optional[str] = None

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

    type Part = Annotated[
        TextPart | ThinkPart | ToolPart,
        Field(discriminator="kind")
    ]

    kind: Literal["self"] = "self"
    parts: list[Part]
    '''Parts comprising the completion.'''
    model: Optional[str] = None
    '''The model which originated this memory.'''

    @override
    def ipld_model(self):
        '''Return the object as an IPLD model.'''
        return {
            "kind": self.kind,
            "parts": [part.ipld_model() for part in self.parts],
            "model": self.model
        }

class TextData(IPLDModel):
    '''External text memory.'''
    kind: Literal["text"] = "text"
    content: str

class FileData(IPLDModel):
    '''A file memory containing a file CID and metadata about it.'''
    kind: Literal["file"] = "file"
    file: CID
    filename: Optional[str] = None
    mimetype: str
    filesize: int

class MetaData(IPLDModel):
    '''
    A memory which is just metadata. Used primarily for coordinating context
    like provenance.
    '''
    kind: Literal["metadata"] = "metadata"
    metadata: dict[str, json_t]

type MemoryData = Annotated[
    SelfData | TextData | FileData | MetaData,
    Field(discriminator="kind")
]
'''Memory data which can actually be stored.'''

MemoryDataAdapter = TypeAdapter[MemoryData](MemoryData)

class BaseMemory[D: MemoryData=MemoryData](BaseModel):
    '''Base memory model.'''

    data: D
    '''Data contained in the memory.'''
    metadata: Optional[dict[str, json_t]] = None
    '''Freeform metadata about the memory.'''
    edges: list[Edge[CIDv1]]
    '''Edges to other memories.'''

    def edge(self, target: CIDv1) -> Optional[Edge[CIDv1]]:
        '''Get the edge to the target memory, if it exists.'''
        for edge in self.edges:
            if edge.target == target:
                return edge
        return None

    def has_edge(self, target: CIDv1):
        return any(target == e.target for e in self.edges)

class DraftMemory[D: MemoryData=MemoryData](BaseMemory[D]):
    '''
    A memory which is incomplete and in the process of being created.
    Cannot be assigned a CID.
    '''

    cid: None = None

    def insert_edge(self, target: CIDv1, weight: float):
        '''Insert an edge to the target memory with the given weight.'''
        if self.edge(target) is not None:
            raise ValueError(f"Edge to {target} already exists")
        
        self.edges.append(Edge(
            target=target,
            weight=weight
        ))

    def complete(self) -> 'Memory[D]':
        '''Complete the memory by adding edges and returning a Memory object.'''
        return Memory(data=self.data, edges=self.edges)

class PartialMemory[D: MemoryData=MemoryData](BaseMemory[D]):
    '''
    A memory which is complete but does not have its full contents. The
    CID can't be calculated and must be provided explicitly.
    '''
    cid: CIDv1

    def partial(self):
        return self

class Memory[D: MemoryData=MemoryData](BaseMemory[D], IPLDRoot):
    '''A completed memory which can be referred to by CID.'''

    model_config = ConfigDict(frozen=True)

    @override
    def ipld_model(self) -> IPLData:
        # Edges must be sorted by target CID to ensure deterministic ordering
        return {
            "data": self.data.ipld_model(),
            "edges": sorted(self.edges, key=lambda e: e.target),
        }
    
    def partial(self) -> PartialMemory[D]:
        '''Return a PartialMemory with the same data and edges.'''
        return PartialMemory(
            cid=self.cid,
            data=self.data,
            edges=self.edges
        )

type CompleteMemory[D: MemoryData=MemoryData] = PartialMemory[D] | Memory[D]
'''A memory with a CID.'''

type AnyMemory[D: MemoryData=MemoryData] = DraftMemory[D] | CompleteMemory[D]
'''Any kind of instantiable memory.'''

class MemoryContext[M: AnyMemory](BaseModel):
    memory: M
    '''The memory itself.'''
    
    timestamp: Optional[int] = None
    '''Timestamp of the memory, if any.'''
    importance: Optional[float] = None
    '''Importance of the memory, used for recall weighting.'''
    sonas: Optional[list[UUID|str]] = None
    '''Sonas the memory belongs to.'''

class IncompleteACThread[D: MemoryData=MemoryData](BaseModel):
    '''A thread of memories in the agent's context.'''
    cid: None = None # Incomplete threads can't have a cid
    sona: UUID
    memory: DraftMemory[D] # Can't be referred to by cid
    prev: Optional[CIDv1] = None

class ACThread(IPLDRoot):
    '''A thread of memories in the agent's context.'''
    sona: UUID
    memory: CIDv1
    prev: Optional[CIDv1] = None

    @override
    def ipld_model(self) -> IPLData:
        '''Return the thread as an IPLD model.'''
        return {
            "sona": str(self.sona),
            "memory": self.memory,
            "prev": self.prev
        }

type AnyACThread[D: MemoryData=MemoryData] = IncompleteACThread[D] | ACThread

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

class MemoryDAG(IntrusiveGraph[CIDv1, float, MemoryContext[PartialMemory]]):
    '''IPLD data model for memories implementing the IGraph interface.'''

    type Node = MemoryContext[PartialMemory]

    def __init__(self, keys: dict[CIDv1, Node]|None = None):
        super().__init__()
        self.adj = keys or {}

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        adj_schema = handler(dict[CIDv1, PartialMemory])
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
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: instance.adj
            )
        )
    
    @override
    def _node(self, value: Node) -> Node:
        return MemoryContext(
            memory=PartialMemory(
                cid=value.memory.cid,
                data=value.memory.data,
                edges=[]
            ),
            timestamp=value.timestamp,
            importance=value.importance,
            sonas=value.sonas
        )
    
    @override
    def _setvalue(self,  node: Node, value: Node):
        node.memory.cid = value.memory.cid
        node.memory.data = value.memory.data
        node.memory.edges = value.memory.edges
        node.timestamp = value.timestamp
        node.importance = value.importance
        node.sonas = value.sonas
    
    @override
    def _edges(self, node: Node) -> Iterable[tuple[CIDv1, float]]:
        for edge in node.memory.edges:
            yield edge.target, edge.weight

    @override
    def _add_edge(self, src: Node, dst: CIDv1, edge: float):
        es = src.memory.edges
        if not any(dst == e.target for e in es):
            es.append(Edge[CIDv1](
                target=dst,
                weight=edge
            ))
    
    @override
    def _pop_edge(self, src: Node, dst: CIDv1) -> Optional[float]:
        es = src.memory.edges
        for i, edge in enumerate(es):
            if edge.target == dst:
                del es[i]
                return edge.weight
        return None