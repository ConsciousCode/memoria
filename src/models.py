from datetime import datetime
from functools import cached_property
from typing import Annotated, Iterable, Literal, Optional, overload, override
from uuid import UUID

from mcp.types import ModelPreferences, Role
from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from src.graph import IGraph
from src.ipld import dagcbor, CIDv1, cidhash

__all__ = (
    'MemoryKind',
    'StopReason',
    'RecallConfig', 'SampleConfig',
    'IPLDModel',
    'Edge', 'BaseMemory', 'LeafMemory', 'NodeMemory',
    'DraftMemory', 'IncompleteMemory', 'PartialMemory', 'Memory',
    'CompleteMemory', 'AnyMemory',
    'AnyACThread', 'IncompleteACThread', 'ACThread',
    'Sona', 'MemoryDAG'
)

type MemoryKind = Literal["self", "other", "text", "image", "file"]
type StopReason = Literal["endTurn", "stopSequence", "maxTokens"] | str

class RecallConfig(BaseModel):
    '''Configuration for how to weight memory recall.'''
    importance: Annotated[
        float,
        Field(description="Weight of memory importance.")
    ] = 0.30
    recency: Annotated[
        float,
        Field(description="Weight of the recency of the memory.")
    ] = 0.30
    sona: Annotated[
        float,
        Field(description="Weight of the sona relevance.")
    ] = 0.10
    fts: Annotated[
        float,
        Field(description="Weight of the full-text search relevance.")
    ] = 0.15
    vss: Annotated[
        float,
        Field(description="Weight of the vector similarity.")
    ] = 0.25
    k: Annotated[
        int,
        Field(description="Number of memories to return. 20 by default.")
    ] = 20
    decay: Annotated[
        Optional[float],
        Field(description="Time decay factor for recency. 0.995 by default.")
    ] = 0.995

class SampleConfig(BaseModel):
    '''Configuration for sampling responses.'''
    temperature: Annotated[
        Optional[float],
        Field(description="Sampling temperature for the response. If `null`, uses the default value.")
    ] = None
    max_tokens: Annotated[
        Optional[int],
        Field(description="Maximum number of tokens to generate in the response. If `null`, uses the default value.")
    ] = None
    model_preferences: Annotated[
        Optional[ModelPreferences | str | list[str]],
        Field(description="List of preferred models to use for the response. If `null`, uses the default model.")
    ] = None

class InsertMemory(BaseModel):
    '''Individual request to insert a memory into the system.'''
    
    kind: MemoryKind = Field(
        description='Role of the memory, e.g. "user", "assistant".'
    )
    name: Optional[str] = Field(
        default=None,
        description='Name of the memory, if available.'
    )
    model: Optional[str] = Field(
        default=None,
        description='Model used to generate the memory, if available.'
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description='Timestamp of the memory, if available.'
    )
    content: str = Field(
        description='Content of the memory. Right now just text.'
    )

class InsertConvo(BaseModel):
    '''Request to insert a sequence of memories.'''

    sona: Optional[UUID|str] = Field(
        default=None,
        description="Sona to insert the memories into."
    )
    prev: Optional[CIDv1] = Field(
        default=None,
        description="CID of the previous memory in the thread, if any."
    )
    chatlog: list[InsertMemory] = Field(
        description="Chatlog to insert into the system."
    )

class IPLDModel(BaseModel):
    '''Base model for IPLD objects.'''

    def as_block(self) -> bytes:
        '''Return the object as an IPLD block.'''
        return dagcbor.marshal(self.model_dump())

    @cached_property
    def cid(self):
        return cidhash(self.as_block())

class Edge[T](BaseModel):
    '''Edge from one memory to another.'''
    target: T
    weight: float

class BaseMemory(BaseModel):
    '''Base memory model.'''
    class SelfData(BaseModel):
        class Part(BaseModel):
            content: str = ""
            model: Optional[str] = None
        
        kind: Literal["self"] = "self"
        name: Optional[str] = None
        parts: list[Part]
        stop_reason: Optional[StopReason] = None

        def document(self):
            return "".join(part.content for part in self.parts)

    class OtherData(BaseModel):
        kind: Literal["other"] = "other"
        name: Optional[str] = None
        content: str

        def document(self):
            return self.content
    
    class TextData(BaseModel):
        kind: Literal["text"] = "text"
        content: str

        def document(self):
            return self.content
    
    class FileData(BaseModel):
        kind: Literal["file"] = "file"
        name: Annotated[Optional[str],
            Field(description="Name of the file at time of upload, if available.")
        ] = None
        content: Annotated[str,
            Field(description="Base64 encoded file contents.")
        ]
        mimeType: Optional[str] = None

        def document(self):
            return self.content
    
    type MemoryData = Annotated[
        SelfData | OtherData | TextData | FileData,
        Field(discriminator="kind")
    ]

    data: MemoryData
    timestamp: Optional[int] = None
    importance: Optional[float] = Field(
        default=None,
        exclude=True,
        description="Importance of the memory, used for recall weighting."
    )
    sonas: Optional[list[UUID|str]] = Field(
        default=None,
        exclude=True,
        description="Sonas the memory belongs to."
    )

    def document(self) -> str:
        return self.data.document()

    @overload
    @classmethod
    def build_data(cls, kind: Literal["self"], json: str) -> SelfData: ...
    @overload
    @classmethod
    def build_data(cls, kind: Literal["other"], json: str) -> OtherData: ...
    @overload
    @classmethod
    def build_data(cls, kind: Literal["text"], json: str) -> TextData: ...
    @overload
    @classmethod
    def build_data(cls, kind: Literal["image", "file"], json: str) -> FileData: ...

    @classmethod
    def build_data(cls, kind: MemoryKind, json: str) -> MemoryData:
        '''Build memory data based on the kind.'''
        match kind:
            case "self": return cls.SelfData.model_validate_json(json)
            case "other": return cls.OtherData.model_validate_json(json)
            case "text": return cls.TextData.model_validate_json(json)
            case "image" | "file": return cls.FileData.model_validate_json(json)
            case _:
                raise ValueError(f"Unknown memory kind: {kind}")

class LeafMemory(BaseMemory):
    '''A memory which is a leaf in the memory graph, i.e. has no edges.'''

class NodeMemory(BaseMemory):
    '''A memory which is a node in the memory graph, i.e. has edges.'''

    edges: list[Edge[CIDv1]] = Field(
        default_factory=list,
        description="Edges to other memories."
    )

    def edge(self, target: CIDv1) -> Optional[Edge[CIDv1]]:
        '''Get the edge to the target memory, if it exists.'''
        for edge in self.edges:
            if edge.target == target:
                return edge
        return None

    def has_edge(self, target: CIDv1):
        return any(target == e.target for e in self.edges)

    def insert_edge(self, target: CIDv1, weight: float):
        '''Insert an edge to the target memory with the given weight.'''
        if self.edge(target) is None:
            self.edges.append(Edge(
                target=target,
                weight=weight
            ))

class DraftMemory(NodeMemory):
    '''A memory which cannot derive a CID.'''
    
    def insert_edge(self, target: CIDv1, weight: float):
        '''Insert an edge to the target memory with the given weight.'''
        if self.edge(target) is not None:
            raise ValueError(f"Edge to {target} already exists")
        
        self.edges.append(Edge(
            target=target,
            weight=weight
        ))

class IncompleteMemory(DraftMemory):
    '''
    A memory which is incomplete and in the process of being created.
    Cannot be assigned a CID.
    '''

    cid: None = None

    def complete(self) -> 'Memory':
        '''Complete the memory by adding edges and returning a Memory object.'''
        return Memory(
            data=self.data,
            timestamp=self.timestamp,
            edges=self.edges
        )

class PartialMemory(DraftMemory):
    '''
    A memory which is complete but does not have its full contents. Thus
    the CID can't be calculated from it and must be provided.
    '''
    cid: CIDv1

    def partial(self):
        return self

class Memory(NodeMemory, IPLDModel):
    '''A completed memory which can be referred to by CID.'''

    model_config = ConfigDict(frozen=True)

    @override
    def as_block(self) -> bytes:
        # Edges must be sorted by target CID to ensure deterministic ordering
        self.edges.sort(key=lambda e: e.target)
        return super().as_block()
    
    def partial(self):
        '''Return a PartialMemory with the same data and edges, but without the CID.'''
        return PartialMemory(
            data=self.data,
            timestamp=self.timestamp,
            edges=self.edges,
            cid=self.cid
        )

'''A memory which may or may not have a CID.'''
type CompleteMemory = PartialMemory | Memory
'''A memory with a CID.'''
type AnyMemory = IncompleteMemory | PartialMemory | Memory

class IncompleteACThread(BaseModel):
    '''A thread of memories in the agent's context.'''
    cid: None = None # Incomplete threads can't have a cid
    sona: UUID
    memory: IncompleteMemory # Can't be referred to by cid
    prev: Optional[CIDv1] = None

class ACThread(IPLDModel):
    '''A thread of memories in the agent's context.'''
    sona: UUID
    memory: CIDv1
    prev: Optional[CIDv1] = None

type AnyACThread = IncompleteACThread | ACThread

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

    def human_json(self):
        '''Return a human-readable JSON representation of the Sona.'''
        # Need to do this separately because if we use model_dump(),
        # it will trigger UUID's serialization as a CID and lose the
        # human-readable UUID.
        return {
            "uuid": str(self.uuid),
            "aliases": self.aliases,
            "active": self.active and self.active.model_dump(),
            "pending": self.pending and self.pending.model_dump()
        }

class MemoryDAG(IGraph[CIDv1, float, PartialMemory, PartialMemory]):
    '''IPLD data model for memories implementing the IGraph interface.'''

    def __init__(self, keys: dict[CIDv1, PartialMemory]|None = None):
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
    def _node(self, value: PartialMemory) -> PartialMemory:
        copy = value.model_copy(deep=True)
        copy.edges = []
        return copy
    
    @override
    def _setvalue(self,  node: PartialMemory, value: PartialMemory):
        # We're assigning the discriminant with the data together so despite
        #  not being technically correct, there is no observable type violations
        node.kind = value.kind # type: ignore
        node.data = value.data # type: ignore
        node.timestamp = value.timestamp
        node.edges = value.edges
    
    @override
    def _valueof(self, node: PartialMemory) -> PartialMemory:
        return node
    
    @override
    def _edges(self, node: PartialMemory) -> Iterable[tuple[CIDv1, float]]:
        for edge in node.edges:
            yield edge.target, edge.weight

    @override
    def _add_edge(self, src: PartialMemory, dst: CIDv1, edge: float):
        if not any(dst == e.target for e in src.edges):
            src.edges.append(Edge[CIDv1](
                target=dst,
                weight=edge
            ))
    
    @override
    def _pop_edge(self, src: PartialMemory, dst: CIDv1) -> Optional[float]:
        edges = src.edges
        for i, edge in enumerate(edges):
            if edge.target == dst:
                del edges[i]
                return edge.weight
        return None