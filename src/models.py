from abc import abstractmethod
from datetime import datetime
from functools import cached_property
from typing import Annotated, Iterable, Literal, Optional, overload, override
from uuid import UUID
import json

from mcp.types import ModelPreferences
from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, TypeAdapter
from pydantic_core import CoreSchema, core_schema

from .graph import IGraph
from .ipld import dagcbor, CIDv1, CID

__all__ = (
    'MemoryKind',
    'StopReason',
    'RecallConfig', 'SampleConfig',
    'IPLDModel',
    'Edge', 'BaseMemory', 'NodeMemory',
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
    # See ModelSettings from pydantic for more to include
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
        Optional[ModelPreferences],
        Field(description="List of preferred models to use for the response. If `null`, uses the default model.")
    ] = None

class IPLDModel(BaseModel):
    '''Base model for IPLD objects.'''

    def as_block(self) -> bytes:
        '''Return the object as an IPLD block.'''
        return dagcbor.marshal(self.model_dump())

    @cached_property
    def cid(self):
        return CIDv1.hash(self.as_block())

class Edge[T](BaseModel):
    '''Edge from one memory to another.'''
    target: T
    weight: float

class Documenting(BaseModel):
    @abstractmethod
    def document(self) -> str:
        '''Return the document representation of the memory.'''

class SelfData(Documenting):
    '''A memory from the agent's own perspective.'''
    class Part(BaseModel):
        content: str = ""
        model: Optional[str] = None
    
    kind: Literal["self"] = "self"
    name: Optional[str] = None
    parts: list[Part]
    stop_reason: Optional[StopReason] = None

    @override
    def document(self):
        return "".join(part.content for part in self.parts)

class OtherData(Documenting):
    '''A memory produced by someone else (eg a user).'''
    kind: Literal["other"] = "other"
    name: Optional[str] = None
    content: str

    @override
    def document(self):
        return self.content

class TextData(Documenting):
    '''A text memory, which is just a string.'''
    kind: Literal["text"] = "text"
    content: str

    @override
    def document(self):
        return self.content

class FileData(Documenting):
    '''A file memory, which contains a file and metadata about it.'''
    kind: Literal["file"] = "file"
    file: CID
    filename: Optional[str] = None
    mimetype: str
    filesize: int

    @override
    def document(self): # ???
        return "" # self.content

class MetaData(Documenting):
    '''A memory which is just metadata.'''
    class Content(BaseModel):
        class Export(BaseModel):
            '''Metadata about exports from different providers.'''
            provider: Literal['anthropic', 'openai']
            convo_uuid: Optional[UUID] = None
            convo_title: Optional[str] = None
        
        export: Optional[Export] = None

    kind: Literal["metadata"] = "metadata"
    metadata: Optional[Content]
    
    @override
    def document(self):
        return json.dumps(self.metadata)

class ImportFileData(BaseModel):
    '''
    A memory which is an imported file. This will never appear in the
    IPLD model, but is used for importing files into the memory system.
    '''
    kind: Literal["import"] = "import"
    file: str = Field(
        description="File path or URL of the file to import."
    )
    filename: Optional[str] = Field(
        default=None,
        description="Filename to use for the imported file, if different from the original."
    )
    mimetype: Optional[str] = Field(
        default=None,
        description="MIME type of the file to import, if known."
    )

type AnyMemoryData = Annotated[
    SelfData | OtherData | TextData | FileData | MetaData | ImportFileData,
    Field(discriminator="kind")
]

type MemoryData = Annotated[
    SelfData | OtherData | TextData | FileData | MetaData,
    Field(discriminator="kind")
]

class BaseMemory[D: AnyMemoryData](BaseModel):
    '''Base memory model.'''

    data: D
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
        if isinstance(self.data, Documenting):
            return self.data.document()
        return ""

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
    def build_data(cls, kind: MemoryKind, json: str) -> AnyMemoryData:
        '''Build memory data based on the kind.'''
        match kind:
            case "self": return SelfData.model_validate_json(json)
            case "other": return OtherData.model_validate_json(json)
            case "text": return TextData.model_validate_json(json)
            case "image" | "file": return FileData.model_validate_json(json)
            case _:
                raise ValueError(f"Unknown memory kind: {kind}")

class NodeMemory[D: AnyMemoryData](BaseMemory[D]):
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

class DraftMemory[D: AnyMemoryData](NodeMemory[D]):
    '''A memory which cannot derive a CID.'''
    
    def insert_edge(self, target: CIDv1, weight: float):
        '''Insert an edge to the target memory with the given weight.'''
        if self.edge(target) is not None:
            raise ValueError(f"Edge to {target} already exists")
        
        self.edges.append(Edge(
            target=target,
            weight=weight
        ))

class IncompleteMemory[D: AnyMemoryData](DraftMemory[D]):
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
            edges=self.edges,
            importance=self.importance,
            sonas=self.sonas
        )

class PartialMemory[D: AnyMemoryData](DraftMemory[D]):
    '''
    A memory which is complete but does not have its full contents. Thus
    the CID can't be calculated from it and must be provided.
    '''
    cid: CIDv1

    def partial(self):
        return self

class Memory[D: AnyMemoryData](NodeMemory[D], IPLDModel):
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
            cid=self.cid,
            importance=self.importance,
            sonas=self.sonas
        )

type CompleteMemory = PartialMemory | Memory
'''A memory with a CID.'''

type AnyMemory = IncompleteMemory | PartialMemory | Memory

class ImportMemory[D: AnyMemoryData](DraftMemory[D]):
    '''Base memory model.'''
    type: Literal['memory'] = "memory"
    deps: Optional[list[DraftMemory]] = Field(
        default=None,
        description="Non-conversational dependencies of the memory, if any."
    )

class ImportConvo(BaseModel):
    '''Request to insert a sequence of memories.'''
    class Metadata(BaseModel):
        timestamp: Optional[datetime] = Field(
            default=None,
            description="Date and time when the conversation was created."
        )
        provider: Literal['anthropic', 'openai'] = Field(
            description="Provider of the conversation data."
        )
        uuid: UUID = Field(
            description="Unique identifier for the conversation."
        )
        title: str = Field(
            description="Title of the conversation."
        )
        importance: Optional[float] = Field(
            default=None,
            description="Importance of the conversation, used for recall weighting."
        )
    
    type: Literal['convo'] = "convo"

    sona: Optional[UUID|str] = Field(
        default=None,
        description="Sona to insert the memories into."
    )
    metadata: Optional[Metadata] = Field(
        description="Metadata about the conversation being inserted."
    )
    prev: Optional[CIDv1] = Field(
        default=None,
        description="CID of the previous memory in the thread, if any."
    )
    chatlog: list[ImportMemory[AnyMemoryData]] = Field(
        description="Chatlog to insert into the system."
    )

type ImportData = Annotated[
    ImportMemory | ImportConvo,
    Field(
        discriminator="type",
        description="Data to import into the system. Can be a memory or a conversation."
    )
]
type AnyImport = ImportData | list[ImportData]
ImportAdapter = TypeAdapter[AnyImport](AnyImport)

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

class UploadResponse(BaseModel):
    created: bool = Field(
        description="Whether the file was newly inserted."
    )
    size: int = Field(
        description="Size of the uploaded file including IPFS overhead in bytes."
    )
    cid: CID = Field(
        description="CID of the uploaded file."
    )

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