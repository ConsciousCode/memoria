from functools import cached_property
from typing import Annotated, Iterable, Literal, Optional, overload, override
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, StrictInt, aliases

from graph import IGraph
from ipld import dagcbor
from ipld.cid import CIDv1, cidhash

type MemoryKind = Literal["self", "other", "text", "image", "file"]
type UUIDCID = Annotated[UUID,
    PlainSerializer(
        lambda u: cidhash(u.bytes, codec='raw'),
        return_type=CIDv1
    )
]
type StopReason = Literal["endTurn", "stopSequence", "maxTokens"] | str

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
    decay: Annotated[
        Optional[float],
        Field(description="Time decay factor for recency. 0.995 by default.")
    ]=None

class IPLDModel(BaseModel):
    '''Base model for IPLD objects.'''
    @cached_property
    def cid(self):
        print(type(self), dagcbor._dagcbor_encode(self.model_dump()))
        return cidhash(dagcbor.marshal(self.model_dump()))

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
    timestamp: Optional[StrictInt] = None
    edges: list[Edge[CIDv1]] = Field(
        default_factory=list,
        description="Edges to other memories."
    )
    importance: Optional[float] = Field(
        default=None,
        exclude=True,
        description="Importance of the memory, used for recall weighting."
    )
    sonas: Optional[list[UUID]] = Field(
        default=None,
        exclude=True,
        description="Sonas the memory belongs to."
    )

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

    def edge(self, target: CIDv1) -> Optional[Edge[CIDv1]]:
        '''Get the edge to the target memory, if it exists.'''
        
        for edge in self.edges:
            if edge.target == target:
                return edge
        return None

class IncompleteMemory(BaseMemory):
    '''
    A memory which is still incomplete and thus can't be referred to by
    CID. Allows mutation of edges and data.
    '''

    cid: None = None

    def complete(self) -> 'Memory':
        '''Complete the memory by adding edges and returning a Memory object.'''
        return Memory(
            data=self.data,
            timestamp=self.timestamp,
            edges=self.edges
        )
    
    def insert_edge(self, target: CIDv1, weight: float):
        '''Insert an edge to the target memory with the given weight.'''
        if self.edge(target) is not None:
            raise ValueError(f"Edge to {target} already exists")
        
        self.edges.append(Edge(
            target=target,
            weight=weight
        ))

class PartialMemory(BaseMemory):
    '''
    A memory which is complete, but does not have its full contents. Thus
    the CID can't be calculated from it and must be provided.
    '''
    
    cid: CIDv1

    def insert_edge(self, target: CIDv1, weight: float):
        '''Insert an edge to the target memory with the given weight.'''
        if self.edge(target) is not None:
            raise ValueError(f"Edge to {target} already exists")
        
        self.edges.append(Edge(
            target=target,
            weight=weight
        ))

class Memory(BaseMemory, IPLDModel):
    '''A completed memory which can be referred to by CID.'''

    model_config = ConfigDict(frozen=True)

    @cached_property
    @override
    def cid(self) -> CIDv1:
        # Edges must be sorted by target CID to ensure deterministic ordering
        self.edges.sort(key=lambda e: e.target)
        return super().cid
    
    def incomplete(self):
        '''Return an incomplete memory which can be mutated.'''
        return IncompleteMemory(
            data=self.data,
            timestamp=self.timestamp,
            edges=self.edges
        )

type MaybeMemory = IncompleteMemory | PartialMemory
'''A memory which may or may not have a CID.'''
type CompleteMemory = PartialMemory | Memory
'''A memory with a CID.'''
type AnyMemory = IncompleteMemory | PartialMemory | Memory

class Chatlog(BaseModel):
    '''Data model for a single-turn chat as returned by the server.'''
    chatlog: list[PartialMemory]
    response: Memory

class IncompleteACThread(BaseModel):
    '''A thread of memories in the agent's context.'''
    cid: None = None # Incomplete threads can't have a cid
    sona: UUIDCID
    memory: IncompleteMemory # Can't be referred to by cid
    prev: Optional[CIDv1] = None

class ACThread(IPLDModel):
    '''A thread of memories in the agent's context.'''
    sona: UUIDCID
    memory: CIDv1
    prev: Optional[CIDv1] = None

type AnyACThread = IncompleteACThread | ACThread

class Sona(BaseModel):
    '''
    Sona model for returning from memory queries. It is not immutable and
    is thus referred to by UUID rather than CID.
    '''
    uuid: UUIDCID = Field(
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
        # it will trigger UUIDCID's serialization as a CID and lose the
        # human-readable UUID.
        return {
            "uuid": str(self.uuid),
            "aliases": self.aliases,
            "active": self.active.model_dump() if self.active else None,
            "pending": self.pending.model_dump() if self.pending else None
        }

class MemoryDAG(IGraph[CIDv1, float, PartialMemory, PartialMemory]):
    '''IPLD data model for memories implementing the IGraph interface.'''

    def __init__(self, keys: dict[CIDv1, PartialMemory]|None = None):
        super().__init__()
        self.adj = keys or {}
    
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