'''
This is all the utilities for implementing an LLM-powered emulator of memoria
memories stores.
'''

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Awaitable, Iterable, Optional

from mcp import CreateMessageResult, SamplingMessage
from mcp.types import PromptMessage, Role, TextContent
from pydantic import BaseModel, Field

from src.ipld import CIDv1
from src.models import AnyMemory, CompleteMemory, DraftMemory, IncompleteMemory, Memory, MemoryDAG, NodeMemory, PartialMemory, RecallConfig, SampleConfig

__all__ = (
    'EdgeAnnotation',
    'QueryResult',
    'Emulator'
)

class EdgeAnnotation(BaseModel):
    '''Edge annotation result.'''

    rel: dict[CIDv1, float] = Field(
        description="Relevance scores for each memory."
    )
    prompt: Optional[float] = Field(
        description="Relevance score for the prompt memory."
    )
    imp: dict[str, float] = Field(
        description="Importance scores for the response."
    )

    @property
    def importance(self):
        '''Total importance score for the response.'''
        if imp := self.imp:
            return sum(imp.values()) / len(imp)
        return 0
    
    def apply(self, memory: DraftMemory):
        '''Apply the edge annotations to the memory, modifying in-place.'''
        if not memory.edges:
            memory.edges = []
        
        for ref, weight in self.rel.items():
            if weight < 0: continue
            memory.insert_edge(ref, weight / 10)
        
        # Add importance scores as metadata
        if self.imp:
            memory.importance = self.importance

class QueryResult(BaseModel):
    '''Result of a query to the Memoria system.'''
    
    g: MemoryDAG = Field(
        description="Memory DAG of the query."
    )
    chatlog: list[PartialMemory] = Field(
        description="Chatlog of the query."
    )
    response: CreateMessageResult = Field(
        description="Response from the LLM to the query."
    )

class Emulator(ABC):
    '''Base class for emulating an artificial person.'''

    @abstractmethod
    def recall(self,
            prompt: IncompleteMemory,
            recall_config: RecallConfig = RecallConfig()
        ) -> Awaitable[MemoryDAG]:
        '''Recall supporting memories for the provided memory.'''

    @abstractmethod
    def annotate(self,
            g: MemoryDAG,
            response: NodeMemory,
            annotate_config: SampleConfig
        ) -> Awaitable[EdgeAnnotation]:
        '''
        Use sampling with a faster model to annotate which memories are
        connected and by how much.
        '''

    @abstractmethod
    def insert(self,
            memory: IncompleteMemory,
            recall_config: RecallConfig = RecallConfig(),
            annotate_config: SampleConfig = SampleConfig()
        ) -> Awaitable[CompleteMemory]:
        '''Insert a memory into the Memoria system.'''
    
    @abstractmethod
    def query(self,
            prompt: IncompleteMemory,
            system_prompt: str,
            recall_config: RecallConfig = RecallConfig(),
            chat_config: SampleConfig = SampleConfig()
        ) -> Awaitable[QueryResult]:
        '''Query the Memoria system for memories related to the query.'''
    
    @abstractmethod
    def chat(self,
            prompt: IncompleteMemory,
            system_prompt: str,
            recall_config: RecallConfig = RecallConfig(),
            chat_config: SampleConfig = SampleConfig(),
            annotate_config: SampleConfig = SampleConfig(),
        ) -> Awaitable[list[PartialMemory]]:
        '''Single-turn chat with the Memoria system.'''


def build_tags(tags: list[str], timestamp: Optional[float|datetime]) -> str:
    if timestamp is not None:
        if not isinstance(timestamp, datetime):
            timestamp = datetime.fromtimestamp(timestamp)
        tags.append(timestamp.replace(microsecond=0).isoformat())
    return f"[{'\t'.join(tags)}]\t"

def memory_to_message(ref: int, deps: list[int], memory: AnyMemory, final: bool=False) -> tuple[Role, str]:
    '''Render memory for the context.'''
    
    tags = []
    if final: tags.append("final")
    tags.append(
        f"ref:{ref}" + (f"->{','.join(map(str, deps))}" if deps else "")
    )

    if (ts := memory.timestamp) is not None:
        ts = datetime.fromtimestamp(ts)
    
    match memory.data:
        case Memory.SelfData():
            if name := memory.data.name:
                tags.append(f"name:{name}")
            if sr := memory.data.stop_reason:
                if sr != "finish":
                    tags.append(f"stop_reason:{sr}")
            return (
                "assistant",
                build_tags(tags, ts) +
                    ''.join(p.content for p in memory.data.parts)
            )
        
        case Memory.TextData():
            tags.append("kind:raw_text")
            return (
                "user",
                build_tags(tags, ts) + memory.data.content
            )
        
        case Memory.OtherData():
            if name := memory.data.name:
                tags.append(f"name:{name}")
            return (
                "user",
                build_tags(tags, ts) + memory.data.content
            )
        
            '''
        case "file":
            return ConvoMessage(
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri=AnyUrl(f"memory://{memory.cid}"),
                        mimeType=memory.data.mimeType or
                            "application/octet-stream",
                        blob=memory.data.content
                    )
                ),
                role="user"
            )
        '''
        
        case _:
            raise ValueError(f"Unknown memory kind: {memory.data.kind}")

def sampling_message(role: Role, content: str) -> SamplingMessage:
    '''Create a SamplingMessage from role and content.'''
    return SamplingMessage(
        role=role,
        content=TextContent(
            type="text",
            text=content
        )
    )

class ConvoMessage(BaseModel):
    '''
    I hate everything about this and would like nothing more than some
    other approach to serializing the conversation that doen't involve
    *REIFYING THE ENTIRE FUCKING ITERATOR STATE*
    '''
    g: MemoryDAG
    ref: int
    refs: dict[CIDv1, int]
    memory: PartialMemory
    
    def message(self, include_final: bool = False):
        cid = self.memory.cid
        return memory_to_message(
            self.ref,
            [self.refs[dst] for dst, _ in self.g.edges(cid)],
            self.memory,
            final=include_final and (self.ref >= len(self.refs))
        )

    def prompt_message(self, include_final: bool = False) -> PromptMessage:
        role, content = self.message(include_final)
        return PromptMessage(
            role=role,
            content=TextContent(
                type="text",
                text=content
            )
        )
    
    def sampling_message(self, include_final: bool = False) -> SamplingMessage:
        role, content = self.message(include_final)
        return SamplingMessage(
            role=role,
            content=TextContent(
                type="text",
                text=content
            )
        )

def serialize_dag(g: MemoryDAG, refs: Optional[dict[CIDv1, int]]=None) -> Iterable[ConvoMessage]:
    '''Convert a memory DAG to a conversation.'''
    if refs is None:
        refs = {}
    for cid in g.invert().toposort(key=lambda v: v.timestamp):
        # We need ids for each memory so their edges can be annotated later
        ref = refs[cid] = len(refs) + 1
        #yield ref, g[cid]
        yield ConvoMessage(
            g=g,
            ref=ref,
            refs=refs,
            memory=g[cid]
        )

