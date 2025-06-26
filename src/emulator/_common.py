'''
This is all the utilities for implementing an LLM-powered emulator of memoria
memories stores.
'''

from abc import ABC, abstractmethod
from typing import Awaitable, Optional

from mcp import CreateMessageResult
from pydantic import BaseModel, Field

from src.ipld import CIDv1
from src.models import CompleteMemory, DraftMemory, IncompleteMemory, MemoryDAG, NodeMemory, PartialMemory, RecallConfig, SampleConfig

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