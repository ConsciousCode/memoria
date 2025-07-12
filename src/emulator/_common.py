'''
This is all the utilities for implementing an LLM-powered emulator of memoria
memories stores.
'''

from abc import ABC, abstractmethod
from typing import Annotated, Awaitable, Optional
from uuid import UUID
import re

from mcp import CreateMessageResult
from pydantic import BaseModel, Field, field_validator

from src.ipld import CIDv1
from src.models import CompleteMemory, DraftMemory, Edge, IncompleteMemory, MemoryDAG, MemoryData, NodeMemory, PartialMemory, RecallConfig, SampleConfig

__all__ = (
    'EdgeAnnotation',
    'QueryResult',
    'Emulator'
)

LocalRef = Annotated[int, Field(ge=0)]
LLMScore = Annotated[int, Field(ge=0, le=10)]

class ImportanceAnnotation(BaseModel):
    novelty: LLMScore = Field(
        description="Novelty score for the response [0-10]."
    )
    intensity: LLMScore = Field(
        description="Intensity score for the response [0-10]."
    )
    future: LLMScore = Field(
        description="Future score for the response [0-10]."
    )
    personal: LLMScore = Field(
        description="Personal score for the response [0-10]."
    )
    saliency: LLMScore = Field(
        description="Saliency score for the response [0-10]."
    )

class EdgeAnnotation(BaseModel):
    '''Model for LLM edge annotation.'''
    relevance: dict[LocalRef, LLMScore] = Field(
        description="Relevance scores for each memory [0-10]."
    )
    importance: ImportanceAnnotation = Field(
        description="Importance scores for the response [0-10]."
    )

    @field_validator('relevance')
    @classmethod
    def validate_relevance(cls, v: dict[str, int]) -> dict[LocalRef, LLMScore]:
        '''Ensure relevance scores are within range.'''
        out: dict[int, LLMScore] = {}
        for k, score in v.items():
            # LLMs are unbelievably bad at following instructions so we need to
            # account for random garbage like "[ref:12]" or "#12"
            if m := re.match(r"\d+", k):
                k = int(m[0])
            else:
                raise ValueError(f"Memory key {k} must be an integer.")
            
            if not (0 <= score <= 10):
                raise ValueError(f"Relevance score for memory {k} must be between 0 and 10.")
            out[k] = score
        return out

class EdgeAnnotationResult(BaseModel):
    '''Edge annotation result.'''

    relevance: dict[CIDv1, float] = Field(
        description="Relevance scores for each memory."
    )
    prompt: Optional[float] = Field(
        description="Relevance score for the prompt memory."
    )
    imp: ImportanceAnnotation = Field(
        description="Importance scores for the response."
    )

    @property
    def importance(self):
        '''Total importance score for the response.'''
        if imp := self.imp.model_dump():
            return sum(imp.values()) / len(imp)
        return 0
    
    def apply(self, memory: DraftMemory[MemoryData]):
        '''Apply the edge annotations to the memory, modifying in-place.'''
        if not memory.edges:
            memory.edges = []
        
        for ref, weight in self.relevance.items():
            if weight < 0: continue
            if not memory.has_edge(ref):
                memory.insert_edge(ref, weight / 10)
        
        # Add importance scores as metadata
        if self.imp:
            memory.importance = self.importance

class QueryResult(BaseModel):
    '''Result of a query to the Memoria system.'''
    
    g: MemoryDAG = Field(
        description="Memory DAG of the query."
    )
    chatlog: list[PartialMemory[MemoryData]] = Field(
        description="Chatlog of the query."
    )
    response: CreateMessageResult = Field(
        description="Response from the LLM to the query."
    )

class Emulator(ABC):
    '''Base class for emulating an artificial person.'''

    @abstractmethod
    def recall(self,
            prompt: IncompleteMemory[MemoryData],
            recall_config: RecallConfig = RecallConfig()
        ) -> Awaitable[MemoryDAG]:
        '''Recall supporting memories for the provided memory.'''

    @abstractmethod
    def annotate(self,
            g: MemoryDAG,
            response: NodeMemory[MemoryData],
            annotate_config: SampleConfig
        ) -> Awaitable[EdgeAnnotationResult]:
        '''
        Use sampling with a faster model to annotate which memories are
        connected and by how much.
        '''

    @abstractmethod
    def insert(self,
            memory: IncompleteMemory[MemoryData],
            recall_config: RecallConfig = RecallConfig(),
            annotate_config: SampleConfig = SampleConfig()
        ) -> Awaitable[CompleteMemory[MemoryData]]:
        '''Insert a memory into the Memoria system.'''
    
    @abstractmethod
    def query(self,
            prompt: IncompleteMemory[MemoryData],
            system_prompt: str,
            recall_config: RecallConfig = RecallConfig(),
            chat_config: SampleConfig = SampleConfig()
        ) -> Awaitable[QueryResult]:
        '''Query the Memoria system for memories related to the query.'''
    
    @abstractmethod
    def chat(self,
            prompt: IncompleteMemory[MemoryData],
            system_prompt: str,
            recall_config: RecallConfig = RecallConfig(),
            chat_config: SampleConfig = SampleConfig(),
            annotate_config: SampleConfig = SampleConfig(),
        ) -> Awaitable[list[PartialMemory[MemoryData]]]:
        '''Single-turn chat with the Memoria system.'''
    
    @abstractmethod
    async def act_push(
            self,
            sona: UUID|str,
            include: list[Edge[CIDv1]]
        ) -> Optional[UUID]:
        '''Push a prompt to the sona for processing by its ACT.'''