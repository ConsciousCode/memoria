'''
This is all the utilities for implementing an LLM-powered emulator of memoria
memories stores.
'''

from abc import ABC, abstractmethod
from datetime import datetime
import json
import re
from typing import Coroutine, Iterable, Optional

from mcp import CreateMessageResult, SamplingMessage
from mcp.types import ModelPreferences, PromptMessage, Role, TextContent
from pydantic import BaseModel, Field

from src.ipld import CIDv1
from src.memoria import Memoria
from src.models import AnyMemory, Chatlog, CompleteMemory, DraftMemory, IncompleteMemory, Memory, MemoryDAG, NodeMemory, PartialMemory, RecallConfig, SampleConfig
from src.prompts import ANNOTATE_EDGES
from src.util import ifnone

def parse_edge_ref(k: str):
    # Some models are stupid cunts which can't follow instructions.
    if m := re.search(r"\d+", k):
        return int(m[0])

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

    def __init__(self, memoria: Memoria):
        super().__init__()
        self.memoria = memoria

    @abstractmethod
    def sample(self,
        messages: Iterable[SamplingMessage],
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model_preferences: Optional[ModelPreferences | str | list[str]] = None
    ) -> Coroutine[None, None, CreateMessageResult]:
        '''Sample a response from the LLM based on the provided messages and system prompt.'''
        pass

    async def annotate(self,
            g_inv: MemoryDAG,
            response: NodeMemory,
            annotate_config: SampleConfig
        ) -> EdgeAnnotation:
        '''
        Use sampling with a faster model to annotate which memories are
        connected and by how much.
        '''
        refs: dict[CIDv1, int] = {}
        cids: list[CIDv1] = []
        context: list[dict] = []

        # Load conversation
        for cid in g_inv.toposort(key=lambda v: v.timestamp):
            refs[cid] = index = len(context) + 1
            cids.append(cid)
            m = g_inv[cid]
            c = {
                "index": index,
                "role": "assistant" if m.data.kind == "self" else "user",
                "content": m.data.document()
            }
            if m.edges:
                c['deps'] = [refs[e.target] for e in m.edges]
            context.append(c)
        
        # Add the response to the conversation
        context.append({
            "deps": None,
            "role": "assistant",
            "content": response.data.document()
        })
        
        # Annotate the edges
        result = await self.sample(
            [SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=json.dumps(context, indent=2)
                )
            )],
            system_prompt=ANNOTATE_EDGES,
            temperature=0,
            max_tokens=None,
            model_preferences=annotate_config.model_preferences
        )
        if not isinstance(result, TextContent):
            raise ValueError(
                f"Edge annotation response must be text, got {type(result)}: {result}"
            )
        
        try:
            # Parse the annotation (LLMs are really bad at this and MCP
            # doesn't support structured outputs yet)
            assert (m := re.match(r"^(?:```(?:json)?)?([\s\S\n]+?)(?:```)?$", result.text))
            data = json.loads(m[1])
            
            if not isinstance(data, dict):
                raise ValueError(f"Edge annotation response must be a JSON object, got {type(result)}: {result}")
            
            # Extract the data we want from the LLM response
            importance = data.get("importance", {})
            edges: dict[CIDv1, float] = {}
            prompt_rel: Optional[float] = None
            for k, v in data.get("relevance", {}).items():
                if v < 0: continue
                if ki := parse_edge_ref(k):
                    if not isinstance(v, (int, float)):
                        raise ValueError(
                            f"Edge relevance score must be a number, got {type(v)}: {v}"
                        )
                    if ki >= len(cids):
                        prompt_rel = v
                    else:
                        edges[cids[ki]] = v

            return EdgeAnnotation(
                rel=edges,
                prompt=prompt_rel,
                imp=importance
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Edge annotation response is not valid JSON: {e}") from e
    
    def raw_recall(self,
            memory: DraftMemory,
            recall_config: RecallConfig
        ) -> MemoryDAG:
        '''Recall supporting memories for the provided memory.'''
        return self.memoria.recall(memory, recall_config)

    def raw_insert(self, memory: IncompleteMemory) -> CompleteMemory:
        '''Just insert the memory into memoria, completing it.'''
        cmem = memory.complete()
        self.memoria.insert(cmem)
        return cmem

    async def insert(self,
            memory: IncompleteMemory,
            recall_config: RecallConfig = RecallConfig(),
            annotate_config: SampleConfig = SampleConfig()
        ) -> CompleteMemory:
        '''Insert a memory into the Memoria system.'''
        # If no timestamp is provided, use the current time
        if memory.timestamp is None:
            memory.timestamp = int(datetime.now().timestamp())
        
        # Recall relevant memories to ground the memory
        g = self.memoria.recall(memory, recall_config)

        # Annotate the edges with relevance scores
        annotation = await self.annotate(g.invert(), memory, annotate_config)
        annotation.apply(memory)
        
        # Insert it
        return self.raw_insert(memory)
    
    async def query(self,
            prompt: IncompleteMemory,
            system_prompt: str,
            recall_config: RecallConfig = RecallConfig(),
            chat_config: SampleConfig = SampleConfig()
        ) -> QueryResult:
        '''Query the Memoria system for memories related to the query.'''
        g = self.raw_recall(prompt, recall_config)

        # Because this isn't committed to memory, we don't actually need
        # to annotate edges, and thus don't need localized references.

        refs: dict[CIDv1, int] = {}
        chatlog: list[PartialMemory] = []
        messages: list[SamplingMessage] = []
        for msg in serialize_dag(g, refs):
            chatlog.append(msg.memory)
            messages.append(msg.sampling_message(include_final=True))
        
        tag = f"ref:{len(messages)}"
        deps = [refs[e.target] for e in prompt.edges]
        if deps:
            tag += f"->{','.join(map(str, deps))}"

        # Inserting prompt before it's been annotated is fine because the
        # model can figure out the grounding.
        messages.append(sampling_message(
            "user", build_tags(
                ["final", tag], prompt.timestamp
            ) + prompt.document()
        ))

        response = await self.sample(
            messages,
            system_prompt,
            ifnone(chat_config.temperature, 0.7),
            chat_config.max_tokens,
            chat_config.model_preferences
        )

        return QueryResult(g=g, chatlog=chatlog, response=response)
    
    async def chat(self,
            prompt: IncompleteMemory,
            system_prompt: str,
            recall_config: RecallConfig = RecallConfig(),
            chat_config: SampleConfig = SampleConfig(),
            annotate_config: SampleConfig = SampleConfig(),
        ) -> Chatlog:
        '''Single-turn chat with the Memoria system.'''
        
        # Query the chat model
        query = await self.query(
            prompt,
            system_prompt,
            recall_config,
            chat_config
        )
        assert isinstance(content := query.response.content, TextContent)
        g = query.g

        other_note = await self.annotate(
            g.invert(), prompt, annotate_config
        )
        other_note.apply(prompt)
        other_memory = self.raw_insert(prompt).partial()

        g.insert(other_memory.cid, other_memory)

        response = IncompleteMemory(
            data=Memory.SelfData(
                parts=[Memory.SelfData.Part(
                    content=content.text
                )],
                stop_reason=query.response.stopReason
            ),
            timestamp=int(datetime.now().timestamp())
        )

        # Annotate the edges with relevance scores
        self_note = await self.annotate(g.invert(), response, annotate_config)
        self_note.apply(response)
        self_memory = self.raw_insert(response)

        return Chatlog(
            chatlog=query.chatlog,
            response=self_memory
        )