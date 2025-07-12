from abc import abstractmethod
from datetime import datetime
from typing import Awaitable, Iterable, Optional, override
import json
from uuid import UUID

from mcp import CreateMessageResult, SamplingMessage
from mcp.types import Role, TextContent
from pydantic import BaseModel

from ._common import EdgeAnnotation, Emulator, EdgeAnnotationResult, QueryResult
from src.prompts import ANNOTATE_EDGES, CHAT_PROMPT
from src.ipld import CIDv1
from src.models import AnyMemory, CompleteMemory, DraftMemory, Edge, IncompleteMemory, MemoryDAG, MemoryData, NodeMemory, OtherData, PartialMemory, RecallConfig, SampleConfig, SelfData, TextData
from src.memoria import Repository

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
        case SelfData():
            if name := memory.data.name:
                tags.append(f"name:{name}")
            if sr := memory.data.stop_reason:
                if sr != "finish":
                    tags.append(f"stop_reason:{sr}")
            content = ''.join(p.content for p in memory.data.parts)
            return ("assistant", build_tags(tags, ts) + content)
        
        case TextData():
            tags.append("kind:raw_text")
            return ("user", build_tags(tags, ts) + memory.data.content)
        
        case OtherData():
            if name := memory.data.name:
                tags.append(f"name:{name}")
            return ("user", build_tags(tags, ts) + memory.data.content)
        
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

class AnnotateMessage(BaseModel):
    index: int
    deps: list[int]
    memory: PartialMemory[MemoryData]

class ServerEmulator(Emulator):
    '''
    Emulator with direct access to memoria, sampling left unimplemented. Used
    for implementing closed-loop emulators on the server for debug endpoints.
    '''

    def __init__(self, repo: Repository):
        super().__init__()
        self.repo = repo
    
    @abstractmethod
    def sample_chat(self,
        messages: Iterable[SamplingMessage],
        *,
        system_prompt: str,
        chat_config: SampleConfig
    ) -> Awaitable[CreateMessageResult]:
        '''Sample a response from the chat model based on the provided messages and system prompt.'''

    @abstractmethod
    def sample_annotate(self,
        messages: Iterable[AnnotateMessage],
        response: NodeMemory[MemoryData],
        *,
        system_prompt: str,
        annotate_config: SampleConfig
    ) -> Awaitable[EdgeAnnotation]:
        '''Sample a response from the LLM based on the provided messages and system prompt.'''

    @override
    async def recall(self,
            prompt: DraftMemory[MemoryData],
            recall_config: RecallConfig = RecallConfig()
        ) -> MemoryDAG:
        return self.repo.recall(prompt, recall_config)

    @override
    async def annotate(self,
            g: MemoryDAG,
            response: NodeMemory[MemoryData],
            annotate_config: SampleConfig
        ) -> EdgeAnnotationResult:
        '''
        Use sampling with a faster model to annotate which memories are
        connected and by how much.
        '''
        refs: dict[CIDv1, int] = {}
        cids: list[CIDv1] = []
        context: list[AnnotateMessage] = []

        # Load conversation
        for cid in g.invert().toposort(key=lambda v: v.timestamp):
            refs[cid] = index = len(context) + 1
            cids.append(cid)
            m = g[cid]
            context.append(AnnotateMessage(
                index=index,
                deps=[refs[e.target] for e in m.edges],
                memory=m
            ))
        
        # Annotate the edges
        result = await self.sample_annotate(
            context,
            response,
            system_prompt=ANNOTATE_EDGES,
            annotate_config=annotate_config
        )
        
        # Project from relative-ref space to absolute CIDs
        edges: dict[CIDv1, float] = {}
        prompt_rel: Optional[float] = None
        for k, v in result.relevance.items():
            if isinstance(v, int):
                # Here we're pedantic about the type because it's going
                # into IPLD which can distinguish between int and float
                v = float(v)
            elif not isinstance(v, float):
                raise ValueError(
                    f"Edge relevance score must be a number, got {type(v)}: {v}"
                )
            if k >= len(cids):
                prompt_rel = v
            else:
                edges[cids[k]] = v

        return EdgeAnnotationResult(
            relevance=edges,
            prompt=prompt_rel,
            imp=result.importance
        )
    
    def raw_insert(self, memory: IncompleteMemory[MemoryData]) -> CompleteMemory[MemoryData]:
        '''Just insert the memory into memoria, completing it.'''
        cmem = memory.complete()
        self.repo.insert(cmem)
        return cmem

    @override
    async def insert(self,
            memory: IncompleteMemory[MemoryData],
            recall_config: RecallConfig = RecallConfig(),
            annotate_config: SampleConfig = SampleConfig()
        ) -> CompleteMemory[MemoryData]:
        '''Insert a memory into the Memoria system.'''
        # If no timestamp is provided, use the current time
        if memory.timestamp is None:
            memory.timestamp = int(datetime.now().timestamp())
        
        # Recall relevant memories to ground the memory
        g = self.repo.recall(memory, recall_config)

        # Annotate the edges with relevance scores
        annotation = await self.annotate(g.invert(), memory, annotate_config)
        annotation.apply(memory)
        
        # Insert it
        return self.raw_insert(memory)
    
    @override
    async def query(self,
            prompt: IncompleteMemory[MemoryData],
            system_prompt: str,
            recall_config: RecallConfig = RecallConfig(),
            chat_config: SampleConfig = SampleConfig()
        ) -> QueryResult:
        '''Query the Memoria system for memories related to the query.'''
        g = await self.recall(prompt, recall_config)

        # Because this isn't committed to memory, we don't actually need
        # to annotate edges, and thus don't need localized references.

        refs: dict[CIDv1, int] = {}
        chatlog: list[PartialMemory[MemoryData]] = []
        messages: list[SamplingMessage] = []
        #for msg in serialize_dag(g, refs):
        for cid in g.invert().toposort(key=lambda v: v.timestamp):
            # We need ids for each memory so their edges can be annotated later
            ref = refs[cid] = len(refs) + 1
            memory = g[cid]
            chatlog.append(memory)
            role, content = memory_to_message(
                ref,
                [refs[dst] for dst, _ in g.edges(cid)],
                memory,
                final=(ref >= len(refs))
            )
            messages.append(
                SamplingMessage(
                    role=role,
                    content=TextContent(type="text", text=content)
                )
            )
        
        tag = f"ref:{len(messages)}"
        deps = [refs[e.target] for e in prompt.edges]
        if deps:
            tag += f"->{','.join(map(str, deps))}"

        # Inserting prompt before it's been annotated is fine because the
        # model can figure out the grounding.
        messages.append(sampling_message(
            "user", build_tags(
                ["final", tag], prompt.timestamp or int(datetime.now().timestamp())
            ) + prompt.document()
        ))

        response = await self.sample_chat(
            messages,
            system_prompt=system_prompt,
            chat_config=chat_config
        )

        return QueryResult(g=g, chatlog=chatlog, response=response)
    
    @override
    async def chat(self,
            prompt: IncompleteMemory[MemoryData],
            system_prompt: str,
            recall_config: RecallConfig = RecallConfig(),
            chat_config: SampleConfig = SampleConfig(),
            annotate_config: SampleConfig = SampleConfig(),
        ) -> list[PartialMemory[MemoryData]]:
        '''Single-turn chat with the Memoria system.'''
        
        # Query the chat model
        query = await self.query(
            prompt,
            system_prompt,
            recall_config,
            chat_config
        )
        note = await self.annotate(query.g, prompt, annotate_config)
        note.apply(prompt)
        memory = self.raw_insert(prompt).partial()
        return query.chatlog + [memory]

    @override
    async def act_push(
            self,
            sona: UUID|str,
            include: list[Edge[CIDv1]]
        ) -> Optional[UUID]:
        return self.repo.act_push(sona, include)
