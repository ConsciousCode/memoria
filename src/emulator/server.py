from abc import abstractmethod
from datetime import datetime
from typing import Awaitable, Iterable, Optional, override
import json
import re

from mcp import CreateMessageResult, SamplingMessage
from mcp.types import ModelPreferences, TextContent

from ._common import Emulator, EdgeAnnotation, QueryResult, sampling_message, build_tags, serialize_dag
from src.prompts import ANNOTATE_EDGES
from src.ipld import CIDv1
from src.models import CompleteMemory, DraftMemory, IncompleteMemory, Memory, MemoryDAG, NodeMemory, PartialMemory, RecallConfig, SampleConfig
from src.memoria import Memoria
from src.util import ifnone

class ServerEmulator(Emulator):
    '''Emulator with direct access to memoria, sampling left unimplemented.'''

    def __init__(self, memoria: Memoria):
        super().__init__()
        self.memoria = memoria

    @abstractmethod
    def sample(self,
        messages: Iterable[SamplingMessage],
        *,
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model_preferences: Optional[ModelPreferences | str | list[str]] = None
    ) -> Awaitable[CreateMessageResult]:
        '''Sample a response from the LLM based on the provided messages and system prompt.'''

    @override
    async def recall(self,
            prompt: DraftMemory,
            recall_config: RecallConfig = RecallConfig()
        ) -> MemoryDAG:
        return self.memoria.recall(prompt, recall_config)

    @override
    async def annotate(self,
            g: MemoryDAG,
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
        for cid in g.invert().toposort(key=lambda v: v.timestamp):
            refs[cid] = index = len(context) + 1
            cids.append(cid)
            m = g[cid]
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

        if not isinstance(content := result.content, TextContent):
            raise ValueError(
                f"Edge annotation response must be text, got {type(content)}: {content}"
            )
        
        try:
            # Parse the annotation (LLMs are really bad at this and MCP
            # doesn't support structured outputs yet)
            assert (m := re.match(r"^(?:```(?:json)?)?([\s\S\n]+?)(?:```)?$", content.text))
            data = json.loads(m[1])
            
            if not isinstance(data, dict):
                raise ValueError(f"Edge annotation response must be a JSON object, got {type(result)}: {result}")
            
            # Extract the data we want from the LLM response
            importance = data.get("importance", {})
            edges: dict[CIDv1, float] = {}
            prompt_rel: Optional[float] = None
            for k, v in data.get("relevance", {}).items():
                if v < 0: continue
                if m := re.match(r"\d+", k):
                    ki = int(m[0])
                    if ki == 0: continue
                    ki -= 1 # Convert to 0-based index
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
    
    def raw_insert(self, memory: IncompleteMemory) -> CompleteMemory:
        '''Just insert the memory into memoria, completing it.'''
        cmem = memory.complete()
        self.memoria.insert(cmem)
        return cmem

    @override
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
    
    @override
    async def query(self,
            prompt: IncompleteMemory,
            system_prompt: str,
            recall_config: RecallConfig = RecallConfig(),
            chat_config: SampleConfig = SampleConfig()
        ) -> QueryResult:
        '''Query the Memoria system for memories related to the query.'''
        g = await self.recall(prompt, recall_config)

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
            system_prompt=system_prompt,
            temperature=ifnone(chat_config.temperature, 0.7),
            max_tokens=chat_config.max_tokens,
            model_preferences=chat_config.model_preferences
        )

        return QueryResult(g=g, chatlog=chatlog, response=response)
    
    @override
    async def chat(self,
            prompt: IncompleteMemory,
            system_prompt: str,
            recall_config: RecallConfig = RecallConfig(),
            chat_config: SampleConfig = SampleConfig(),
            annotate_config: SampleConfig = SampleConfig(),
        ) -> list[PartialMemory]:
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

        return query.chatlog + [self_memory.partial()]
    