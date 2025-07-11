'''
Implements a Memoria MCP client.
'''

from datetime import timedelta
from uuid import UUID
from typing import Optional, cast, overload, override

from fastmcp.client.transports import ClientTransport
from fastmcp.exceptions import ToolError
from fastmcp.client.client import Client
from mcp.types import TextContent
from pydantic import BaseModel, TypeAdapter

from ipld.cid import CIDv1

from ._common import EdgeAnnotationResult, Emulator, QueryResult
from ..models import CompleteMemory, Edge, IncompleteMemory, MemoryDAG, NodeMemory, PartialMemory, RecallConfig, SampleConfig

class ClientEmulator[TransportT: ClientTransport](Emulator):
    '''Emulator with direct access to memoria, sampling left unimplemented.'''

    timeout: Optional[timedelta]

    def __init__(self, client: Client[TransportT]):
        super().__init__()
        self.client = client
        self.timeout = None#timedelta(seconds=5)

    async def progresss_handler(self,
            progress: float,
            total: Optional[float] = None,
            message: Optional[str] = None
        ) -> None:
        # Do nothing by default
        pass

    @overload
    async def _call_tool[T: BaseModel](self, model: type[T], tool: str, args: dict[str, object]) -> T: ...
    @overload
    async def _call_tool[T](self, model: TypeAdapter[T], tool: str, args: dict[str, object]) -> T: ...

    async def _call_tool(self, model, tool: str, args: dict[str, object]):
        '''Call a tool on the Memoria MCP client.'''
        res = await self.client.call_tool_mcp(
            tool, args,
            self.progresss_handler,
            self.timeout
        )
        if res.isError:
            raise ToolError(cast(TextContent, res.content[0]).text)
        
        if (structured := res.structuredContent) is None:
            content = res.content[0]
            if not isinstance(content, TextContent):
                raise ValueError(
                    f"Edge annotation response must be text, got {type(content)}: {content}"
                )
            
            if isinstance(model, TypeAdapter):
                return model.validate_json(content.text)
            return model.model_validate_json(content.text)
        else:
            if isinstance(model, TypeAdapter):
                return model.validate_python(structured)
            return model.model_validate_python(structured)

    @override
    async def act_push(
            self,
            sona: UUID|str,
            include: list[Edge[CIDv1]]
        ) -> Optional[UUID]:
        return await self._call_tool(
            TypeAdapter(UUID),
            "act_push", {
                'sona': sona,
                'include': include
            }
        )

    @override
    async def act_advance(self,
            sona: UUID|str,
            recall_config: RecallConfig = RecallConfig(),
            chat_config: SampleConfig = SampleConfig(),
            annotate_config: SampleConfig = SampleConfig()
        ) -> list[PartialMemory]:
        return await self._call_tool(
            TypeAdapter(list[PartialMemory]),
            "act_advance", {
                'sona': sona,
                'recall_config': recall_config,
                'chat_config': chat_config,
                'annotate_config': annotate_config
            }
        )

    @override
    async def recall(self,
            prompt: IncompleteMemory,
            recall_config: RecallConfig = RecallConfig()
        ) -> MemoryDAG:
        return await self._call_tool(TypeAdapter(MemoryDAG), "recall", {
            "prompt": prompt,
            "recall_config": recall_config
        })

    @override
    async def annotate(self,
            g: MemoryDAG,
            response: NodeMemory,
            annotate_config: SampleConfig
        ) -> EdgeAnnotationResult:
        return await self._call_tool(EdgeAnnotationResult, "annotate", {
            "g": g,
            "response": response,
            "annotate_config": annotate_config
        })
    
    @override
    async def insert(self,
            memory: NodeMemory,
            recall_config: RecallConfig = RecallConfig(),
            annotate_config: SampleConfig = SampleConfig()
        ) -> CompleteMemory:
        return await self._call_tool(TypeAdapter(CompleteMemory), "insert", {
            "memory": memory,
            "recall_config": recall_config,
            "annotate_config": annotate_config
        })
    
    @override
    async def query(self,
            prompt: NodeMemory,
            system_prompt: str,
            recall_config: RecallConfig = RecallConfig(),
            chat_config: SampleConfig = SampleConfig()
        ) -> QueryResult:
        return await self._call_tool(QueryResult, "query", {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "recall_config": recall_config,
            "chat_config": chat_config
        })
    
    @override
    async def chat(self,
            prompt: IncompleteMemory,
            system_prompt: str,
            recall_config: RecallConfig = RecallConfig(),
            chat_config: SampleConfig = SampleConfig(),
            annotate_config: SampleConfig = SampleConfig(),
        ) -> list[PartialMemory]:
        return await self._call_tool(
            TypeAdapter(list[PartialMemory]),
            "chat", {
                'prompt': prompt,
                'system_prompt': system_prompt,
                'recall_config': recall_config,
                'chat_config': chat_config,
                'annotate_config': annotate_config
            }
        )