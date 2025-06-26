'''
MCP Server for Memoria
'''

from datetime import datetime
from contextlib import asynccontextmanager
from typing import Annotated, Iterable, Optional, override
from uuid import UUID

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ResourceError, ToolError
from mcp import CreateMessageResult, SamplingMessage
from mcp.types import ModelPreferences, PromptMessage, TextContent
from pydantic import Field

import memoria

from ._common import mcp_context, lifespan
from src.ipld import CIDv1
from src.memoria import Memoria
from src.models import DraftMemory, Edge, IncompleteMemory, Memory, RecallConfig, SampleConfig, StopReason
from src.prompts import CHAT_PROMPT, QUERY_PROMPT
from src.emulator.server import ServerEmulator

DEFAULT_RECALL_CONFIG = RecallConfig()

DEFAULT_ANNOTATE_CONFIG = SampleConfig(
    temperature=0,
    max_tokens=1024,
    model_preferences=ModelPreferences(
        intelligencePriority=0.2,
        speedPriority=0.8,
        costPriority=0.1
    )
)

DEFAULT_CHAT_CONFIG = SampleConfig(
    temperature=1,
    max_tokens=16384,
    model_preferences=ModelPreferences(
        intelligencePriority=0.8,
        speedPriority=0.2,
        costPriority=0.5
    )
)

class MCPEmulator(ServerEmulator):
    def __init__(self, context: Context, memoria: Memoria):
        super().__init__(memoria)
        self.context = context

    @override
    async def sample(self,
            messages: Iterable[SamplingMessage],
            *,
            system_prompt: str,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            model_preferences: Optional[ModelPreferences | str | list[str]] = None
        ) -> CreateMessageResult:
        return await self.context.request_context.session.create_message(
            messages=list(messages),
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens or 16384,
            model_preferences=model_preferences
        )

@asynccontextmanager
async def mcp_lifespan(server: FastMCP):
    '''Lifespan context for the FastAPI app.'''
    with lifespan() as memoria:
        yield memoria

mcp = FastMCP("memoria",
    """Coordinates a "sona" representing a cohesive identity and memory.""",
    lifespan=mcp_lifespan,
    #log_level="DEBUG"
)

@mcp.resource("ipfs://{cid}")
def ipfs_resource(ctx: Context, cid: CIDv1):
    '''IPFS resource handler.'''
    memoria = mcp_context(ctx)
    if (m := memoria.lookup_memory(cid)) is None:
        raise ResourceError("Memory not found")
    
    if m.data.kind != "file":
        raise ResourceError("Memory is not a file")
    
    return m.data.content

@mcp.resource("memoria://sona/{uuid}")
def sona_resource(ctx: Context, uuid: UUID):
    '''Sona resource handler.'''
    memoria = mcp_context(ctx)
    if m := memoria.find_sona(uuid):
        return m
    raise ResourceError("Sona not found")

@mcp.resource("memoria://memory/{cid}")
def memory_resource(ctx: Context, cid: str):
    '''Memory resource handler.'''
    if m := mcp_context(ctx).lookup_memory(CIDv1(cid)):
        return m
    raise ResourceError("Memory not found")

@mcp.tool(
    annotations=dict(
        idempotentHint=True,
        openWorldHint=False
    )
)
async def insert(
        ctx: Context,
        memory: Annotated[
            IncompleteMemory,
            Field(description="Memory to insert.")
        ],
        recall_config: Annotated[
            RecallConfig,
            Field(description="Configuration for how to weight memory recall.")
        ] = DEFAULT_RECALL_CONFIG,
        annotate_config: Annotated[
            SampleConfig,
            Field(description="Configuration for how to sample the response for edge annotation.")
        ] = DEFAULT_ANNOTATE_CONFIG
    ):
    '''Insert a new memory into the sona.'''
    emu = MCPEmulator(ctx, mcp_context(ctx))
    return await emu.insert(memory, recall_config, annotate_config)

@mcp.tool(
    annotations=dict(
        readOnlyHint=True,
        openWorldHint=False,
    )
)
async def recall(
        ctx: Context,
        prompt: Annotated[
            DraftMemory,
            Field(description="Prompt to base the recall on.")
        ],
        recall_config: Annotated[
            RecallConfig,
            Field(description="Configuration for how to weight memory recall.")
        ] = DEFAULT_RECALL_CONFIG
    ):
    '''
    Recall memories related to the prompt, including relevant included memories
    and their dependencies.
    '''
    emu = MCPEmulator(ctx, mcp_context(ctx))
    g = await emu.recall(prompt, recall_config)
    return dict(g.items())

@mcp.tool(
    annotations=dict(
        readOnlyHint=True,
        openWorldHint=False,
    )
)
async def query(
        ctx: Context,
        prompt: Annotated[
            IncompleteMemory,
            Field(description="Prompt for the chat. If `null`, use only the included memories.")
        ],
        system_prompt: Annotated[
            Optional[str],
            Field(description="System prompt to use for the query. If `null`, uses the default query prompt.")
        ] = None,
        recall_config: Annotated[
            RecallConfig,
            Field(description="Configuration for how to weight memory recall.")
        ] = DEFAULT_RECALL_CONFIG,
        chat_config: Annotated[
            SampleConfig,
            Field(description="Configuration for how to sample the response.")
        ] = DEFAULT_CHAT_CONFIG
    ):
    '''Single-turn conversation returning the response.'''
    memoria = mcp_context(ctx)
    emu = MCPEmulator(ctx, memoria)

    qr = await emu.query(
        prompt,
        system_prompt or QUERY_PROMPT,
        recall_config,
        chat_config
    )
    response = qr.response
    assert isinstance(response, TextContent)

    return qr.chatlog + [IncompleteMemory(
        timestamp=int(datetime.now().timestamp()),
        data=Memory.SelfData(
            parts=[Memory.SelfData.Part(content=response.text)],
            stop_reason=qr.response.stopReason
        )
    )]

@mcp.tool(
    annotations=dict(
        openWorldHint=False,
    )
)
async def chat(
        ctx: Context,
        prompt: Annotated[
            IncompleteMemory,
            Field(description="Prompt for the chat. If `null`, use only the included memories.")
        ],
        system_prompt: Annotated[
            Optional[str],
            Field(description="System prompt to use for the chat. If `null`, uses the default chat prompt.")
        ] = None,
        recall_config: Annotated[
            RecallConfig,
            Field(description="Configuration for how to weight memory recall.")
        ] = DEFAULT_RECALL_CONFIG,
        chat_config: Annotated[
            SampleConfig,
            Field(description="Configuration for how to sample the response.")
        ] = DEFAULT_CHAT_CONFIG,
        annotate_config: Annotated[
            SampleConfig,
            Field(description="Configuration for how to sample the response for edge annotation.")
        ] = DEFAULT_ANNOTATE_CONFIG
    ):
    '''
    Single-turn conversation returning the response. This is committed to memory.
    '''
    memoria = mcp_context(ctx)
    emu = MCPEmulator(ctx, memoria)

    return await emu.chat(
        prompt,
        system_prompt or CHAT_PROMPT,
        recall_config,
        chat_config,
        annotate_config
    )

@mcp.tool(
    annotations=dict(
        openWorldHint=False
    )
)
def act_push(
        ctx: Context,
        sona: Annotated[
            UUID|str,
            Field(description="Sona to push the memory to.")
        ],
        memories: Annotated[
            list[Edge[CIDv1]],
            Field(description="Additional memories to include in the ACT, keyed by label.")
        ]
    ):
    '''
    Insert a new memory into the sona, formatted for an ACT
    (Autonomous Cognitive Thread).
    '''
    memoria = mcp_context(ctx)
    emu = MCPEmulator(ctx, memoria)
    if u := emu.act_push(sona, memories):
        return u
    raise ToolError("Sona not found or prompt memory not found.")

@mcp.tool(
    annotations=dict(
        openWorldHint=False
    )
)
def act_stream(
        ctx: Context,
        sona: Annotated[
            UUID|str,
            Field(description="Sona to push the memory to.")
        ],
        delta: Annotated[
            Optional[str],
            Field(description="Delta to append to the memory.")
        ],
        model: Annotated[
            Optional[str],
            Field(description="Model which generated this delta.")
        ] = None,
        stop_reason: Annotated[
            Optional[StopReason],
            Field(description="Reason for stopping the stream, if applicable.")
        ] = None,
    ):
    '''
    Stream tokens from the LLM to the sona to be committed to memory in case
    the LLM is interrupted or the session ends unexpectedly.
    '''
    memoria = mcp_context(ctx)
    return memoria.act_stream(sona, delta, model, stop_reason)

@mcp.prompt()
def act_next(
        ctx: Context,
        sona: Annotated[
            str,
            Field(description="Sona to process, either a name or UUID.")
        ],
        timestamp: Annotated[
            Optional[float],
            Field(description="Timestamp to use for the recall recency. If `null`, uses the current time.")
        ] = None,
        chat_config: Annotated[
            RecallConfig,
            Field(description="Configuration for how to weight memory recall.")
        ] = DEFAULT_RECALL_CONFIG
    ) -> Optional[list[PromptMessage]]:
    '''Get the prompt for the next step of an ACT (Autonomous Cognitive Thread).'''
    raise NotImplementedError()
    memoria = mcp_context(ctx)
    g = memoria.act_next(
        sona,
        timestamp or datetime.now().timestamp(),
        chat_config
    )
    if g is None:
        return None

    return [
        m.prompt_message(include_final=True)
            for m in [] #serialize_dag(g)
    ]