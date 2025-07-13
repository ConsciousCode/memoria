'''
MCP Server for Memoria
'''

from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from typing import Annotated, Iterable, Optional, override
from uuid import UUID
import base64

from fastmcp import Context
from fastmcp.exceptions import ResourceError, ToolError
from mcp import SamplingMessage
from mcp.types import ModelPreferences, PromptMessage, TextContent
from pydantic import Field
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.mcp_sampling import MCPSamplingModel, MCPSamplingModelSettings

from ..emulator._common import EdgeAnnotation

from ._common import AddParameters, AppState, get_appstate, get_repo, mcp
from ..ipld import CIDv1, CIDResolveError
from ..memory import AnyMemory, DraftMemory, Edge, Memory, RecallConfig, SampleConfig, SelfData, StopReason, UploadResponse
from ..prompts import CHAT_PROMPT, QUERY_PROMPT
from ..emulator.server import AnnotateMessage, ServerEmulator

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

def sample_to_model(sample: SampleConfig) -> MCPSamplingModelSettings:
    """Convert SampleConfig to ModelSettings."""
    settings: MCPSamplingModelSettings = {}
    if sample.temperature is not None:
        settings["temperature"] = sample.temperature
    if sample.max_tokens is not None:
        settings["max_tokens"] = sample.max_tokens
    if sample.model_preferences:
        settings["mcp_model_preferences"] = sample.model_preferences
    return settings

class MCPEmulator(ServerEmulator):
    def __init__(self, context: Context, state: AppState):
        super().__init__(state.repo)
        self.context = context
        self.state = state
    
    def convert_history(self, msgs: Iterable[SamplingMessage]) -> Iterable[ModelMessage]:
        """Convert SamplingMessages to ModelRequest/ModelResponse history."""
        for m in msgs:
            assert isinstance(m.content, TextContent), "Only TextContent is supported"
            text = m.content.text
            if m.role == "user":
                yield ModelRequest.user_text_prompt(text)
            elif m.role == "assistant":
                yield ModelResponse(parts=[TextPart(content=text)])
            else:
                raise ValueError(f"Unknown role {m.role!r}")

    @override
    async def sample_chat(self,
            messages: Iterable[SamplingMessage],
            *,
            system_prompt: str,
            chat_config: SampleConfig
        ) -> DraftMemory:
        agent = Agent(
            model=MCPSamplingModel(
                session=self.context.request_context.session
            ),
            instructions=system_prompt,
            name="chat"
        )
        result = await agent.run(
            message_history=list(self.convert_history(messages)),
            model_settings=sample_to_model(chat_config)
        )
        return DraftMemory(
            data=SelfData(
                parts=[SelfData.Part(content=result.output)],
                stop_reason=None
            ),
            timestamp=int(datetime.now().timestamp())
        )
    
    @override
    async def sample_annotate(self,
            messages: Iterable[AnnotateMessage],
            response: AnyMemory,
            *,
            system_prompt: str,
            annotate_config: SampleConfig
        ) -> EdgeAnnotation:
        agent = Agent(
            model=MCPSamplingModel(
                session=self.context.request_context.session
            ),
            output_type=EdgeAnnotation,
            instructions=system_prompt,
            name="annotate"
        )
        print(list(messages), response)
        raise NotImplementedError("Edge annotation is not yet implemented in MCPEmulator")
        with capture_run_messages() as run_messages:
            try:
                result = await agent.run(
                    message_history=list(self.convert_history(messages)),
                    model_settings=sample_to_model(annotate_config)
                )
                return result.output
            except:
                print(run_messages)
                raise

@contextmanager
def mcp_emu(ctx: Context):
    '''Get the application state from the context.'''
    with get_appstate() as state:
        yield MCPEmulator(ctx, state)

@mcp.resource("ipfs://{cid}")
def ipfs_resource(ctx: Context, cid: CIDv1):
    '''IPFS resource handler.'''
    with get_appstate() as state:
        try: return state.dag_get(cid)
        except CIDResolveError:
            raise ResourceError(f"CID {cid} not found in IPFS blockstore")

@mcp.resource("memoria://sona/{uuid}")
def sona_resource(ctx: Context, uuid: UUID):
    '''Sona resource handler.'''
    with get_repo() as repo:
        if m := repo.find_sona(uuid):
            return m
        raise ResourceError("Sona not found")

@mcp.resource("memoria://memory/{cid}")
def memory_resource(ctx: Context, cid: CIDv1):
    '''Memory resource handler.'''
    with get_repo() as repo:
        if m := repo.lookup_memory(cid):
            return m
        raise ResourceError("Memory not found")

@mcp.tool(
    annotations=dict(
        idempotentHint=True,
        openWorldHint=False
    )
)
async def upload(
        ctx: Context,
        file: Annotated[
            str,
            Field(description="File to upload encoded as a base64 string.")
        ],
        filename: Annotated[
            Optional[str],
            Field(description="Filename to use for the uploaded file.")
        ] = None,
        mimetype: Annotated[
            Optional[str],
            Field(description="MIME type of the file.")
        ] = None,
        params: Annotated[
            AddParameters,
            Field(description="Parameters for adding the file to the blockstore.")
        ] = AddParameters()
    ):
    '''
    Upload a file to the local block store and return its CID.
    '''
    with mcp_emu(ctx) as emu:
        created, size, cid = emu.state.upload_file(
            BytesIO(base64.b64decode(file)),
            filename=filename,
            mimetype=mimetype or "application/octet-stream",
            params=params
        )
        return UploadResponse(
            created=created,
            size=size,
            cid=cid
        )

@mcp.tool(
    annotations=dict(
        idempotentHint=True,
        openWorldHint=False
    )
)
async def insert(
        ctx: Context,
        memory: Annotated[
            DraftMemory,
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
    with mcp_emu(ctx) as emu:
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
    with mcp_emu(ctx) as emu:
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
            DraftMemory,
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
    with mcp_emu(ctx) as emu:
        qr = await emu.query(
            prompt,
            system_prompt or QUERY_PROMPT,
            recall_config,
            chat_config
        )
        return qr.chatlog + [qr.response]

@mcp.tool(
    annotations=dict(
        openWorldHint=False,
    )
)
async def chat(
        ctx: Context,
        prompt: Annotated[
            Memory,
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
    with mcp_emu(ctx) as emu:
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
def push(
        ctx: Context,
        sona: Annotated[
            UUID|str,
            Field(description="Sona to push the memory to.")
        ],
        include: Annotated[
            list[Edge[CIDv1]],
            Field(description="Additional memories to include in the ACT, keyed by label.")
        ]
    ):
    '''
    Insert a new memory into the sona to be processed by its next ACT
    (Autonomous Cognitive Thread).
    '''
    with mcp_emu(ctx) as emu:
        if u := emu.act_push(sona, include):
            return u
        raise ToolError("Sona not found or prompt memory not found.")
