'''
MCP Server for Memoria
'''

from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from typing import Annotated, Optional, cast
from uuid import UUID
import base64

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ResourceError, ToolError
from mcp import SamplingMessage
from mcp.types import ModelPreferences, Role, TextContent
from pydantic import Field
from pydantic_ai.mcp import ToolResult
from pydantic_ai.models.mcp_sampling import MCPSamplingModelSettings

from ipld import CIDv1
from ipfs import CIDResolveError

from memoria.repo import Repository
from memoria.memory import AnyMemory, DraftMemory, Edge, OtherData, RecallConfig, SampleConfig, SelfData, TextData, UploadResponse
from memoria.prompts import QUERY_PROMPT

from . _common import AddParameters, MemoriaBlockstore, context_blockstore, context_repo

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

mcp = FastMCP[Repository]("memoria",
    """Coordinates a "sona" representing a cohesive identity and memory."""
)
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

@contextmanager
def mcp_emu(ctx: Context):
    '''Get the application state from the context.'''
    with context_blockstore() as bs:
        yield MemoriaBlockstore(bs.repo, bs)

@mcp.resource("ipfs://{cid}")
def ipfs_resource(ctx: Context, cid: CIDv1):
    '''IPFS resource handler.'''
    with context_blockstore() as state:
        try: return state.dag_get(cid)
        except CIDResolveError:
            raise ResourceError(f"CID {cid} not found in IPFS blockstore")

@mcp.resource("memoria://sona/{uuid}")
def sona_resource(ctx: Context, uuid: UUID):
    '''Sona resource handler.'''
    with context_repo() as repo:
        if m := repo.find_sona(uuid):
            return m
        raise ResourceError("Sona not found")

@mcp.resource("memoria://memory/{cid}")
def memory_resource(ctx: Context, cid: CIDv1):
    '''Memory resource handler.'''
    with context_repo() as repo:
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
        created, size, cid = emu.upload_file(
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
        return emu.repo.recall(prompt, recall_config).adj

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
            str,
            Field(description="System prompt to use for the query.")
        ] = QUERY_PROMPT,
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
        g = emu.repo.recall(prompt, recall_config)

        refs: dict[CIDv1, int] = {}
        chatlog: list[AnyMemory] = []
        messages: list[str | SamplingMessage] = []
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

        response = await ctx.sample(
            messages,
            system_prompt=system_prompt,
            temperature=chat_config.temperature,
            max_tokens=chat_config.max_tokens,
            model_preferences=chat_config.model_preferences
        )
        match response:
            case TextContent():
                data=SelfData(
                    parts=[SelfData.Part(content=response.text)],
                    stop_reason=None
                )
            
            case _:
                raise NotImplementedError(
                    "Query response must be a TextContent, got: "
                    f"{type(response)}: {response}"
                )
        
        chatlog.append(DraftMemory(
            data=data,
            timestamp=int(datetime.now().timestamp())
        ))
        # TODO: No support for images, would require inspecting the memories
        # themselves for content. Don't want to spend more time right now on
        # a debug endpoint
        return cast(ToolResult, [
            m.model_dump() for m in chatlog
        ])

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
        if u := emu.repo.act_push(sona, include):
            return u
        raise ToolError("Sona not found or prompt memory not found.")
