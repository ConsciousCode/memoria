'''
MCP Server for the Memoria Subject
'''

from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from typing import Annotated
import base64

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ResourceError
from mcp import SamplingMessage
from mcp.types import ModelPreferences, Role, TextContent
from pydantic import BaseModel, Field
from pydantic_ai.models.mcp_sampling import MCPSamplingModelSettings

from cid import CID, CIDv1
from ipfs import CIDResolveError

from memoria.repo import Repository
from memoria.memory import AnyMemory, DraftMemory, TextData, SelfData
from memoria.config import RecallConfig, SampleConfig
from memoria.prompts import QUERY_PROMPT

from ._common import AddParameters, MemoriaBlockstore, context_blockstore, context_repo

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
def build_tags(tags: list[str]) -> str:
    return f"[{'\t'.join(tags)}]\t"

def memory_to_message(ref: int, deps: list[int], memory: AnyMemory, final: bool=False) -> tuple[Role, str]:
    '''Render memory for the context.'''
    
    tags = list[str]()
    if final: tags.append("final")
    tags.append(
        f"ref:{ref}" + (f"->{','.join(map(str, deps))}" if deps else "")
    )

    match memory.data:
        case SelfData():
            return ("assistant", build_tags(tags) + ''.join(
                getattr(p, 'content', '') for p in memory.data.parts
            ))
        
        #case TextData():
        #    tags.append("kind:raw_text")
        #    return ("user", build_tags(tags) + memory.data.content)
        
        case TextData():
            return ("user", build_tags(tags) + memory.data.content)
        
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
def mcp_emu(_ctx: Context):
    '''Get the application state from the context.'''
    with context_blockstore() as bs:
        yield MemoriaBlockstore(bs.repo, bs)

@mcp.resource("ipfs://{cid}")
def ipfs_resource(_ctx: Context, cid: CIDv1):
    '''IPFS resource handler.'''
    with context_blockstore() as state:
        try: return state.dag_get(cid)
        except CIDResolveError:
            raise ResourceError(f"CID {cid} not found in IPFS blockstore")

@mcp.resource("memoria://memory/{cid}")
def memory_resource(_ctx: Context, cid: CIDv1):
    '''Memory resource handler.'''
    with context_repo() as repo:
        if m := repo.lookup_memory(cid):
            return m
        raise ResourceError("Memory not found")

class UploadResponse(BaseModel):
    created: bool = Field(
        description="Whether the file was newly inserted."
    )
    size: int = Field(
        description="Size of the uploaded file including IPFS overhead in bytes."
    )
    cid: CID = Field(
        description="CID of the uploaded file."
    )

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
            str | None,
            Field(description="Filename to use for the uploaded file.")
        ] = None,
        mimetype: Annotated[
            str | None,
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
        index: Annotated[
            list[str] | None,
            Field(description="List of representative samples of the memory for use in vector similarity searching.")
        ] = None,
        timestamp: Annotated[
            int | None,
            Field(description="Timestamp to use for the recall. If not provided, uses the current time.")
        ] = None,
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
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())
        return emu.repo.recall(
            prompt, timestamp, index, recall_config
        ).adj

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
        index: Annotated[
            list[str]|None,
            Field(description="List of representative samples of the memory for use in vector similarity searching.")
        ] = None,
        timestamp: Annotated[
            int | None,
            Field(description="Timestamp to use for the query. If not provided, uses the current time.")
        ] = None,
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
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())
        g = emu.repo.recall(prompt, timestamp, index, recall_config)

        refs: dict[CIDv1, int] = {}
        chatlog: list[AnyMemory] = []
        messages: list[str | SamplingMessage] = []
        #for msg in serialize_dag(g, refs):
        for cid in g.invert().toposort(key=lambda v: v.timestamp):
            # We need ids for each memory so their edges can be annotated later
            ref = refs[cid] = len(refs) + 1
            memory = g[cid].memory
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
                    parts=[SelfData.TextPart(content=response.text)]
                )
            
            case _:
                raise NotImplementedError(
                    "Query response must be a TextContent, got: "
                    + f"{type(response)}: {response}"
                )
        
        # TODO: empty edges?
        chatlog.append(DraftMemory(data=data, edges=[]))
        # TODO: No support for images, would require inspecting the memories
        # themselves for content. Don't want to spend more time right now on
        # a debug endpoint
        return [m.model_dump() for m in chatlog]
