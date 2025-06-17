from contextlib import asynccontextmanager
from datetime import datetime
import inspect
import json
from typing import Annotated, Iterable, Literal, Optional
from uuid import UUID

from fastapi import FastAPI, Request, Response
from fastmcp.exceptions import ResourceError, ToolError
from mcp import SamplingMessage
from mcp.types import ModelPreferences, PromptMessage, TextContent
from fastmcp import Context, FastMCP
from fastmcp.prompts.prompt import Message
from pydantic import BaseModel, Field

from ipld import dagcbor_marshal
from ipld.cid import CIDv1

from db import Edge
from memoria import Database, Memoria
from models import Memory, MemoryDAG, OtherMemory, RecallConfig, SelfMemory, StopReason

@asynccontextmanager
async def lifespan(server: FastMCP):
    global gmemoria
    with Database("private/memoria.db", "files") as db:
        yield (gmemoria := Memoria(db))

mcp = FastMCP("memoria",
    """Coordinates a "sona" representing a cohesive identity and memory.""",
    lifespan=lifespan,
    #log_level="DEBUG"
)
mcp_app = mcp.http_app()
app = FastAPI(lifespan=mcp_app.lifespan)
ipfs = FastAPI()
app.mount("/ipfs", ipfs)
app.mount("", mcp_app)

def mcp_context(ctx: Context) -> Memoria:
    '''Get memoria from the FastAPI context.'''
    return ctx.request_context.lifespan_context

## IPFS trustless gateway

@ipfs.get("/bafkqaaa")
def get_ipfs_empty():
    '''Empty block for PING'''
    return Response()

@ipfs.get("/{path:path}")
def get_ipfs(path: str, request: Request):
    cid = CIDv1(path)
    if ob := gmemoria.lookup_memory(cid):
        return dagcbor_marshal(ob)
    if ob := gmemoria.lookup_act(cid):
        return dagcbor_marshal(ob)
    # ipfs_api.lookup(cid) # TODO
    return Response(
        status_code=404,
        content=f"Memory or ACT not found for CID {cid}"
    )

## MCP

@mcp.resource("ipfs://{cid}")
def ipfs_resource(ctx: Context, cid: CIDv1):
    '''IPFS resource handler.'''
    memoria = mcp_context(ctx)
    if (m := memoria.lookup_memory(cid)) is None:
        raise ResourceError("Memory not found")
    
    if m.kind != "file":
        raise ResourceError("Memory is not a file")
    
    return m.data.content

@mcp.resource("memoria://sona/{uuid}")
def sona_resource(ctx: Context, uuid: UUID):
    '''Sona resource handler.'''
    if m := mcp_context(ctx).find_sona(uuid):
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
def insert_memory(
        ctx: Context,
        sona: Annotated[
            Optional[UUID|str],
            Field(description="Sona to push the memory to.")
        ],
        memory: Annotated[
            Memory,
            Field(description="Memory to insert.")
        ],
        index: Annotated[
            Optional[str],
            Field(description="Plaintext indexing field for the memory if available.")
        ] = None
    ):
    '''Insert a new memory into the sona.'''
    return mcp_context(ctx).insert(memory, sona, index)

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
    if u := mcp_context(ctx).act_push(sona, memories):
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

@mcp.tool(
    annotations=dict(
        readOnlyHint=True,
        openWorldHint=False,
    )
)
def recall(
        ctx: Context,
        sona: Annotated[
            Optional[str],
            Field(description="Sona to focus the recall on.")
        ],
        prompt: Annotated[
            str,
            Field(description="Prompt to base the recall on.")
        ],
        timestamp: Annotated[
            Optional[float],
            Field(description="Timestamp to use for the recall, if available. If `null`, uses the current time.")
        ] = None,
        config: Annotated[
            RecallConfig,
            Field(description="Configuration for how to weight memory recall.")
        ] = RecallConfig(),
        memories: Annotated[
            Optional[list[CIDv1]],
            Field(description="List of memories to include as part of the recall. Example: the previous message in a chat log.")
        ] = None
    ):
    '''
    Recall memories related to the prompt, including relevant included memories
    and their dependencies.
    '''
    return {
        cid: mem
            for cid, mem in mcp_context(ctx).recall(
                sona,
                prompt,
                timestamp,
                config,
                memories or []
            ).items()
    }

class ConvoMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str

def build_tags(tags: list[str], timestamp: Optional[datetime]) -> str:
    if timestamp:
        tags.append(timestamp.replace(microsecond=0).isoformat())
    return f"[{'\t'.join(tags)}]\t"

def memory_to_message(ref: int, refs: list[int], memory: Memory, final: bool=False) -> ConvoMessage:
    '''Render memory for the context.'''
    
    tags = []
    if final:
        tags.append("final")
    tags.append(
        f"ref:{ref}" +
        (f" -> {','.join(map(str, refs))}" if refs else "")
    )

    if memory.timestamp:
        ts = datetime.fromtimestamp(memory.timestamp)
    else:
        ts = None
    
    match memory.kind:
        case "self":
            if name := memory.data.name:
                tags.append(f"name:{name}")
            if sr := memory.data.stop_reason:
                if sr != "finish":
                    tags.append(f"stop_reason:{sr}")
            return ConvoMessage(
                role="assistant",
                content=build_tags(tags, ts) +
                    ''.join(p.content for p in memory.data.parts),
            )
        case "text":
            tags.append("kind:raw_text")
            return ConvoMessage(
                role="user",
                content=build_tags(tags, ts) + memory.data
            )
        case "other":
            if name := memory.data.name:
                tags.append(f"name:{name}")
            return ConvoMessage(
                role="user",
                content=build_tags(tags, ts) + memory.data.content
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
            raise ValueError(f"Unknown memory kind: {memory.kind}")

def dag_to_convo(g: MemoryDAG, include_final=False) -> Iterable[tuple[int, CIDv1, ConvoMessage]]:
    '''Convert a memory DAG to a conversation.'''
    refs: dict[CIDv1, int] = {}

    for cid in g.invert().toposort(key=lambda v: v.timestamp):
        # We need ids for each memory so their edges can be annotated later
        ref = refs[cid] = len(refs) + 1
        yield ref, cid, memory_to_message(
            ref,
            [refs[dst] for dst, _ in g.edges(cid)],
            g[cid],
            final=include_final and (ref == len(g))
        )

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
        config: Annotated[
            RecallConfig,
            Field(description="Configuration for how to weight memory recall.")
        ] = RecallConfig()
    ) -> Optional[list[PromptMessage]]:
    '''Get the prompt for the next step of an ACT (Autonomous Cognitive Thread).'''
    memoria = mcp_context(ctx)
    g = memoria.act_next(
        sona,
        timestamp or datetime.now().timestamp(),
        config
    )
    if g is None:
        return None

    return [
        Message(
            role=m.role,
            content=m.content
        ) for _, _, m in dag_to_convo(g, include_final=True)
    ]

async def annotate_edges(
        ctx: Context,
        messages: list[str|SamplingMessage]
    ) -> tuple[dict[int, float], dict[str, float]]:
    '''Use sampling with a faster model to annotate which memories are connected and by how much.'''
    result = await ctx.sample(
        messages,
        system_prompt=inspect.cleandoc("""
            This task is annotation.
            
            Given the conversational memory up to the [final] tag and the agent's response, which memories were relevant to that response? Irrelevant memories are ignored. Memories which contributed are ranked 1-10 with 1 being a trivial reference and 10 meaning the response would not be possible without it.

            Additionally, the response itself is annotated for structurally recognizable importance with limited broader context. The dimensions of importance are novelty, emotional intensity, future relevance, saliency, and personal relevance. These are scored 0-10.

            The process answers with only a JSON object following this schema:
            {
                "relevance": {
                    [index]: score
                },
                "importance": {
                    "novelty": score,
                    "intensity": score,
                    "future": score,
                    "personal": score
                }
            }
            "relevance" maps memory indices (quoted, to be JSON-compliant) to their relevance score, e.g. {"2": 10, "8": 5}. The indices are the same as in the conversation, marked with ref:(index). "importance" scores the importance of the agent's response. 
        """),
        temperature=0,
        max_tokens=len(str(len(messages))) * 3,
        model_preferences=ModelPreferences(
            costPriority=0,
            speedPriority=1,
            intelligencePriority=0.2
        )
    )
    if not isinstance(result, TextContent):
        raise ValueError(
            f"Edge annotation response must be text, got {type(result)}: {result}"
        )
    print("Annotation", result.text)
    try:
        result = json.loads(result.text)
        if not isinstance(result, dict):
            raise ValueError(f"Edge annotation response must be a JSON object, got {type(result)}: {result}")
        
        importance = result.get("importance", {})
        return {
            int(k): v
                for k, v in result.get("relevance", {}).items()
        }, {
            "novelty": importance.get("novelty", 0)/10,
            "intensity": importance.get("intensity", 0)/10,
            "future": importance.get("future", 0)/10,
            "personal": importance.get("personal", 0)/10
        }
    except json.JSONDecodeError as e:
        raise ValueError(f"Edge annotation response is not valid JSON: {e}") from e

@mcp.tool(
    annotations=dict(
        openWorldHint=False,
    )
)
async def chat(
        ctx: Context,
        sona: Annotated[
            Optional[str],
            Field(description="Sona to focus the chat on.")
        ],
        message: Annotated[
            Optional[str],
            Field(description="Prompt for the chat. If `null`, use only the included memories.")
        ],
        timestamp: Annotated[
            Optional[datetime|float],
            Field(description="Timestamp to use for the chat, if available. If `null`, uses the current time.")
        ] = None,
        config: Annotated[
            RecallConfig,
            Field(description="Configuration for how to weight memory recall.")
        ] = RecallConfig(),
        memories: Annotated[
            Optional[set[CIDv1]],
            Field(description="Memories to include as part of the chat. Example: the previous message in a chat log. These are the minimum set of included memories, but more will be recalled.")
        ] = None,
        temperature: Annotated[
            Optional[float],
            Field(description="Sampling temperature for the response. If `null`, uses the default value.")
        ] = None,
        max_tokens: Annotated[
            Optional[int],
            Field(description="Maximum number of tokens to generate in the response. If `null`, uses the default value.")
        ] = None,
        model_preferences: Annotated[
            Optional[ModelPreferences | str | list[str]],
            Field(description="List of preferred models to use for the response. If `null`, uses the default model.")
        ] = None
    ):
    '''
    Single-turn conversation returning the response. This is committed to memory.
    '''
    if isinstance(timestamp, datetime):
        ts = timestamp.timestamp()
        dt = timestamp
    else:
        ts = timestamp
        dt = datetime.fromtimestamp(timestamp) if timestamp else None

    memoria = mcp_context(ctx)
    g = memoria.recall(
        sona, message, ts, config, list(memories or ())
    )

    refs: dict[int, CIDv1] = {}
    messages = []
    for ref, cid, m in dag_to_convo(g):
        refs[ref] = cid
        messages.append(
            SamplingMessage(
                role=m.role,
                content=TextContent(type="text", text=m.content)
            )
        )
    lastref = len(refs) + 1
    if message:
        messages.append(
            build_tags(
                ["final", f"ref:{lastref}"],
                timestamp=dt
            ) + message
        )
    print("First", messages)

    result = await ctx.sample(
        messages,
        system_prompt="I'm talking to a user. The Memoria system will replay my memories with metadata annotations, then I can respond in plaintext after the memory with the [final] tag.",
        temperature=temperature,
        max_tokens=max_tokens,
        model_preferences=model_preferences
    )
    if not isinstance(result, TextContent):
        raise ValueError(
            f"Chat response must be text, got {type(result)}: {result}"
        )
    print("Result", result)
    
    messages.append(result.text)
    rel, imp = await annotate_edges(ctx, messages)
    importance = sum(x**2 for x in imp.values()) / sum(imp.values()) / 10

    # Now that there's no risk of interruption, insert the message
    if message:
        refs[lastref] = memoria.insert(
            OtherMemory(
                timestamp=ts,
                data=OtherMemory.Data(
                    content=message
                ),
                importance=importance*rel.get(lastref, 0),
            ),
            sona,
            index=message
        )

    # Insert the response
    memoria.insert(
        SelfMemory(
            timestamp=datetime.now().timestamp(),
            data=SelfMemory.Data(
                parts=[SelfMemory.Data.Part(content=result.text)]
            ),
            edges=[
                Edge(target=refs[ref], weight=weight)
                    for ref, weight in rel.items()
            ],
            importance=importance
        ),
        sona=sona,
        index=result.text
    )

    return result

@mcp.tool(
    annotations=dict(
        readOnlyHint=True,
        openWorldHint=False,
    )
)
async def query(
        ctx: Context,
        sona: Annotated[
            Optional[str],
            Field(description="Sona to focus the chat on.")
        ],
        message: Annotated[
            Optional[str],
            Field(description="Prompt for the chat. If `null`, use only the included memories.")
        ],
        timestamp: Annotated[
            Optional[datetime|float],
            Field(description="Timestamp to use for the chat, if available. If `null`, uses the current time.")
        ] = None,
        config: Annotated[
            RecallConfig,
            Field(description="Configuration for how to weight memory recall.")
        ] = RecallConfig(),
        memories: Annotated[
            Optional[set[CIDv1]],
            Field(description="Memories to include as part of the chat. Example: the previous message in a chat log. These are the minimum set of included memories, but more will be recalled.")
        ] = None,
        temperature: Annotated[
            Optional[float],
            Field(description="Sampling temperature for the response. If `null`, uses the default value.")
        ] = None,
        max_tokens: Annotated[
            Optional[int],
            Field(description="Maximum number of tokens to generate in the response. If `null`, uses the default value.")
        ] = None,
        model_preferences: Annotated[
            Optional[ModelPreferences | str | list[str]],
            Field(description="List of preferred models to use for the response. If `null`, uses the default model.")
        ] = None
    ):
    '''Single-turn conversation returning the response.'''
    if isinstance(timestamp, datetime):
        ts = timestamp.timestamp()
        dt = timestamp
    else:
        ts = timestamp
        dt = datetime.fromtimestamp(timestamp) if timestamp else None

    memoria = mcp_context(ctx)
    g = memoria.recall(
        sona, message, ts, config, list(memories or ())
    )

    refs: dict[int, CIDv1] = {}
    messages = []
    for ref, cid, m in dag_to_convo(g):
        refs[ref] = cid
        messages.append(
            SamplingMessage(
                role=m.role,
                content=TextContent(type="text", text=m.content)
            )
        )
    if message:
        messages.append(
            build_tags(
                ["final", f"ref:{len(refs) + 1}"],
                timestamp=dt
            ) + message
        )

    return await ctx.sample(
        messages,
        system_prompt="I'm talking to a user. The Memoria system will replay my memories with metadata annotations, then I can respond in plaintext after the memory with the [final] tag. This is being run as a query so I won't remember it.",
        temperature=temperature,
        max_tokens=max_tokens,
        model_preferences=model_preferences
    )

def main():
    import uvicorn
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    main()