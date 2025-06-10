from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
import json
from typing import Annotated, Any, Iterable, Literal, Optional, TypedDict, cast

from fastapi import FastAPI, Request
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts.base import AssistantMessage, Message, UserMessage
from mcp.server.session import ServerSession
from mcp.types import BlobResourceContents, EmbeddedResource
from pydantic import AnyUrl, Field
from starlette.exceptions import HTTPException

from ipld import CIDv1

from db import Edge
from memoria import Database, Memoria
from models import Memory, MemoryDAG, RecallConfig

@asynccontextmanager
async def lifespan(server: FastMCP):
    with Database("private/memoria.db", "files") as db:
        yield Memoria(db)

app = FastAPI()
mcp = FastMCP("memoria",
    """Coordinates a "sona" representing a cohesive identity and memory.""",
    lifespan=lifespan
)
ipfs = FastAPI()

app.mount("/mcp", mcp.sse_app())
app.mount("/ipfs", ipfs)

def mcp_context() -> tuple[Context[ServerSession, Memoria], Memoria]:
    '''Get the current request context from the FastAPI request.'''
    ctx = cast(Context[ServerSession, Memoria], mcp.get_context())
    return ctx, ctx.request_context.lifespan_context

@ipfs.get("/bafkqaaa")
def get_ipfs_empty(request: Request):
    '''Empty block for PING'''
    # Return empty body? What format? Probably depends on car/ipld format parameter.
    return

@ipfs.get("/{path:path}")
def get_ipfs(path: str, request: Request):
    pass

@mcp.resource("memory://{cid}")
def memory_resource(cid: str):
    '''
    Memory resource handler.
    '''
    try:
        ctx = mcp.get_context()
        memoria = cast(Memoria, ctx.request_context.lifespan_context)
        if (m := memoria.lookup_memory(CIDv1(cid))) is None:
            return {"error": "Memory not found"}, 404
        
        return m
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@mcp.tool()
def insert(
        sona: Annotated[
            Optional[str],
            Field(description="Sona to push the memory to.")
        ],
        memory: Annotated[
            Memory,
            Field(description="Memory to insert.")
        ],
        index: Annotated[
            Optional[str],
            Field(description="Plaintext indexing field for the memory if available.")
        ] = None,
        importance: Annotated[
            Optional[float],
            Field(description="Initial importance of the memory [0-1] biasing how easily it's recalled.")
        ] = None
    ) -> str:
    '''Insert a new memory into the sona.'''
    ctx, memoria = mcp_context()
    return str(memoria.insert(memory, sona, index, importance))

@mcp.tool()
def act_push(
        sona: Annotated[
            str,
            Field(description="Sona to push the memory to.")
        ],
        prompt: Annotated[
            Memory|str,
            Field(description="Memory to insert as a prompt for the ACT.")
        ],
        index: Annotated[
            Optional[str],
            Field(description="Plaintext indexing field for the memory if available.")
        ] = None,
        include: Annotated[
            Optional[dict[str, list[Edge]]],
            Field(description="Additional memories to include in the ACT, keyed by label.")
        ] = None,
        importance: Annotated[
            Optional[float],
            Field(description="Initial importance of the memory [0-1] biasing how easily it's recalled.")
        ] = None
    ):
    '''Insert a new memory into the sona, formatted for an ACT (Autonomous Cognitive Thread).'''
    ctx, memoria = mcp_context()
    if not memoria.act_push(sona, prompt, index, include, importance):
        raise HTTPException(404, detail=f"Sona '{sona}' not found or prompt memory not found.")

@mcp.tool()
def recall(
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
        include: Annotated[
            Optional[list[str]],
            Field(description="List of memory CIDv1 to include as part of the recall. Example: the previous message in a chat log.")
        ] = None
    ) -> dict[str, Memory]:
    '''
    Recall memories related to the prompt, including relevant included memories
    and their dependencies.
    '''
    ctx, memoria = mcp_context()

    dag = memoria.recall(
        sona,
        prompt,
        timestamp,
        config,
        map(CIDv1, include or [])
    )
    return {
        str(cid): mem
            for cid, mem in dag.items()
    }

def memory_to_message(ref: Optional[int], refs: dict[CIDv1, int], memory: Memory) -> Message:
    '''Render memory for the context.'''

    p: dict[str, Any] = {
        label: [r for e in edges if (r := refs.get(e.target))]
            for label, edges in memory.edges.items()
                if edges
    }
    # Not included:
    # - importance - do not expose to the agent never ever
    if ref: p["id"] = ref
    if memory.timestamp:
        p['datetime'] = (datetime
            .fromtimestamp(memory.timestamp)
            .replace(microsecond=0)
            .isoformat()
        )
    
    match memory.kind:
        case "self":
            return AssistantMessage(json.dumps({
                **memory.data.model_dump(),
                **p
            }))
        case "text" if isinstance(memory.data, str):
            return AssistantMessage(json.dumps({
                "text": memory.data,
                **p
            }))
        case "other":
            return UserMessage(json.dumps({
                **memory.data.model_dump(),
                **p
            }))
        case "file":
            return AssistantMessage(EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(
                    uri=AnyUrl(f"memory://{memory.cid}"),
                    mimeType=memory.data.mimeType or "application/octet-stream",
                    blob=memory.data.content
                )
            ))
        
        case _:
            raise ValueError(f"Unknown memory kind: {memory.kind}")

@mcp.prompt()
def act_next(
        sona: Annotated[
            str,
            Field(description="Sona to process within.")
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
        include: Annotated[
            Optional[list[str]],
            Field(description="List of memory CIDv1 to include as part of the recall. Example: the previous message in a chat log.")
        ] = None
    ) -> list[Message]:
    '''Get the prompt for the next step of an ACT (Autonomous Cognitive Thread).'''
    ctx, memoria = mcp_context()

    g = memoria.recall(
        sona,
        prompt,
        timestamp or datetime.now().timestamp(),
        config,
        (CIDv1(inc) for inc in include or [])
    )

    # Serialize the memory graph with localized references and edges.
    gv = g.invert()
    refs: dict[CIDv1, int] = {}
    messages: list[Message] = []

    for cid in gv.toposort(key=lambda v: v.timestamp):
        memory = gv[cid]
        # Only include ids if they have references
        if gv.edges(cid):
            ref = refs[cid] = len(refs)
        else:
            ref = None
        
        messages.append(memory_to_message(ref, refs, memory))
    
    return messages

async def main():
    import uvicorn

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="DEBUG"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())