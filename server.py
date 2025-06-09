from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
import json
from typing import Annotated, Any, Iterable, Literal, Optional, TypedDict, cast

from fastapi import FastAPI, Request
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts.base import Message
from mcp.server.session import ServerSession
from openai import BaseModel
from pydantic import Field
from starlette.exceptions import HTTPException

from ipld import CID, CIDv0, CIDv1

from db import MemoryKind, MemoryRow, Edge
from memoria import Database, Memoria
from models import FileMemory, Memory, MemoryDAG, OtherMemory, RecallConfig, SelfMemory, TextMemory
from util import X, json_t
from graph import Graph

def localize_memory(g: Graph[int, tuple[str, float], MemoryRow]) -> Iterable[tuple[Optional[int], MemoryRow, dict[str, int]]]:
    '''
    Serialize the memory graph into a sequence of memories with localized
    references and edges.
    '''
    gv = g.invert()
    refs: dict[int, int] = {} # rowid: ref index

    for rowid in gv.toposort(key=lambda v: v.timestamp):
        # Only include ids if they have references
        if gv.edges(rowid):
            ref = refs[rowid] = len(refs)
        else:
            ref = None
        
        # We don't have to check if rowid is in refs because of toposort
        yield ref, gv[rowid], {
            e: refs[rowid]
                for rowid, (e, w) in gv.edges(rowid).items()
        }

def format_memory(ref: Optional[int], memory: MemoryRow, edges: dict[str, Any]) -> str:
    '''Render memory for the context.'''

    p = {
        "id": ref,
        "datetime": memory.timestamp and datetime
            .fromtimestamp(memory.timestamp)
            .replace(microsecond=0)
            .isoformat(),
        "importance": memory.importance,
        **edges
    }
    
    match memory.kind:
        case "self":
            data = cast(SelfMemory, memory.data)
            return X("self", name=data['name'], **p)(data['content'])
        case "other":
            data = cast(OtherMemory, memory.data)
            return X("other", name=data['name'], **p)(data['content'])
        case "text" if isinstance(memory.data, str):
            return X("text", **p)(memory.data)
        case kind:
            return X(kind, **p)(json.dumps(memory.data))

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

class MemoryEdge(BaseModel):
    weight: float
    target: CIDv1

class MemorySchema(BaseModel):
    kind: MemoryKind
    data: json_t
    timestamp: Optional[float]
    edges: dict[str, list[MemoryEdge]]

class ErrorResult(TypedDict):
    error: str

'''
memory://[{cidv1}/edges/{label}/{index}/target]
'''

@mcp.resource("memory://{cid}")
def memory_resource(cid: str):
    '''
    Memory resource handler.
    '''
    try:
        ctx = mcp.get_context()
        memoria = cast(Memoria, ctx.request_context.lifespan_context)
        if (m := memoria.db.select_memory(CIDv1(cid))) is None:
            return {"error": "Memory not found"}, 404
        
        edges = defaultdict(list[MemoryEdge])
        for edge in memoria.db.backward_edges(m.rowid):
            edges[edge.label].append(MemoryEdge(
                target=CIDv1(edge.dst.cid),
                weight=edge.weight
            ))

        return MemorySchema(
            timestamp=m.timestamp,
            kind=m.kind,
            data=m.data,
            edges=edges
        )
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
            str,
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
    ) -> MemoryDAG:
    '''
    Recall memories related to the prompt, including relevant included memories
    and their dependencies.
    '''
    ctx, memoria = mcp_context()

    return memoria.recall(
        prompt,
        list(map(CIDv1, include or [])),
        timestamp,
        config
    )

@mcp.prompt()
def acthread(
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
    '''Formatted context for an ACT (Autonomous Cognitive Thread).'''
    ctx, memoria = mcp_context()

    g = memoria.recall(
        prompt,
        (CIDv1(inc) for inc in include or []),
        timestamp or datetime.now().timestamp(),
        config
    )

    # Serialize the memory graph with localized references and edges.
    gv = g.invert()
    refs: dict[int, tuple[int, MemoryRow]] = {}

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