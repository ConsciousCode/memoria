from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
import json
from typing import Annotated, Any, Iterable, Literal, Optional, TypedDict, cast

from fastapi import FastAPI, Request
from mcp.server.fastmcp import Context, FastMCP
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

@mcp.resource("memory://{path}")
def memory_resource(rowid: int):
    '''
    Memory resource handler.
    '''
    try:
        ctx = mcp.get_context()
        memoria = cast(Memoria, ctx.request_context.lifespan_context)
        if (m := memoria.db.select_memory(rowid)) is None:
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
        memory: Annotated[
            Memory,
            Field(description="Memory to insert into the sona.")
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
    ctx = mcp.get_context()
    memoria = cast(Memoria, ctx.request_context.lifespan_context)
    return str(memoria.insert(memory, index, importance))

@mcp.tool()
def recall(
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
    Recall memories related to the prompt, including relevant extra memories
    and their dependencies.
    '''
    ctx, memoria = mcp_context()

    return memoria.recall(
        prompt,
        list(map(CIDv1, include or [])),
        timestamp,
        config
    )

class ProcessResponse(BaseModel):
    sona_uuid: str
    result_cid: str

@mcp.tool()
def process(
        sona: Annotated[str, Field(description="Sona to process within.")],
        prompt: Annotated[str, Field(description="Prompt to process.")],
        include: Annotated[Optional[dict[str, str]], Field(description="List of memory CIDv1 to include as part of the processing. Example: the previous message in a chat log.")] = None
    ) -> str:
    '''
    Process a prompt and return a response.
    '''
    ctx = mcp.get_context()
    memoria = cast(Memoria, ctx.request_context.lifespan_context)
    
    if include is None:
        include = {}
    ts = datetime.now()

    s = memoria.find_sona(sona)

    g = memoria.recall(prompt, (
        cid.from_string(e) for edges in include.values()
            for e in edges
    ))

    # Serialize the memory graph with localized references and edges.
    gv = g.invert()
    refs: dict[int, tuple[int, MemoryRow]] = {} # rowid: ref index
    memories: list[str] = []

    for rowid in gv.toposort(key=lambda v: v.timestamp):
        # Only include ids if they have references
        if gv.edges(rowid):
            ref = len(refs)
            refs[rowid] = (ref, gv[rowid])
        else:
            ref = None
        
        memories.append(format_memory(
            ref, gv[rowid], {
                e: refs[rowid]
                    for rowid, (e, w) in gv.edges(rowid).items()
            }
        ))

    result = yield memories
    
    yield f"I remember... {memory}"
    with capture_run_messages() as messages:
        try:
            result = await system1.run(
                prompt,
                deps=System1Deps(
                    instructions=instructions,
                    memories=memories
                ),
                output_type=System1ResponseModel
            )
        finally:
            print(messages)
    output = result.output

    p = memoria.append(
        "other", {
            "name": name,
            "content": prompt
        },
        prompt, ts, None
    )
    # Do I need the importance of the response?
    r = memoria.append(
        "self", {
            "name": None,
            "content": output.response
        },
        output.response,
        datetime.now(),
        edges={
            "prompt": [Edge(1.0, p.cid)],
            "ref": [
                Edge(weight / 10, cid.from_bytes(refs[int(ref)][1].cid))
                    for ref, weight in output.weights.items()
            ],
            **{
                label: [Edge(1.0, cid) for cid in edges]
                    for label, edges in include.items()
            }
        }
    )
    
    memoria.db.commit()
    
    return {
        "id": r,
        "response": output.response,
        "weights": output.weights
    }

@mcp.prompt()
def acthread(
        include: Annotated[Optional[dict[str, str]], Field(description="List of memory CIDv1 to include as part of the processing. Example: the previous message in a chat log.")] = None
    ) -> ProcessResponse:
    '''
    Process a prompt and return a response.
    '''
    ctx = mcp.get_context()
    memoria = cast(Memoria, ctx.request_context.lifespan_context)
    
    if include is None:
        include = {}
    
    ts = datetime.now()

    g = memoria.recall(prompt, (
        cid.from_string(e) for edges in include.values()
            for e in edges
    ))

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