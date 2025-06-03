from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Annotated, Optional, TypedDict, cast

from mcp.server.fastmcp import Context, FastMCP
from openai import BaseModel
from pydantic import Field
from starlette.exceptions import HTTPException
import cid

from memoria import Database, Memoria
from util import json_t

@asynccontextmanager
async def lifespan(server: FastMCP):
    with Database("memoria.db", "files") as db:
        yield Memoria(db)


mcp = FastMCP("memoria",
    """Coordinates a "sona" representing a cohesive identity and memory.""",
    lifespan=lifespan
)

class MemoryEdge(BaseModel):
    weight: float
    target: cid.CIDv1

class MemorySchema(BaseModel):
    kind: str
    data: json_t
    timestamp: Optional[float]
    edges: dict[str, list[MemoryEdge]]

class ErrorResult(TypedDict):
    error: str

'''
memory://[{cidv1}/edges/{label}/{index}/target]
'''

@mcp.resource("memory://{path}")
async def memory_resource(rowid: int):
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
                target=cid.from_bytes(edge.dst.cid),
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

@mcp.tool("recall")
async def recall(
        prompt: Annotated[str, Field(description="Prompt to base the recall on.")],
        include: Annotated[Optional[list[str]], Field(description="List of memory UUIDs to include as part of the recall. Example: the previous message in a chat log.")] = None
    ) -> list[MemorySchema]:
    '''
    Recall memories related to the prompt, including relevant extra memories
    and their dependencies.
    '''
    ctx = mcp.get_context()
    memoria = cast(Memoria, ctx.request_context.lifespan_context)
    inc = []
    for s in include or []:
        if isinstance(c := cid.make_cid(s), cid.CIDv1):
            inc.append(c)
        else:
            raise ValueError(f"Expected CIDv1, got {type(c)}")
    results = memoria.recall(prompt, inc)

    edges = defaultdict(list[MemoryEdge])
    for m in results.values():
        for edge in memoria.db.backward_edges(m.rowid):
            edges[edge.label].append(MemoryEdge(
                target=cid.from_bytes(edge.dst.cid),
                weight=edge.weight
            ))
    
    return [
        MemorySchema(
            kind=m.kind,
            data=m.data,
            timestamp=m.timestamp,
            edges=edges
        ) for m in results.values()
    ]

@mcp.prompt()
def converse(name: str=""):
    '''Talk to the agent as a user.'''
    if not name:
        return "I'm talking to a user."
    return f"I'm talking to a user named {name}."

def main():
    mcp.run()

if __name__ == "__main__":
    main()