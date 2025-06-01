from contextlib import asynccontextmanager
from typing import Annotated, Optional, TypedDict, cast
from mcp.server.fastmcp import Context, FastMCP
from openai import BaseModel
from pydantic import Field
from starlette.exceptions import HTTPException

from memoria import Database, Memoria
from util import JsonValue

@asynccontextmanager
async def lifespan(server: FastMCP):
    with Database("memoria.db", "files") as db:
        yield Memoria(db)


mcp = FastMCP("memoria",
    """Coordinates a "sona" representing a cohesive identity and memory.""",
    lifespan=lifespan
)

class MemoryEdge(BaseModel):
    dst: int
    weight: float

class MemorySchema(BaseModel):
    rowid: int
    timestamp: Optional[float]
    kind: str
    data: JsonValue
    importance: Optional[float]
    edges: dict[str, MemoryEdge]

class ErrorResult(TypedDict):
    error: str

@mcp.resource("memory://{rowid}")
async def memory_resource(rowid: int):
    '''
    Memory resource handler.
    '''
    try:
        ctx = mcp.get_context()
        memoria = cast(Memoria, ctx.request_context.lifespan_context)
        m = memoria.db.select_memory(rowid)
        if m is None:
            return {"error": "Memory not found"}, 404
        
        return MemorySchema(
            rowid= m.rowid,
            timestamp= m.timestamp and m.timestamp.timestamp(),
            kind= m.kind,
            data= m.data,
            importance= m.importance,
            edges={
                edge.label: MemoryEdge(
                    dst=edge.dst.rowid,
                    weight=edge.weight
                ) for edge in memoria.db.backward_edges(m.rowid)
            }
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@mcp.tool("recall")
async def recall(
        prompt: Annotated[str, Field(description="Prompt to base the recall on.")],
        include: Annotated[Optional[list[int]], Field(description="List of memory ids to include as part of the recall. Example: the previous message in a chat log.")] = None
    ) -> list[MemorySchema]:
    '''
    Recall memories related to the prompt, including relevant extra memories
    and their dependencies.
    '''
    ctx = mcp.get_context()
    memoria = cast(Memoria, ctx.request_context.lifespan_context)
    results = memoria.recall(include, prompt)
    return [
        MemorySchema(
            rowid=m.rowid,
            timestamp=m.timestamp and m.timestamp.timestamp(),
            kind=m.kind,
            data=m.data,
            importance=m.importance,
            edges={
                edge.label: MemoryEdge(
                    dst=edge.dst.rowid,
                    weight=edge.weight
                ) for edge in memoria.db.backward_edges(m.rowid)
            }
            ) for m in results
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