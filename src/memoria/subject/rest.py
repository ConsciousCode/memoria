'''
Simple RESTful API separate from the MCP API.
'''
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Query, Response
from fastapi.responses import JSONResponse

from ipld import dagcbor
from cid import CIDv1

from memoria.repo import Repository
from memoria.memory import Memory

from ._common import get_repo

rest_api = FastAPI(
    title="Memoria REST API",
    description="A RESTful API for the Memoria system."
)

@rest_api.post("/memory")
async def add_memory(
        memory: Memory,
        repo: Repository = Depends(get_repo)
    ):
    '''Add a memory to the Memoria system.'''
    repo.insert(memory)
    return JSONResponse({"cid": memory.cid})

@rest_api.get("/memory/{cid}")
def get_memory(
        cid: CIDv1,
        accept: Annotated[list[str] | None, Header()] = None,
        repo: Repository = Depends(get_repo),
    ):
    '''Get a memory by CID.'''
    if memory := repo.lookup_memory(cid):
        accept = accept or []
        if "application/cbor" in accept:
            return dagcbor.marshal(memory)
        return memory
    return Response(
        status_code=404,
        content=f"Memory with CID {cid} not found."
    )

@rest_api.get("/memories")
def list_memories(
        page: Annotated[
            int, Query(description="Page number to return.")
        ] = 1,
        perpage: Annotated[
            int, Query(description="Number of messages to return per page.")
        ] = 100,
        accept: Annotated[list[str] | None, Header()] = None,
        repo: Repository = Depends(get_repo)
    ):
    '''List messages in the Memoria system.'''
    messages = repo.list_messages(page, perpage)
    accept = accept or []
    if "application/cbor" in accept:
        return dagcbor.marshal(messages)
    return messages
