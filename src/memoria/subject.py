#!/usr/bin/env python3.13

'''
Server which hosts the Memoria Subject.

Endpoints are provided for a trustless IPFS gateway, Kubo-compatible IPFS API,
a REST API for Memoria, and an Anthropic Model Context Protocol (MCP) interface.
'''

from contextlib import contextmanager
from typing import Annotated, Literal, override

from fastapi import Depends, FastAPI, Header, Query, Response
from fastapi.responses import JSONResponse

from cid import CID, CIDv1, BlockCodec
from ipfs import Blockstore, CompositeBlocksource, FlatfsBlockstore
from ipld import dagcbor

from memoria.memory import Memory
from memoria.repo import Repository
from memoria.db import database

class AppState(Blockstore):
    '''Application state for the FastAPI app.'''
    def __init__(self, blockstore: Blockstore, repo: Repository):
        self.blockstore: Blockstore = blockstore
        self.repo: Repository = repo
        self.blocksource: CompositeBlocksource = CompositeBlocksource(repo, blockstore)
    
    @override
    def block_has(self, cid: CID) -> bool:
        return self.blocksource.block_has(cid)

    @override
    def block_get(self, cid: CID) -> bytes | None:
        return self.blocksource.block_get(cid)
    
    @override
    def block_put(self,
            block: bytes,
            *,
            cid_version: Literal[0, 1] = 1,
            codec: BlockCodec = 'dag-cbor',
            function: str = 'sha2-256'
        ) -> CID:
        return self.blockstore.block_put(
            block,
            cid_version=cid_version,
            codec=codec,
            function=function
        )

def get_repo():
    '''Dependency to get the repository.'''
    with database("private/memoria.db") as db:
        yield Repository(db)

context_repo = contextmanager(get_repo)

def get_blockstore():
    '''Context manager to get the application state.'''
    with context_repo() as repo:
        yield MemoriaBlockstore(repo, FlatfsBlockstore("private/blocks"))

context_blockstore = contextmanager(get_blockstore)

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

def build_app():
    mcp_http = mcp.http_app()
    app = FastAPI(lifespan=mcp_http.lifespan)

    app.mount("/ipfs", ipfs_gateway)
    app.mount("/api/v0", ipfs_api)
    app.mount("/memoria", rest_api)

    # Nothing can be mounted after this
    app.mount("", mcp_http)

    return app

def main():
    import uvicorn
    config = uvicorn.Config(
        build_app(),
        host="0.0.0.0",
        port=8000
    )
    server = uvicorn.Server(config)
    try: server.run()
    except KeyboardInterrupt:
        print("Subject server stopped by user.")

if __name__ == "__main__":
    main()
