'''
Common server utilities.
'''
from contextlib import asynccontextmanager
from typing import IO, Annotated, Literal, Optional, override
from datetime import datetime

from fastapi import Depends, FastAPI
from fastmcp import FastMCP
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from src.ipld.cid import CID, Codec
from src.ipld.ipfs import Blocksource, Blockstore, CompositeBlocksource, FlatfsBlockstore
from src.models import Memory
from src.memoria import Memoria, Database

class UnsupportedError(NotImplementedError):
    pass

class NotSupportedValidator:
    def validate_with_field_name(self, value, wrap, info):
        raise UnsupportedError(info.field_name)
    
    def __get_pydantic_core_schema__(self,
            source_type,
            handler: GetCoreSchemaHandler,
        ) -> CoreSchema:
        return core_schema.with_info_wrap_validator_function(
            self.validate_with_field_name,
            handler(source_type)
        )

def NOT_SUPPORTED(description: str):
    return NotSupportedValidator(), Field(
        description=description + " [NOT SUPPORTED]"
    )

class AddParameters(BaseModel):
    """Kubo /api/v0/add parameters with validation"""
    recursive: Annotated[
        bool, *NOT_SUPPORTED('Add directory paths recursively.')
    ] = False
    wrap_with_directory: Annotated[
        bool, *NOT_SUPPORTED('wrap_with_directory')
    ] = False
    pin: Annotated[
        bool, *NOT_SUPPORTED("Pin locally to protect from GC.")
    ] = False
    progress: Annotated[
        bool, Field(description="Stream progress data.")
    ] = False
    quiet: Annotated[
        bool, Field(description="Write minimal output.")
    ] = False
    quieter: Annotated[
        bool, Field(description="Write only final hash.")
    ] = False
    silent: Annotated[
        bool, Field(description="Write no output.")
    ] = False
    only_hash: Annotated[
        bool, Field(description="Only chunk and hash.")
    ] = False
    trickle: Annotated[
        bool, *NOT_SUPPORTED("Use trickle-dag format.")
    ] = False
    raw_leaves: Annotated[
        bool, Field(description="Use raw blocks for leaves.")
    ] = False
    cid_version: Annotated[
        Literal[0, 1], Field(description="CID version.")
    ] = 0
    hash: Annotated[
        str, Field(description="Hash function.")
    ] = "sha2-256"
    chunker: Annotated[
        Optional[str], Field(description="Chunking algorithm.")
    ] = None
    mtime: Annotated[
        Optional[int], Field(description="File modification time in seconds since epoch.")
    ] = None

class AppState(Blockstore):
    '''Application state for the FastAPI app.'''
    def __init__(self, blockstore: Blockstore, memoria: Memoria):
        self.blockstore: Blockstore = blockstore
        self.memoria: Memoria = memoria
        self.blocksource: Blocksource = CompositeBlocksource(
            memoria, blockstore
        )
    
    def upload_file(self,
            stream: IO[bytes],
            filename: Optional[str],
            mimetype: str,
            timestamp: Optional[int]
        ) -> tuple[bool, int, CID]:
        '''Upload a file to the IPFS blockstore.'''
        cid = self.blockstore.ipfs_add(stream)
        root = self.blockstore.block_get(cid)
        assert root, "Failed to retrieve root block after adding file."
        created = not self.memoria.lookup_file(cid)
        self.memoria.insert(
            Memory(
                data=Memory.FileData(
                    file=cid,
                    filename=filename,
                    mimetype=mimetype,
                    filesize=stream.tell()
                ),
                timestamp=timestamp or int(datetime.now().timestamp())
            )
        )
        return created, len(root), cid
    
    @override
    def block_has(self, cid: CID) -> bool:
        return self.memoria.block_has(cid) or self.blocksource.block_has(cid)

    @override
    def block_get(self, cid: CID) -> Optional[bytes]:
        return self.memoria.block_get(cid) or self.blocksource.block_get(cid)
    
    @override
    def block_put(self,
            block: bytes,
            *,
            codec: Codec = 'dag-cbor',
            function: str = 'sha2-256'
        ) -> CID:
        return self.blockstore.block_put(block, codec=codec, function=function)

@asynccontextmanager
async def root_lifespan(app: FastAPI):
    '''Lifespan context for the FastAPI app.'''
    with Database("private/memoria.db", "files") as db:
        app.state.data = AppState(
            FlatfsBlockstore("private/blocks"),
            Memoria(db)
        )
        yield

app = FastAPI(lifespan=root_lifespan)

@asynccontextmanager
async def subapp_lifespan(subapp: FastAPI):
    '''Lifespan context for a subapp.'''
    if not hasattr(app.state, "state"):
        raise RuntimeError("App state not initialized.")
    subapp.state.data = app.state.data
    yield
    del subapp.state.data

depend_appstate = Depends(lambda request: request.app.state.data)

@asynccontextmanager
async def mcp_lifespan(server: FastMCP):
    '''Lifespan context for the FastAPI app.'''
    yield app.state.data