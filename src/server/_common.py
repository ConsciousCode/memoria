'''
Common server utilities.
'''
from contextlib import contextmanager
from typing import IO, Annotated, Literal, Optional, override
from datetime import datetime

from fastapi import Depends, FastAPI
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from ..ipld import CID, BlockCodec, Blockstore, CompositeBlocksource, FlatfsBlockstore
from ..memory import FileData, Memory
from ..repo import Repository
from ..db import database

class AddParameters(BaseModel):
    """Kubo /api/v0/add parameters with validation"""
    '''
    recursive: Annotated[
        bool, NOT_SUPPORTED('Add directory paths recursively.')
    ] = False
    wrap_with_directory: Annotated[
        bool, NOT_SUPPORTED('wrap_with_directory')
    ] = False
    only_hash: Annotated[
        bool, NOT_SUPPORTED(description="Only chunk and hash.")
    ] = False
    trickle: Annotated[
        bool, NOT_SUPPORTED("Use trickle-dag format.")
    ] = False
    raw_leaves: Annotated[
        bool, NOT_SUPPORTED(description="Use raw blocks for leaves.")
    ] = True
    '''
    cid_version: Annotated[
        Literal[0, 1], Field(description="CID version.")
    ] = 1 # We use raw-leaves by default, so CIDv1 is preferred.
    hash: Annotated[
        str, Field(description="Hash function.")
    ] = "sha2-256"
    #chunker: Annotated[
    #    Optional[str], NOT_SUPPORTED(description="Chunking algorithm.")
    #] = None
    mtime: Annotated[
        Optional[int], Field(description="File modification time in seconds since epoch.")
    ] = None

class AppState(Blockstore):
    '''Application state for the FastAPI app.'''
    def __init__(self, blockstore: Blockstore, repo: Repository):
        self.blockstore = blockstore
        self.repo = repo
        self.blocksource = CompositeBlocksource(repo, blockstore)
    
    def upload_file(self,
            stream: IO[bytes],
            filename: Optional[str],
            mimetype: str,
            params: AddParameters
        ) -> tuple[bool, int, CID]:
        '''Upload a file to the IPFS blockstore.'''
        cid = self.blockstore.ipfs_add(
            stream,
            cid_version=params.cid_version,
            function=params.hash
        )
        root = self.blockstore.block_get(cid)
        assert root, "Failed to retrieve root block after adding file."
        
        created = not self.repo.lookup_file(cid)
        timestamp = params.mtime
        self.repo.insert(
            Memory(
                data=FileData(
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
        return self.blocksource.block_has(cid)

    @override
    def block_get(self, cid: CID) -> Optional[bytes]:
        return self.blocksource.block_get(cid)
    
    @override
    def block_put(self,
            block: bytes,
            *,
            cid_version: Literal[0, 1]=1,
            codec: BlockCodec = 'dag-cbor',
            function: str = 'sha2-256'
        ) -> CID:
        return self.blockstore.block_put(
            block,
            cid_version=cid_version,
            codec=codec,
            function=function
        )

mcp = FastMCP[Repository]("memoria",
    """Coordinates a "sona" representing a cohesive identity and memory."""
)
mcp_http = mcp.http_app()
app = FastAPI(lifespan=mcp_http.lifespan)

@contextmanager
def get_repo():
    '''Dependency to get the repository.'''
    with database("private/memoria.db") as db:
        yield Repository(db)

@contextmanager
def get_appstate():
    '''Context manager to get the application state.'''
    with get_repo() as repo:
        yield AppState(FlatfsBlockstore("private/blocks"), repo)

depend_appstate = Depends(get_appstate)
depend_repo = Depends(get_repo)