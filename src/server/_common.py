'''
Common server utilities.
'''
from contextlib import asynccontextmanager, contextmanager
from typing import IO, Annotated, Literal, Optional, override
from datetime import datetime

from fastapi import Depends, FastAPI
from fastmcp import FastMCP
from pydantic import BaseModel, Field, GetCoreSchemaHandler, model_validator
from pydantic_core import CoreSchema, core_schema

from ..ipld import CID, BlockCodec, Blockstore, CompositeBlocksource, FlatfsBlockstore
from ..models import FileData, Memory
from ..memoria import Repository, database

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
    return Field(
        description=description + " [NOT SUPPORTED]",
        json_schema_extra={"not_supported": True} # type: ignore
    )

class AddParameters(BaseModel):
    """Kubo /api/v0/add parameters with validation"""
    recursive: Annotated[
        bool, NOT_SUPPORTED('Add directory paths recursively.')
    ] = False
    wrap_with_directory: Annotated[
        bool, NOT_SUPPORTED('wrap_with_directory')
    ] = False
    pin: Annotated[
        bool, NOT_SUPPORTED("Pin locally to protect from GC.")
    ] = False
    progress: Annotated[
        bool, NOT_SUPPORTED(description="Stream progress data.")
    ] = False
    quiet: Annotated[
        bool, NOT_SUPPORTED(description="Write minimal output.")
    ] = False
    quieter: Annotated[
        bool, NOT_SUPPORTED(description="Write only final hash.")
    ] = False
    silent: Annotated[
        bool, NOT_SUPPORTED(description="Write no output.")
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
    cid_version: Annotated[
        Literal[0, 1], Field(description="CID version.")
    ] = 1 # We use raw-leaves by default, so CIDv1 is preferred.
    hash: Annotated[
        str, Field(description="Hash function.")
    ] = "sha2-256"
    chunker: Annotated[
        Optional[str], NOT_SUPPORTED(description="Chunking algorithm.")
    ] = None
    mtime: Annotated[
        Optional[int], NOT_SUPPORTED(description="File modification time in seconds since epoch.")
    ] = None

    @model_validator(mode="after")
    def check_not_supported_fields(self):
        for name, field in AddParameters.model_fields.items():
            extra = field.json_schema_extra
            if callable(extra):
                continue
            if extra and extra.get("not_supported"):
                value = getattr(self, name)
                default = field.default
                # If the value is not the default, raise error
                if value != default:
                    raise UnsupportedError(f"Field '{name}' is not supported.")
        return self

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
        yield AppState(
            FlatfsBlockstore("private/blocks"),
            repo
        )

depend_appstate = Depends(get_appstate)
depend_repo = Depends(get_repo)