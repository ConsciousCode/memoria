'''
Implement the IPFS Trustless Gateway specification and any IPFS-related utilities.
'''
from typing import Annotated, Optional

from fastapi import FastAPI, Header, Query, Request, Response

from ._common import AppState, subapp_lifespan, depend_appstate
from src.ipld import CID

ROOT = "private/blocks"

EMPTY_CID = CID("bafkqaaa")

ipfs_gateway = FastAPI(
    lifespan=subapp_lifespan,
    title="IPFS Trustless Gateway",
    description="A FastAPI implementation of the IPFS Trustless Gateway specification.",
    version="0.1.0"
)

@ipfs_gateway.get(f"/{EMPTY_CID}")
def get_ipfs_empty():
    '''Empty block for PING'''
    return Response()

@ipfs_gateway.get("/{path:path}")
def get_ipfs(
        path: str,
        accept: Annotated[Optional[list[str]], Header()] = None,
        state: AppState = depend_appstate
    ):
    if accept is None:
        accept = []
    
    if "application/cbor" in accept:
        output_codec = "dag-cbor"
        mime = "application/cbor"
    else:
        output_codec = "dag-json"
        mime = "application/json"
    
    block = state.dag_get(CID(path), output_codec)
    if block is None:
        return Response(
            status_code=404,
            content=f"Block with CID {path} not found"
        )
    return Response(
        content=block,
        media_type=mime
    )

ipfs_api = FastAPI(
    lifespan=subapp_lifespan,
    title="IPFS API",
    description="A limited implementation of the Kubo IPFS RPC API"
)

@ipfs_api.post("/block/get")
def ipfs_block_get(
        arg: CID,
        output_codec = Query(
            default="dag-json",
            description="Output codec for the block, defaults to 'dag-json'."
        ),
        state: AppState = depend_appstate
    ):
    if (block := state.dag_get(arg, output_codec)) is None:
        return Response(
            status_code=404,
            content=f"Block with CID {arg} not found"
        )
    
    match output_codec:
        case "dag-cbor": mime = "application/cbor"
        case "dag-json": mime = "application/json"
        case _:
            raise NotImplementedError(f"Unknown output codec {output_codec}")
    
    return Response(
        content=block,
        media_type=mime
    )