'''
Implement the IPFS Trustless Gateway specification and any IPFS-related utilities.
'''

from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import FastAPI, Header, Query, Request, Response

from ipld import dagjson, dagpb

from ._common import mcp_context
from src.ipld import dagcbor, CID
from src.ipld.ipfs import Blocksource, FlatfsBlockstore

ROOT = "private/blocks"

EMPTY_CID = CID("bafkqaaa")

@asynccontextmanager
async def ipfs_lifespan(app: FastAPI):
    yield {"blockstore": FlatfsBlockstore(ROOT)}

ipfs_gateway = FastAPI(
    lifespan=ipfs_lifespan,
    title="IPFS Trustless Gateway",
    description="A FastAPI implementation of the IPFS Trustless Gateway specification.",
    version="0.1.0"
)

def internal_ipfs_dag_get(
        request: Request,
        cid: CID,
        output_codec: str = "dag-json"
    ) -> Optional[str|bytes]:
    '''Internal function to retrieve a block from the IPFS DAG.'''
    memoria = mcp_context(request)
    bsource: Blocksource = request.app.state.blockstore

    if cid == 'bafkqaaa':
        return b''
    
    if ob := memoria.lookup_memory(cid):
        ob = ob.model_dump()
    elif ob := memoria.lookup_memory(cid):
        ob = ob.model_dump()
    elif ob := bsource.dag_get(cid):
        match cid.codec:
            case 'dag-pb':
                ob = dagpb.unmarshal(ob)
                ob = {
                    "Links": [
                        {
                            "Name": link.Name,
                            "Hash": str(CID(link.Hash)),
                            "Size": link.ByteSize()
                        } for link in ob.Links
                    ],
                    "Data": ob.Data
                }
            case 'dag-cbor': ob = dagcbor.unmarshal(ob)
            case 'dag-json': ob = dagjson.unmarshal(ob)
            case _:
                return ob
    else:
        return None
    
    match output_codec:
        case "dag-cbor": return dagcbor.marshal(ob)
        case "dag-json": return dagjson.marshal(ob)
        case _: raise NotImplementedError()

@ipfs_gateway.get(f"/{EMPTY_CID}")
def get_ipfs_empty():
    '''Empty block for PING'''
    return Response()

@ipfs_gateway.get("/{path:path}")
def get_ipfs(
        request: Request,
        path: str,
        accept: Annotated[Optional[list[str]], Header()] = None
    ):
    if accept is None:
        accept = []
    
    if "application/cbor" in accept:
        output_codec = "dag-cbor"
        mime = "application/cbor"
    else:
        output_codec = "dag-json"
        mime = "application/json"
    
    block = internal_ipfs_dag_get(request, CID(path), output_codec)
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
    lifespan=ipfs_lifespan,
    title="IPFS API",
    description="A limited implementation of the Kubo IPFS RPC API"
)

@ipfs_api.post("/block/get")
def ipfs_block_get(
        request: Request,
        arg: CID,
        output_codec = Query(
            default="dag-json",
            description="Output codec for the block, defaults to 'dag-json'."
        )
    ):

    block = internal_ipfs_dag_get(request, arg, output_codec)
    if block is None:
        return Response(
            status_code=404,
            content=f"Block with CID {arg} not found"
        )
    return Response(
        content=block,
        media_type=f"application/{output_codec.split('-')[1]}"
    )