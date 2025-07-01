'''
Implement the IPFS Trustless Gateway specification and any IPFS-related utilities.
'''
import json
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, Query, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from ._common import AddParameters, AppState, subapp_lifespan, depend_appstate
from src.ipld import CID
from src.ipld.ipfs import CIDResolveError, dag_dump

ROOT = "private/blocks"

type SupportedCodec = Literal['raw', 'dag-json', 'dag-cbor', 'dag-pb']

def codec_mimetype(codec: str) -> str:
    """Return the MIME type for a given codec."""
    return {
        'dag-json': 'application/json',
        'dag-cbor': 'application/cbor',
        'dag-pb': 'application/vnd.ipld.dag-pb'
    }.get(codec, 'application/octet-stream')

############################
## IPFS Trustless Gateway ##
############################

ipfs_gateway = FastAPI(
    lifespan=subapp_lifespan,
    title="IPFS Trustless Gateway",
    description="A FastAPI implementation of the IPFS Trustless Gateway specification.",
    version="0.1.0"
)

@ipfs_gateway.head("/{path:path}")
def head_ipfs(
        path: str,
        accept: list[str] = Header(default_factory=list),
        state: AppState = depend_appstate
    ):
    """
    Handle HEAD requests for IPFS blocks.
    This endpoint checks if a block exists and returns the appropriate headers.
    """
    if accept is None:
        accept = []
    
    if "application/cbor" in accept or "application/vnd.ipld.dag-cbor" in accept:
        output_codec = "dag-cbor"
        mime = "application/cbor"
    else:
        output_codec = "dag-json"
        mime = "application/json"
    
    # Special case for the empty block
    if path == "bafkqaaa":
        return Response(
            headers={
                "Content-Length": "0",
                "Content-Type": mime
            }
        )
    if block := state.block_get(CID(path)):
        return Response(
            headers={
                "Content-Length": str(len(block)),
                "Content-Type": mime
            }
        )
    return Response(
        f"Block with CID {path} not found",
        status_code=404,
        headers={
            "Content-Length": "0",
            "Content-Type": mime
        }
    )

@ipfs_gateway.get("/{path:path}")
def get_ipfs(
        path: str,
        accept: list[str] = Header(default_factory=list),
        state: AppState = depend_appstate
    ):
    if "/" in path and "application/vnd.ipld.car" not in accept:
        return Response(
            "Pathing requires CAR format in Accept header",
            status_code=400
        )
    
    if "application/cbor" in accept or "application/vnd.ipld.dag-cbor" in accept:
        output_codec = "dag-cbor"
        mime = "application/cbor"
    else:
        output_codec = "dag-json"
        mime = "application/json"
    
    # Special case for the empty block
    if path == "bafkqaaa":
        return Response(b'', media_type=mime)
    
    if block := state.block_get(CID(path)):
        return Response(block, media_type=mime)
    return Response(f"{path} not found", status_code=404)

##############################
## Kubo-compatible IPFS API ##
##############################

ipfs_api = FastAPI(
    lifespan=subapp_lifespan,
    title="IPFS API",
    description="A limited implementation of the Kubo IPFS RPC API"
)

## Root commands ##

@ipfs_api.post("/add")
async def ipfs_add(
        request: Request,
        params: AddParameters = Depends(AddParameters),
        state: AppState = depend_appstate
    ):
    """
    Add files to IPFS - Kubo v0.35 compatible implementation
    """
    async def kubo_add_stream():
        try:
            # FastAPI deosn't seem to implement true asynchronous streaming, so
            # here we have to rely on request.form() which is blocking.
            form = await request.form()
            for name, data in form.multi_items():
                if isinstance(data, str):
                    yield json.dumps({
                        "Name": name,
                        "Error": f"Unexpected string data in field {name}"
                    })
                elif (mimetype := data.content_type) is None:
                    yield json.dumps({
                        "Name": data.filename,
                        "Error": f"Missing content type for file {data.filename}"
                    })
                else:
                    _, size, cid = state.upload_file(
                        data.file,
                        data.filename,
                        mimetype,
                        params.mtime
                    )
                    out = {
                        "Bytes": data.size,
                        "Hash": str(cid),
                        "Size": str(size)
                    }
                    if data.filename:
                        out['Name'] = data.filename
                    if params.mtime:
                        out['Mtime'] = params.mtime
                    yield json.dumps(out)
        except Exception as e:
            yield json.dumps({
                "Error": str(e)
            })
    
    # Stream response in Kubo format
    return StreamingResponse(
        kubo_add_stream(),
        media_type="application/json",
        headers={
            "X-Chunked-Output": "1",
            "Cache-Control": "no-cache"
        }
    )

@ipfs_api.get("/cat")
async def ipfs_cat(
        cid: CID,
        state: AppState = depend_appstate,
        offset: int = Query(
            0, description="Offset in bytes to start reading from the block."
        ),
        length: Optional[int] = Query(
            None, description="Length in bytes to read from the block."
        )
    ):
    """Get the content of a block by CID."""
    if state.blocksource.block_has(cid):
        return StreamingResponse(
            state.blocksource.ipfs_cat(cid, offset, length),
            media_type="application/octet-stream"
        )
    
    return Response(f"{cid} not found", status_code=404)



## Block commands ##

@ipfs_api.post("/block/get")
def ipfs_block_get(
        arg: CID,
        state: AppState = depend_appstate
    ):
    """
    Get a block by CID.
    This endpoint retrieves a block from the IPFS blockstore by its CID.
    The CID can be in any format supported by the IPFS blockstore.
    """
    if (block := state.block_get(arg)) is None:
        return Response(
            f"Block with CID {arg} not found", status_code=404
        )
    return Response(block, media_type=codec_mimetype(arg.codec))

@ipfs_api.post("/block/put")
async def ipfs_block_put(
        file: UploadFile,
        cid_codec: SupportedCodec = Query(
            default="raw",
            description="Multicodec to use in returned CID."
        ),
        mhtype: str = Query(
            default="sha2-256",
            description="Multihash hash function."
        ),
        state: AppState = depend_appstate
    ):
    """
    Add a block to the IPFS blockstore.
    This endpoint accepts a file upload and returns the CID of the added block.
    """
    if not file.filename:
        return Response("No file provided.", status_code=400)
    
    cid = state.blockstore.block_put(
        await file.read(), codec=cid_codec, function=mhtype
    )
    return Response(str(cid), media_type="text/plain")

## DAG commands ##

@ipfs_api.post("/dag/export")
def ipfs_dag_export(
        cid: CID,
        state: AppState = depend_appstate
    ):
    """Export a DAG starting from the given CID."""
    return StreamingResponse(state.blocksource.dag_export(cid))

@ipfs_api.post("/dag/get")
def ipfs_dag_get(
        cid: CID,
        state: AppState = depend_appstate,
        output_codec: Optional[SupportedCodec] = Query(
            None,
            description="Output codec for the DAG node. Defaults to 'dag-json'."
        )
    ):
    """Get a DAG node by CID."""
    # Use the CID's codec, no need to transform
    if output_codec is None or output_codec == cid.codec:
        if block := state.block_get(cid):
            return Response(
                block, media_type=codec_mimetype(cid.codec)
            )
        return Response(f"{cid} not found", status_code=404)
        
    try:
        return Response(
            dag_dump(output_codec, state.dag_get(cid)),
            media_type=codec_mimetype(output_codec)
        )
    except CIDResolveError:
        return Response(f"{cid} not found", status_code=404)

# /block/rm deliberately left out because we have no networking to back things up
# TODO: /block/stat, nice to have but requires plumbing
# TODO: /cid/* and /multibase/* for utilities, probably not useful
# TODO: /commands might be necessary for compatibility testing
# TODO: /config, /ping, /refs, /shutdown
# TODO: /repo/{gc?, ls, stat?, verify?}
# TODO: /refs, /refs/local