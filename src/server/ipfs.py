'''
Implement the IPFS Trustless Gateway specification and any IPFS-related utilities.
'''
import json
from typing import Literal, Optional
import traceback

from fastapi import Depends, FastAPI, HTTPException, Header, Query, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

import io
from multipart.multipart import MultipartParser, parse_options_header

from ._common import AddParameters, AppState, depend_appstate
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
    title="IPFS API",
    description="A limited implementation of the Kubo IPFS RPC API"
)

## Root commands ##
@ipfs_api.post("/add")
async def ipfs_add(
        request: Request,
        params: AddParameters = Depends(),
        state: AppState = depend_appstate
    ):
    """
    Add files to IPFS - Kubo v0.35 compatible implementation
    """
    ### Kubo endpoint ###
    # Read full multipart/form-data body then parse and stream events
    print(params)
    if (content_type := request.headers.get("content-type")) is None:
        raise HTTPException(
            status_code=400,
            detail="Missing Content-Type header"
        )
    ctype, pdict = parse_options_header(content_type)
    if ctype != b"multipart/form-data":
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported Content-Type; expected b'multipart/form-data', got {ctype}"
        )
    if (boundary := pdict.get(b"boundary")) is None:
        raise HTTPException(
            status_code=400,
            detail="Missing boundary in Content-Type header"
        )

    # Parser callbacks and state
    events: list[dict] = []
    current_headers: dict[bytes, bytes] = {}
    current_header_field: Optional[bytes] = None
    filename: Optional[str] = None
    mimetype: str = "application/octet-stream"
    buffer: Optional[io.BytesIO] = None

    def on_part_begin():
        nonlocal current_headers, current_header_field, filename, mimetype, buffer
        current_headers = {}
        current_header_field = None
        filename = None
        mimetype = "application/octet-stream"
        buffer = None

    def on_header_field(data: bytes, start: int, end: int):
        nonlocal current_header_field
        current_header_field = data[start:end].lower()

    def on_header_value(data: bytes, start: int, end: int):
        if current_header_field is not None:
            current_headers[current_header_field] = data[start:end]

    def on_headers_finished():
        nonlocal filename, mimetype, buffer
        if cd := current_headers.get(b"content-disposition"):
            _, cd_params = parse_options_header(cd)
            if (fname := cd_params.get(b"filename")) is not None:
                filename = fname.decode("utf-8", "ignore")
        if ctype_hdr := current_headers.get(b"content-type"):
            mimetype = ctype_hdr.decode("latin-1")
        buffer = io.BytesIO()

    def on_part_data(data: bytes, start: int, end: int):
        if buffer is not None:
            buffer.write(data[start:end])

    def on_part_end():
        if buffer is None:
            return
        buffer.seek(0)
        try:
            _, size, cid = state.upload_file(
                buffer, filename, mimetype, params
            )
            event = {
                "Bytes": buffer.tell(),
                "Hash": str(cid),
                "Size": str(size)
            }
            if filename: event["Name"] = filename
            if params.mtime: event["Mtime"] = params.mtime
        except Exception as exc:
            traceback.print_exc()
            event = {"Error": str(exc)}
        events.append(event)

    parser = MultipartParser(boundary, callbacks={
        "on_part_begin": on_part_begin,
        "on_header_field": on_header_field,
        "on_header_value": on_header_value,
        "on_headers_finished": on_headers_finished,
        "on_part_data": on_part_data,
        "on_part_end": on_part_end
    })

    try:
        parser.write(await request.body())
        parser.write(b"")
    except Exception as exc:
        events.append({"Error": str(exc)})

    # Yield all parser events
    return Response(
        content="\n".join(json.dumps(event) for event in events),
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