from typing import Annotated, Optional

from fastapi import FastAPI, Header, Request, Response

from ipld import dagcbor, CIDv1
from ._common import mcp_context

ipfs_app = FastAPI()

@ipfs_app.get("/bafkqaaa")
def get_ipfs_empty():
    '''Empty block for PING'''
    return Response()

@ipfs_app.get("/{path:path}")
def get_ipfs(
        path: str,
        request: Request,
        accept: Annotated[Optional[list[str]], Header()] = None
    ):
    if accept is None:
        accept = []
    cid = CIDv1(path)
    memoria = mcp_context(request)
    if ob := memoria.lookup_memory(cid):
        if "application/cbor" in accept:
            return dagcbor.marshal(ob)
        return ob
    if ob := memoria.lookup_act(cid):
        if "application/cbor" in accept:
            return dagcbor.marshal(ob)
        return ob
    # ipfs_api.lookup(cid) # TODO
    return Response(
        status_code=404,
        content=f"Memory or ACT not found for CID {cid}"
    )