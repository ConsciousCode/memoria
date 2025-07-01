'''
Simple RESTful API separate from the MCP API.
'''
from typing import Annotated, Optional
from uuid import UUID

from fastapi import FastAPI, Header, Query, Request, Response, UploadFile

from ipld.cid import CIDv1

from ._common import AppState, subapp_lifespan
from src.ipld import dagcbor

rest_api = FastAPI(
    title="Memoria REST API",
    description="A RESTful API for the Memoria system.",
    lifespan=subapp_lifespan
)

@rest_api.get("/memory/{cid}")
def get_memory(
        request: Request,
        cid: CIDv1,
        accept: Annotated[Optional[list[str]], Header()] = None
    ):
    '''Get a memory by CID.'''
    state: AppState = request.app.state
    if memory := state.memoria.lookup_memory(cid):
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
        request: Request,
        page: Annotated[
            int, Query(description="Page number to return.")
        ] = 1,
        perpage: Annotated[
            int, Query(description="Number of messages to return per page.")
        ] = 100,
        accept: Annotated[Optional[list[str]], Header()] = None
    ):
    '''List messages in the Memoria system.'''
    state: AppState = request.app.state
    messages = state.memoria.list_messages(page, perpage)
    accept = accept or []
    if "application/cbor" in accept:
        return dagcbor.marshal(messages)
    return messages

@rest_api.get("/sona/{uuid}")
def get_sona(
        request: Request,
        uuid: UUID|str,
        accept: Annotated[Optional[list[str]], Header()] = None
    ):
    '''Get a sona by UUID.'''
    try: uuid = UUID(uuid) # type: ignore
    except Exception:
        pass

    state: AppState = request.app.state
    if sona := state.memoria.find_sona(uuid):
        accept = accept or []
        if "application/cbor" in accept:
            return dagcbor.marshal(sona)
        return sona.human_json()
    return Response(
        status_code=404,
        content=f"Sona with UUID {uuid} not found."
    )

@rest_api.get("/sonas")
def list_sonas(
        request: Request,
        accept: Annotated[
            list[str], Header(default_factory=list)
        ],
        page: Annotated[
            int, Query(description="Page number to return.")
        ] = 1,
        perpage: Annotated[
            int, Query(description="Number of sonas to return per page.")
        ] = 100
    ):
    '''List sonas in the Memoria system.'''
    state: AppState = request.app.state
    sonas = state.memoria.list_sonas(page, perpage)
    accept = accept or []
    if "application/cbor" in accept:
        return dagcbor.marshal(sonas)
    return [sona.human_json() for sona in sonas]

@rest_api.post("/file")
async def upload_file(
        request: Request,
        content_type: str = Header(),
        ctime: Optional[int] = Query(
            None,
            description="Creation time of the file in seconds since epoch."
        )
    ):
    '''Upload a file to the Memoria system.'''
    if not content_type.startswith("multipart/form-data"):
        return Response(
            status_code=415,
            content="Content-Type must be multipart/form-data."
        )
    
    form = await request.form()
    for name, value in form.multi_items():
        if isinstance(value, str):
            value.filename
    
    state: AppState = request.app.state
    fstream = file.file
    created, cid = state.upload_file(
        fstream,
        file.filename,
        file.content_type,
        file.size or fstream.tell(),
        ctime
    )
    
    return Response(
        status_code=201 if created else 200,
        content=cid,
        media_type="text/plain"
    )