'''
Simple RESTful API separate from the MCP API.
'''
from typing import Annotated, Optional
from uuid import UUID

from fastapi import Depends, FastAPI, Header, Query, Response, UploadFile
from fastapi.responses import JSONResponse

from ._common import AddParameters, AppState, depend_appstate, depend_repo
from ..ipld import dagcbor, CIDv1
from ..repo import Repository

rest_api = FastAPI(
    title="Memoria REST API",
    description="A RESTful API for the Memoria system."
)

@rest_api.get("/memory/{cid}")
def get_memory(
        cid: CIDv1,
        accept: Annotated[Optional[list[str]], Header()] = None,
        repo: Repository = depend_repo,
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
        accept: Annotated[Optional[list[str]], Header()] = None,
        repo: Repository = depend_repo
    ):
    '''List messages in the Memoria system.'''
    messages = repo.list_messages(page, perpage)
    accept = accept or []
    if "application/cbor" in accept:
        return dagcbor.marshal(messages)
    return messages

@rest_api.get("/sona/{uuid}")
def get_sona(
        uuid: UUID|str,
        accept: Annotated[Optional[list[str]], Header()] = None,
        repo: Repository = depend_repo,
    ):
    '''Get a sona by UUID.'''
    try: uuid = UUID(uuid) # type: ignore
    except Exception:
        pass

    if sona := repo.find_sona(uuid):
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
        accept: Annotated[
            list[str], Header(default_factory=list)
        ],
        page: Annotated[
            int, Query(description="Page number to return.")
        ] = 1,
        perpage: Annotated[
            int, Query(description="Number of sonas to return per page.")
        ] = 100,
        repo: Repository = depend_repo
    ):
    '''List sonas in the Memoria system.'''
    sonas = repo.list_sonas(page, perpage)
    accept = accept or []
    if "application/cbor" in accept:
        return dagcbor.marshal(sonas)
    return [sona.human_json() for sona in sonas]

@rest_api.get("/sona/{uuid}/act/next")
async def act_next(
        uuid: UUID|str,
        repo: Repository = depend_repo,
    ):
    '''Advance the ACT for a sona by UUID.'''
    try: uuid = UUID(uuid) # type: ignore
    except Exception:
        return Response(
            status_code=400,
            content=f"Invalid UUID {uuid}"
        )

    match repo.act_next(uuid):
        case False:
            return Response(
                status_code=404,
                content=f"Sona with UUID {uuid} not found."
            )
        case True: return JSONResponse(None)
        case g: return JSONResponse(g.adj)

@rest_api.post("/file")
async def upload_file(
        file: UploadFile,
        params: AddParameters = Depends(),
        state: AppState = depend_appstate
    ):
    '''Upload a file to the Memoria system.'''
    if file.content_type is None:
        raise ValueError("Content-Type header is required")
    
    fstream = file.file
    created, size, cid = state.upload_file(
        fstream,
        file.filename,
        file.content_type,
        params
    )
    
    return Response(
        status_code=201 if created else 200,
        content=cid,
        media_type="text/plain"
    )