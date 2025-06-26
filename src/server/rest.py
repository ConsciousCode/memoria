'''
Simple RESTful API separate from the MCP API.
'''
from typing import Annotated, Optional
from uuid import UUID

from fastapi import FastAPI, Header, Query, Request, Response

from ._common import mcp_context
from src.ipld import dagcbor

rest_app = FastAPI()

@rest_app.get("/memories/list")
def list_memories(
        request: Request,
        page: Annotated[
            int,
            Query(description="Page number to return.")
        ] = 1,
        perpage: Annotated[
            int,
            Query(description="Number of messages to return per page.")
        ] = 100,
        accept: Annotated[Optional[list[str]], Header()] = None
    ):
    '''List messages in the Memoria system.'''
    memoria = mcp_context(request)
    messages = memoria.list_messages(page, perpage)
    accept = accept or []
    if "application/cbor" in accept:
        return dagcbor.marshal(messages)
    return messages

@rest_app.get("/sona/{uuid}")
def get_sona(
        request: Request,
        uuid: UUID|str,
        accept: Annotated[Optional[list[str]], Header()] = None
    ):
    '''Get a sona by UUID.'''
    try: uuid = UUID(uuid) # type: ignore
    except (ValueError, TypeError):
        pass

    memoria = mcp_context(request)
    if sona := memoria.find_sona(uuid):
        accept = accept or []
        if "application/cbor" in accept:
            return dagcbor.marshal(sona)
        return sona.human_json()
    return Response(
        status_code=404,
        content=f"Sona with UUID {uuid} not found."
    )

@rest_app.get("/sonas/list")
def list_sonas(
        request: Request,
        page: Annotated[
            int,
            Query(description="Page number to return.")
        ] = 1,
        perpage: Annotated[
            int,
            Query(description="Number of sonas to return per page.")
        ] = 100,
        accept: Annotated[Optional[list[str]], Header()] = None
    ):
    '''List sonas in the Memoria system.'''
    memoria = mcp_context(request)
    sonas = memoria.list_sonas(page, perpage)
    accept = accept or []
    if "application/cbor" in accept:
        return dagcbor.marshal(sonas)
    return [sona.human_json() for sona in sonas]
