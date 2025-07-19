from uuid import UUID
from pydantic import BaseModel

from ipld import CID

class SonaContext(BaseModel):
    """The sona within which to interpret."""
    uuid: UUID
    aliases: list[str]

class InterpretRequest(BaseModel):
    """Request model for the interpreter endpoint."""
    sona: SonaContext
    context: list[CID]