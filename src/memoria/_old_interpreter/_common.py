from uuid import UUID
from pydantic import BaseModel

from cid import CIDv1

class SonaContext(BaseModel):
    """The sona within which to interpret."""
    uuid: UUID
    aliases: list[str]

class InterpretRequest(BaseModel):
    """Request model for the interpreter endpoint."""
    sona: SonaContext
    context: list[CIDv1]