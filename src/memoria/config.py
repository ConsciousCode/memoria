'''
General config within the Memoria system.
'''

from typing import Annotated

from mcp.types import ModelPreferences
from pydantic import BaseModel, Field

class RecallConfig(BaseModel):
    '''Configuration for memory recall.'''
    
    refs: Annotated[
        int | None,
        Field(description="Maximum depth of memories after roots.")
    ] = 5
    deps: Annotated[
        int | None,
        Field(description="Maximum depth of memories before roots.")
    ] = 5
    memories: Annotated[
        int,
        Field(description="Maximum number of memories.")
    ] = 5