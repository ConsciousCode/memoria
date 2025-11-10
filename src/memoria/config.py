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

class SampleConfig(BaseModel):
    # See ModelSettings from pydantic for more to include
    '''Configuration for sampling responses.'''
    temperature: Annotated[
        float | None,
        Field(description="Sampling temperature for the response. If `null`, uses the default value.")
    ] = None
    max_tokens: Annotated[
        int | None,
        Field(description="Maximum number of tokens to generate in the response. If `null`, uses the default value.")
    ] = None
    model_preferences: Annotated[
        ModelPreferences | None,
        Field(description="List of preferred models to use for the response. If `null`, uses the default model.")
    ] = None
