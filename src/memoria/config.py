'''
General config within the Memoria system.
'''

from typing import Annotated, Optional

from mcp.types import ModelPreferences
from pydantic import BaseModel, Field

class RecallConfig(BaseModel):
    '''Configuration for how to weight memory recall.'''
    recency: Annotated[
        float,
        Field(description="Weight of the recency of the memory.")
    ] = 0.30
    sona: Annotated[
        float,
        Field(description="Weight of the sona relevance.")
    ] = 0.10
    fts: Annotated[
        float,
        Field(description="Weight of the full-text search relevance.")
    ] = 0.15
    vss: Annotated[
        float,
        Field(description="Weight of the vector similarity.")
    ] = 0.25
    k: Annotated[
        int,
        Field(description="Number of memories to return. 20 by default.")
    ] = 20
    decay: Annotated[
        Optional[float],
        Field(description="Time decay factor for recency. 0.995 by default.")
    ] = 0.995

class SampleConfig(BaseModel):
    # See ModelSettings from pydantic for more to include
    '''Configuration for sampling responses.'''
    temperature: Annotated[
        Optional[float],
        Field(description="Sampling temperature for the response. If `null`, uses the default value.")
    ] = None
    max_tokens: Annotated[
        Optional[int],
        Field(description="Maximum number of tokens to generate in the response. If `null`, uses the default value.")
    ] = None
    model_preferences: Annotated[
        Optional[ModelPreferences],
        Field(description="List of preferred models to use for the response. If `null`, uses the default model.")
    ] = None