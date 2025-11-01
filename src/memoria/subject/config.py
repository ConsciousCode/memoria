import os
from typing import Annotated, Any, Final, Literal, Optional
import inspect

from pydantic import BaseModel, Field

from memoria.config import RecallConfig, SampleConfig

TEMPERATURE: Final = 0.7

class ModelConfig(BaseModel):
    intelligence: float
    speed: float
    cost: float

class IPFSConfig(BaseModel):
    cid_version: Annotated[
        Literal[0, 1], Field(description="CID version.")
    ] = 1 # We use raw-leaves by default, so CIDv1 is preferred.
    hash: Annotated[
        str, Field(description="Hash function.")
    ] = "sha2-256"

class Config(BaseModel):
    server: str = ""
    '''MCP server URL to connect to.'''
    temperature: float | None = None
    '''Default temperature for chat responses.'''
    
    chat: SampleConfig = Field(default_factory=SampleConfig)
    '''Configuration for chat sampling.'''

    ipfs: IPFSConfig = IPFSConfig()
    recall: RecallConfig = RecallConfig()
    '''Configuration for how to weight memory recall.'''

    @classmethod
    def from_file(cls, path: str) -> 'Config':
        """Load a Memoria client configuration from TOML."""
        import tomllib
        try:
            with open(os.path.expanduser(path), 'r') as f:
                if source := f.read():
                    data = Config.model_validate(tomllib.loads(source))
                else:
                    # Piped to file we're reading from
                    raise FileNotFoundError(f"Empty config file: {path}")
        except FileNotFoundError:
            source = inspect.cleandoc(f'''
                ## Generated from defaults ##
                [recall]
                # Default weighting for recall
                #importance = null
                #recency = null
                #fts = null
                #vss = null
                #k = null
            ''')
            data = Config()
        
        return data
