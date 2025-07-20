import os
from typing import Annotated, Any, Final, Literal, Optional
import inspect

from pydantic import BaseModel, Field

from memoria.config import RecallConfig, SampleConfig

CHAT_SONA: Final = "chat"
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
    source: str
    '''Original source of the config file.'''

    server: str
    '''MCP server URL to connect to.'''
    sona: Optional[str] = None
    '''Default sona to use for chat.'''
    temperature: Optional[float] = None
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
                    data: dict[str, Any] = tomllib.loads(source)
                else:
                    # Piped to file we're reading from
                    raise FileNotFoundError(f"Empty config file: {path}")
        except FileNotFoundError:
            import json
            source = inspect.cleandoc(f'''
                ## Generated from defaults ##
                sona = {json.dumps(CHAT_SONA)}
                
                [recall]
                # Default weighting for recall
                #importance = null
                #recency = null
                #sona = null
                #fts = null
                #vss = null
                #k = null
            ''')
            data = {}
        
        return Config(source=source, **data)