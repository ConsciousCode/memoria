from functools import cached_property
import os
from typing import Any, Final, Iterable, Optional
import inspect

from mcp.types import ModelPreferences
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.models import Model
from pydantic_ai.providers import Provider

from memoria.config import SampleConfig
from memoria.util import warn

CHAT_SONA: Final = "chat"
TEMPERATURE: Final = 0.7

class ModelConfig(BaseModel):
    intelligence: float
    speed: float
    cost: float

class OllamaConfig(BaseModel):
    base_url: str
    api_key: Optional[str] = None

class OpenAIConfig(BaseModel):
    base_url: Optional[str] = None
    api_key: str

class OpenRouterConfig(BaseModel):
    api_key: str

class SimpleAPIProviderConfig(BaseModel):
    '''Model provider which only requires an API key.'''
    api_key: str

class ProviderConfig(BaseModel):
    anthropic: Optional[SimpleAPIProviderConfig] = None
    #azure: Optional[GenericProviderConfig] = None
    cohere: Optional[SimpleAPIProviderConfig] = None
    deepseek: Optional[SimpleAPIProviderConfig] = None
    #google: Optional[GenericProviderConfig] = None
    groq: Optional[SimpleAPIProviderConfig] = None
    mistral: Optional[SimpleAPIProviderConfig] = None
    openai: Optional[OpenAIConfig] = None
    openrouter: Optional[OpenRouterConfig] = None
    ollama: Optional[OllamaConfig] = None

    model_config = ConfigDict(extra='allow')

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
    annotate: SampleConfig = Field(default_factory=SampleConfig)
    '''Configuration for edge annotation sampling.'''

    models: dict[str, dict[str, ModelConfig]] = Field(default_factory=dict)
    '''Model profiles for different AI models.'''
    purposes: dict[str, str] = Field(default_factory=dict)
    '''Map purposes to model names.'''

    providers: ProviderConfig = Field(default_factory=ProviderConfig)
    '''AI model configuration.'''

    @cached_property
    def models_by_name(self):
        return {
            model: (provider, model)
            for provider, models in self.models.items()
                for model, profile in models.items()
        }
    
    def select_model(self, prefs: 'ModelPreferences') -> Iterable[tuple[str, str]]:
        '''
        Select the appropriate model for resolving a sampling. Yields models
        in order of their priority for fallback.
        '''

        # Check by purpose first, if any hints are provided
        for hint in prefs.hints or []:
            if purpose := getattr(hint, 'purpose', None):
                if model := self.purposes.get(purpose):
                    if which := self.models_by_name.get(model):
                        provider, model = which
                        if hasattr(self.providers, provider):
                            yield provider, model
                        else:
                            warn(f"Unknown provider {provider!r} for model {model!r}")
                    else:
                        warn(f"Unknown model {model!r} for purpose {purpose!r}")

        ## ENV model override ##
        if model := os.getenv("MODEL"):
            if ":" not in model:
                # Need to find the provider
                if which := self.models_by_name.get(model):
                    yield which
            
            # Hopefully it's a common name
            yield "", model

        unknown: list[tuple[str, str]] = []

        ## Check by name (preferring known providers) ##
        for p in prefs.hints or []:
            if p.name is None: continue
            if which := self.models_by_name.get(p.name):
                provider, model = which
                if hasattr(self.providers, provider):
                    yield provider, model
                else:
                    unknown.append(which)

        ## Check by priority ##
        intelligence = prefs.intelligencePriority or 0
        speed = prefs.speedPriority or 0
        cost = prefs.costPriority or 0

        # Index lets us sort by their order, disregaring model provider/name
        i = 0
        candidates: list[tuple[float, int, tuple[str, str]]] = []

        for provider, models in self.models.items():
            i -= 1
            for model, profile in models.items():
                score = (
                    + profile.intelligence * intelligence
                    + profile.speed * speed
                    - profile.cost * cost
                )
                candidates.append((score, i, (provider, model)))

        for _, _, which in sorted(candidates, reverse=True):
            yield which
            
        ## Check by name (unknown provider) ##
        yield from unknown
        
        ## Failure ##
        raise ValueError("No suitable model found in configuration.")

    def build_provider(self, provider: str) -> Optional[Provider]:
        '''Build the provider for the given name and API key.'''
        match provider:
            case "anthropic":
                if (pc := self.providers.anthropic) is None:
                    raise ValueError("Anthropic provider configuration is missing.")
                from pydantic_ai.providers.anthropic import AnthropicProvider
                return AnthropicProvider(api_key=pc.api_key)
            
            # Extra config, here for TODO
            #case "azure":
            #    from pydantic_ai.providers.azure import AzureProvider
            #    pk = AzureProvider
            #case "bedrock":
            #    from pydantic_ai.providers.bedrock import BedrockProvider
            #    pk = BedrockProvider
            
            case "cohere":
                if (pc := self.providers.cohere) is None:
                    raise ValueError("Cohere provider configuration is missing.")
                from pydantic_ai.providers.cohere import CohereProvider
                return CohereProvider(api_key=pc.api_key)
            
            case "deepseek":
                if (pc := self.providers.deepseek) is None:
                    raise ValueError("DeepSeek provider configuration is missing.")
                from pydantic_ai.providers.deepseek import DeepSeekProvider
                return DeepSeekProvider(api_key=pc.api_key)
            
            # Extra config, here for TODO
            #case "google":
            #    from pydantic_ai.providers.google import GoogleProvider
            #    pk = GoogleProvider
            #case "google_gla":
            #    from pydantic_ai.providers.google_gla import GoogleGLAProvider
            #    pk = GoogleGLAProvider
            #case "google_vertex":
            #    from pydantic_ai.providers.google_vertex import GoogleVertexProvider
            #    pk = GoogleVertexProvider
            
            case "groq":
                if (pc := self.providers.groq) is None:
                    raise ValueError("Groq provider configuration is missing.")
                from pydantic_ai.providers.groq import GroqProvider
                return GroqProvider(api_key=pc.api_key)
            
            case "ollama":
                if (pc := self.providers.ollama) is None:
                    raise ValueError("Ollama provider configuration is missing.")
                from pydantic_ai.providers.openai import OpenAIProvider
                return OpenAIProvider(api_key=pc.api_key, base_url=pc.base_url)

            case "openai":
                if (pc := self.providers.openai) is None:
                    raise ValueError("OpenAI provider configuration is missing.")
                from pydantic_ai.providers.openai import OpenAIProvider
                return OpenAIProvider(api_key=pc.api_key, base_url=pc.base_url)
            
            case "openrouter":
                if (pc := self.providers.openrouter) is None:
                    raise ValueError("OpenRouter provider configuration is missing.")
                from pydantic_ai.providers.openrouter import OpenRouterProvider
                return OpenRouterProvider(api_key=pc.api_key)
            
            case "mistral":
                if (pc := self.providers.mistral) is None:
                    raise ValueError("Mistral provider configuration is missing.")
                from pydantic_ai.providers.mistral import MistralProvider
                return MistralProvider(api_key=pc.api_key)

    def build_model(self, name: str, provider: Provider) -> Model:
        '''Build the model for the given name and provider.'''

        match provider.name:
            case "anthropic":
                from pydantic_ai.models.anthropic import AnthropicModel
                return AnthropicModel(name, provider=provider)
            
            #case "azure": pass
            
            case "cohere":
                from pydantic_ai.models.cohere import CohereModel
                return CohereModel(name, provider=provider)
            
            case "deepseek"|"openai":
                from pydantic_ai.models.openai import OpenAIModel
                return OpenAIModel(name, provider=provider)
            
            #case "google": pass
            #case "google_gla": pass
            #case "google_vertex": pass

            case "groq":
                from pydantic_ai.models.groq import GroqModel
                return GroqModel(name, provider=provider)
            
            case "openai": # ollama, openrouter
                from pydantic_ai.models.openai import OpenAIModel
                return OpenAIModel(name, provider=provider)

            case "mistral":
                from pydantic_ai.models.mistral import MistralModel
                return MistralModel(name, provider=provider)
            
            case _:
                raise ValueError(f"Unknown provider {provider.name!r} for model {name!r}")

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
            source = inspect.cleandoc(f'''
                ## Generated from defaults ##
                temperature = {TEMPERATURE}

                [models]
                # AI model profiles

                [providers]
                # AI model configuration
                #[providers.openai]
                #model = "gpt-4o"
                #api_key = "sk-###"
            ''')
            data = {}
        
        return Config(source=source, **data)