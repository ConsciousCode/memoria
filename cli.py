#!/usr/bin/env python3

from contextlib import asynccontextmanager
from datetime import datetime
from functools import cached_property
import os
from typing import TYPE_CHECKING, Annotated, Any, Final, Iterable, Iterator, Literal, NoReturn, Optional, Sequence, cast
import inspect
from uuid import UUID
import mimetypes

from mcp.types import ModelPreferences, TextContent
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.models import Model
from pydantic_ai.providers import Provider
from tqdm import tqdm

from client import MemoriaClient
from server._common_server import AddParameters
from src.ipld import CIDv1
from src.models import AnyMemory, AnyMemoryData, Edge, FileData, ImportAdapter, ImportFileData, IncompleteMemory, ImportMemory, MetaData, NodeMemory, PartialMemory, RecallConfig, SampleConfig, TextData

if TYPE_CHECKING:
    # These can be pretty beefy, so put in a type check to avoid
    # importing them unless needed.
    from fastmcp.client.sampling import SamplingParams
    from fastmcp.utilities.types import MCPContent
    from mcp import SamplingMessage

SERVER: Final = "http://127.0.0.1:8000/mcp"
CONFIG: Final = "./private/memoria.toml"
CHAT_SONA: Final = "chat"
TEMPERATURE: Final = 0.7

SYSTEM_PROMPT = """You are a helpful AI assistant. You can answer questions, provide information, and assist with tasks."""

class CmdError(Exception):
    pass

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
    annotate: SampleConfig = Field(default_factory=SampleConfig)
    '''Configuration for edge annotation sampling.'''

    models: dict[str, dict[str, ModelConfig]] = Field(default_factory=dict)
    '''Model profiles for different AI models.'''
    purposes: dict[str, str] = Field(default_factory=dict)
    '''Map purposes to model names.'''

    ipfs: IPFSConfig = IPFSConfig()
    recall: RecallConfig = RecallConfig()
    '''Configuration for how to weight memory recall.'''
    providers: ProviderConfig = Field(default_factory=ProviderConfig)
    '''AI model configuration.'''

    @cached_property
    def models_by_name(self):
        return {
            model: (provider, model)
            for provider, models in self.models.items()
                for model, profile in models.items()
        }

def unpack[*A](args: Iterable[str], *defaults: *A) -> tuple[str, ...]|tuple[*A]:
    '''Unpack rest arguments with defaults and proper typing.'''
    return (*args, *defaults)[:len(defaults)] # type: ignore

def expected(name: str) -> NoReturn:
    raise ValueError(f"Expected a {name}.")

def warn(msg):
    print(f"Warning: {msg}", file=sys.stderr)

def check_overflow(rest):
    if rest: warn("Too many arguments.")

def parse_opts(arg: str) -> Iterator[tuple[str, Optional[str]]]:
    """Parse command line options from a string."""
    if arg.startswith("--"):
        try:
            eq = arg.index('=')
            yield arg[2:eq], arg[eq+1:]
        except ValueError:
            yield arg[2:], None
    else:
        # short options
        for i in range(1, len(arg)):
            yield arg[i], None

def named_value(arg: str, it: Iterator[str]) -> str:
    try: return next(it)
    except StopIteration:
        expected(f"value after {arg}")

def argparse(argv: tuple[str, ...], config: dict[str, bool|int|type[int]|str|type[str]]):
    which = {}
    for aliases, v in config.items():
        als = aliases.split(',')
        match [a for a in als if a.startswith("--")]:
            case []: name = None
            case [name]: name = name.removeprefix('--')
            case long: raise ValueError(f"Multiple long options found: {long}")

        for k in als:
            k = k.lstrip('-')
            which[k] = v, name or k
    
    pos = []
    opts = {
        name: t
        for t, name in which.values()
            if isinstance(t, (bool, int, str))
    }

    it = iter(argv)
    try:
        while True:
            arg = next(it)
            if not arg.startswith("-"):
                pos.append(arg)
                continue
            
            if arg == "--":
                pos.extend(it)
                break
            
            for opt, val in parse_opts(arg):
                if (c := which.get(opt)) is None:
                    raise ValueError(f"Unknown option {opt!r}")
                
                t, name = c

                if isinstance(t, bool) or t is bool:
                    if val is not None:
                        raise ValueError(f"Option {opt!r} does not take a value")
                    opts[name] = not t
                elif isinstance(t, int) or t is int:
                    if val is None:
                        val = named_value(arg, it)
                    try: opts[name] = int(val)
                    except ValueError:
                        raise ValueError(f"Expected an integer after {arg!r}") from None
                elif isinstance(t, str) or t is str:
                    if val is None:
                        val = named_value(arg, it)
                    opts[name] = val
                else:
                    raise TypeError(f"Unsupported type {t} for option {arg!r}")
    except StopIteration:
        pass
    
    return pos, opts

class ChatModel(BaseModel):
    cid: Optional[CIDv1]
    chatlog: list['MCPContent']
    response: 'MCPContent'

def select_model(config: Config, prefs: 'ModelPreferences') -> Iterable[tuple[str, str]]:
    '''
    Select the appropriate model for resolving a sampling. Yields models
    in order of their priority for fallback.
    '''

    # Check by purpose first, if any hints are provided
    for hint in prefs.hints or []:
        if purpose := getattr(hint, 'purpose', None):
            if model := config.purposes.get(purpose):
                if which := config.models_by_name.get(model):
                    provider, model = which
                    if hasattr(config.providers, provider):
                        yield provider, model
                    else:
                        warn(f"Unknown provider {provider!r} for model {model!r}")
                else:
                    warn(f"Unknown model {model!r} for purpose {purpose!r}")

    ## ENV model override ##
    if model := os.getenv("MODEL"):
        if ":" not in model:
            # Need to find the provider
            if which := config.models_by_name.get(model):
                yield which
        
        # Hopefully it's a common name
        yield "", model

    unknown: list[tuple[str, str]] = []

    ## Check by name (preferring known providers) ##
    for p in prefs.hints or []:
        if p.name is None: continue
        if which := config.models_by_name.get(p.name):
            provider, model = which
            if hasattr(config.providers, provider):
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

    for provider, models in config.models.items():
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

def build_provider(provider: str, config: Config) -> Optional[Provider]:
    '''Build the provider for the given name and API key.'''
    match provider:
        case "anthropic":
            if (pc := config.providers.anthropic) is None:
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
            if (pc := config.providers.cohere) is None:
                raise ValueError("Cohere provider configuration is missing.")
            from pydantic_ai.providers.cohere import CohereProvider
            return CohereProvider(api_key=pc.api_key)
        
        case "deepseek":
            if (pc := config.providers.deepseek) is None:
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
            if (pc := config.providers.groq) is None:
                raise ValueError("Groq provider configuration is missing.")
            from pydantic_ai.providers.groq import GroqProvider
            return GroqProvider(api_key=pc.api_key)
        
        case "ollama":
            if (pc := config.providers.ollama) is None:
                raise ValueError("Ollama provider configuration is missing.")
            from pydantic_ai.providers.openai import OpenAIProvider
            return OpenAIProvider(api_key=pc.api_key, base_url=pc.base_url)

        case "openai":
            if (pc := config.providers.openai) is None:
                raise ValueError("OpenAI provider configuration is missing.")
            from pydantic_ai.providers.openai import OpenAIProvider
            return OpenAIProvider(api_key=pc.api_key, base_url=pc.base_url)
        
        case "openrouter":
            if (pc := config.providers.openrouter) is None:
                raise ValueError("OpenRouter provider configuration is missing.")
            from pydantic_ai.providers.openrouter import OpenRouterProvider
            return OpenRouterProvider(api_key=pc.api_key)
        
        case "mistral":
            if (pc := config.providers.mistral) is None:
                raise ValueError("Mistral provider configuration is missing.")
            from pydantic_ai.providers.mistral import MistralProvider
            return MistralProvider(api_key=pc.api_key)

def build_model(name: str, provider: Provider, config: Config) -> Model:
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

class MemoriaApp:
    def __init__(self, *rest, help=None, config=None):
        self.rest = rest
        self.help = help
        self.config = config or CONFIG
    
    def tagname(self, memory: AnyMemory):
        match memory.data.kind:
            case "self": return "assistant"
            case "other": return "user"
            case kind: return kind

    def print_memory(self, memory: AnyMemory, refs: dict[CIDv1, int], verbose=True, extra=True):
        """Print a message to the user."""
        if extra:
            parts = []
            if cid := memory.cid:
                ref = refs[cid]
            else:
                ref = ""
            
            if es := memory.edges:
                rs = ','.join(str(refs[e.target]) for e in es)
                parts.append(f"ref={ref}->{rs}")
            else:
                parts.append(f"ref={ref}")
            
            if ts := memory.timestamp:
                dt = datetime.fromtimestamp(ts).replace(microsecond=0)
                parts.append(f"{dt.isoformat()}")
            
            after = ' ' + ' '.join(parts) if verbose and parts else ""
            print(f"<{self.tagname(memory)}{after}> ", end='')
        print(memory.data.document())
    
    def print_chatlog(self, log: Sequence[AnyMemory], refs: dict[CIDv1, int], meta=False, verbose=True, extra=True):
        """
        Print the chat log in a readable format.
        
        Parameters:
            chatlog: The Chatlog object to print.
            meta: If True, print metadata about the chat log.
        """
        for m in log:
            self.print_memory(m, refs, verbose=verbose, extra=extra)
    
    @classmethod
    def argparse(cls, *argv: str):
        '''Build the app from command line arguments.'''
        
        try:
            opts = {}
            it = iter(argv)
            while True:
                match arg := next(it):
                    case "-h"|"--help":
                        try:
                            opts['help'] = next(it)
                            check_overflow(list(it))
                        except StopIteration:
                            opts['help'] = ''
                        break
                    
                    case '-c'|'--config':
                        opts['config'] = named_value(arg, it)
                    
                    case _:
                        break
            
            return cls(arg, *it, **opts)
        except StopIteration:
            return None

    def run(self):
        if self.help is not None:
            return print(self.usage(self.help))
        
        if not self.rest:
            expected("subcommand")
        
        name, *rest = self.rest

        if subcmd := getattr(self, f"subcmd_{name}", None):
            co = subcmd(*self.rest[1:])
        else:
            co = self.subcmd_chat(*rest)
        
        if inspect.iscoroutine(co):
            import asyncio
            return asyncio.run(co)
        return co

    def usage(self, what: str=""):
        """
        usage: python cli.py cmd ...
        
        Commands:
          import [sona] [file]  Load messages exported from an AI provider.
          query [query]         Talk to the agent without creating memories.
          chat [message]        Single-turn conversation with the agent. (default)
          recall [search]       Recall a chat log from the agent's memory.
          help                  Show this help message.
        """
        
        if what == "":
            doc = inspect.cleandoc(self.usage.__doc__ or "")
        elif sub := getattr(self, f"subcmd_{what}", None):
            doc = inspect.cleandoc(sub.__doc__ or "")
            doc = "usage: {name} " + f"{what} {doc}"
        else:
            doc = inspect.cleandoc(self.usage.__doc__ or "")
            doc = f"Unknown subcommand {what!r}\n\n{doc}"
        
        return doc.format(name=what)
    
    def get_config(self) -> Config:
        import tomllib
        try:
            with open(os.path.expanduser(self.config), 'r') as f:
                if source := f.read():
                    data: dict[str, Any] = tomllib.loads(source)
                else:
                    # Piped to file we're reading from
                    raise FileNotFoundError(f"Empty config file: {self.config}")
        except FileNotFoundError:
            import json
            source = inspect.cleandoc(f'''
                ## Generated from defaults ##
                sona = {json.dumps(CHAT_SONA)}
                temperature = {TEMPERATURE}

                [recall]
                # Default weighting for recall
                #importance = null
                #recency = null
                #sona = null
                #fts = null
                #vss = null
                #k = null

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
    
    async def sampling_handler(self, msgs: list['SamplingMessage'], params: 'SamplingParams', ctx):
        '''Used to implement MCP sampling.'''
        try:
            from pydantic_ai import Agent
            from pydantic_ai.settings import ModelSettings
            from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
            from pydantic_ai.providers.openai import OpenAIProvider
            from mcp.types import TextContent
            from mcp import CreateMessageResult

            config = self.get_config()
            prefs = params.modelPreferences or ModelPreferences()
            for prov, name in select_model(config, prefs):
                settings: ModelSettings = {}
                if v := params.maxTokens:
                    if v < 4096:
                        settings["max_tokens"] = v
                if v := params.stopSequences:
                    settings["stop_sequences"] = v
                
                if v := params.temperature:
                    settings["temperature"] = v
                elif v := config.temperature:
                    settings["temperature"] = v
                
                if provider := build_provider(prov, config):
                    if isinstance(provider, OpenAIProvider) and name.startswith(("o", "gpt-4.1")):
                        # Some models don't support temperature
                        settings['temperature'] = 1
                    model = build_model(name, provider, config)
                else:
                    model = name

                print("Model", model if isinstance(model, str) else model.model_name)
                print(msgs)
                agent = Agent(
                    model, instructions=params.systemPrompt or SYSTEM_PROMPT
                )

                history: list[ModelRequest | ModelResponse] = []
                for m in msgs:
                    assert isinstance(m.content, TextContent), "Only TextContent is supported"
                    content = m.content.text

                    if m.role == "user":
                        history.append(ModelRequest.user_text_prompt(content))
                    elif m.role == "assistant":
                        history.append(ModelResponse(parts=[TextPart(content=content)]))
                    else:
                        raise ValueError(f"Unknown role {m.role!r} in message {m!r}")

                print(history)
                result = await agent.run(
                    message_history=history,
                    model_settings=settings
                )

                return CreateMessageResult(
                    role="assistant",
                    content=TextContent(type="text", text=result.output),
                    model=f"{prov}:{name}",
                    stopReason="endTurn"
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            #print("Hello error")
            raise
        
        raise ValueError(
            "No suitable model found in configuration for sampling handler."
        )
    
    @asynccontextmanager
    async def mcp(self):
        from fastmcp.client.client import Client
        from fastmcp.client.transports import StreamableHttpTransport

        config = self.get_config()

        async with Client(
            StreamableHttpTransport(config.server),
            sampling_handler=self.sampling_handler,
        ) as client:
            yield MemoriaClient(client)
    
    async def insert_memory(self,
            config: Config,
            sona: Optional[UUID|str],
            metadata_cid: Optional[CIDv1],
            prev: Optional[CIDv1],
            client: MemoriaClient,
            memory: NodeMemory[AnyMemoryData]
        ) -> CIDv1:
        if sona:
            if memory.sonas is None:
                memory.sonas = []
            memory.sonas.append(sona)
        
        if metadata_cid:
            memory.edges.append(Edge(target=metadata_cid, weight=0.0))
        
        if prev:
            memory.edges.append(Edge(target=prev, weight=1.0))
        
        if isinstance(memory.data, ImportFileData):
            with open(memory.data.file, "rb") as f:
                data = f.read()
            if (filename := memory.data.filename):
                filename = os.path.basename(filename)
            else:
                filename = memory.data.file
            
            ipfs = self.get_config().ipfs
            params = AddParameters(
                cid_version=ipfs.cid_version,
                hash=ipfs.hash
            )
            mimetype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            uploaded = await client.upload(
                data,
                filename,
                mimetype,
                params
            )
            # Substitute for a normal file memory
            memory = IncompleteMemory(
                data=FileData(
                    file=uploaded.cid,
                    filename=filename,
                    mimetype=mimetype,
                    filesize=len(data)
                ),
                timestamp=memory.timestamp,
                importance=memory.importance,
                edges=memory.edges,
                sonas=memory.sonas
            )

        complete = await client.insert(memory, config.recall, config.annotate)
        return complete.cid

    async def subcmd_import(self, *argv: str):
        '''
        [-s sona] [file]

        Insert a message into the agent's memory.

        parameters:
          file           File containing the messages to insert, or STDIN. Follows this schema:

        \x1b[2m```
        type SelfData = {{
            kind: "self",
            name?: str?,
            parts: [{{content: str, model?: str?}}]
            stop_reason?: ("endTurn" | "stopSequence" | "maxTokens")?
        }}
        type OtherData = {{
            kind: "other",
            name?: str?,
            content: str
        }}
        type TextData = {{
            kind: "text",
            content: str
        }}
        type FileData = {{
            kind: "file",
            file: CID,
            filename?: str?,
            mimetype: str,
            filesize: int
        }}
        type MetaData = {{
            kind: "metadata",
            metadata: {{[str]: any}}
        }}
        type ImportData = {{
            kind: "import",
            file: {{path: path}},
            filename?: str?,
            mimetype?: str?,
            filesize?: int?
        }}

        type Memory = {{
            data: SelfData | OtherData | TextData | FileData | MetaData,
            timestamp: float | iso8601,
            importance?: float?,
            edges?: [{{target: CIDv1 | Memory, weight: float}}],
            sonas?: [UUID | str]?
        }}
        type Convo = {{
            sona?: (UUID | str)?,
            metadata?: {{
                timestamp: float | iso8601,
                provider: str,
                uuid: UUID,
                title: str,
                importance?: float?
            }},
            prev?: CIDv1 | Memory,
            chatlog: [Memory]
        }}

        type Import = Memory | Convo | [Memory | Convo]
        ```\x1b[m

        options:
          -s,--sona       The sona to insert the message into. If not specified, the
                           default chat sona is used.
          -v,--verbose    Show the total context as seen by the agent.
        '''

        args, opts = argparse(argv, {
            "-s,--sona": str,
            "-v,--verbose": False
        })
        
        if len(args) > 1:
            raise ValueError("Expected a single message argument.")
        
        if args and args[0] != "-":
            with open(args[0], "r") as f:
                data = f.read()
        else:
            data = input()
        
        sona = cast(Optional[UUID|str], opts.get('sona'))
        parts = ImportAdapter.validate_json(data)
        if not isinstance(parts, list):
            parts = [parts]

        config = self.get_config()
        verbose: bool = bool(opts.get('verbose'))

        async with self.mcp() as client:
            if verbose:
                parts = tqdm(parts, desc="Inserting messages", unit="convo")
            
            for part in parts:
                if isinstance(part, ImportMemory):
                    prev = await self.insert_memory(
                        config, sona, None, None, client, part
                    )
                    if verbose:
                        tqdm.write(str(prev))
                    continue

                # ImportConvo
                prev = part.prev
                chatlog = part.chatlog
                if verbose:
                    chatlog = tqdm(
                        chatlog,
                        desc="Inserting chat log",
                        unit="msg",
                        leave=False
                    )
                
                # Convo has metadata, insert it to act as a dependency ever
                # memory in it has in common
                if part.metadata:
                    ts = part.metadata.timestamp
                    md = await self.insert_memory(
                        config, sona, None, None, client,
                        IncompleteMemory(
                            data=MetaData(
                                metadata=MetaData.Content(
                                    export=MetaData.Content.Export(
                                        provider=part.metadata.provider,
                                        convo_uuid=part.metadata.uuid,
                                        convo_title=part.metadata.title
                                    )
                                )
                            ),
                            timestamp=ts and int(ts.timestamp()),
                            importance=part.metadata.importance
                        )
                    )
                else:
                    md = None

                for m in chatlog:
                    if not isinstance(m, PartialMemory):
                        raise TypeError(f"Expected PartialMemory, got {type(m)}")
                    
                    prev = await self.insert_memory(
                        config, sona, md, prev, client, m
                    )
                    if verbose:
                        tqdm.write(str(prev))

    async def subcmd_chat(self, *argv: str):
        '''
        [-l [-m]] [msg]

        Interact with the agent in a single-turn conversation.

        parameters:
        msg             The message to send to the agent. If missing, use STDIN.

        options:
          -l,--list       Show the recalled chat log.
          -v,--verbose    Show the total context as seen by the agent.
          -q,--quiet      Do not print extra data, just the raw contents.
        '''

        args, opts = argparse(argv, {
            "-l,--list": False,
            "-v,--verbose": False,
            "-q,--quiet": False,
        })
        if len(args) > 1:
            raise ValueError("Expected a single message argument.")
        
        message = args[0] if args else input("<user> ").strip()
        
        config = self.get_config()
        async with self.mcp() as client:
            chatlog = await client.chat(
                prompt=IncompleteMemory(
                    data=TextData(content=message)
                ),
                system_prompt=SYSTEM_PROMPT,
                recall_config=config.recall or RecallConfig(),
                chat_config=config.chat or SampleConfig(),
                annotate_config=config.annotate or SampleConfig()
            )
            
            # Just dump the response
            if not opts['list']:
                print(chatlog[-1].document())
                return
            
            self.print_chatlog(
                chatlog, {
                    m.cid: i for i, m in enumerate(chatlog, start=1)
                },
                meta=bool(opts.get('meta')),
                verbose=bool(opts.get('verbose')),
                extra=not opts.get('quiet')
            )

    async def subcmd_query(self, *argv: str):
        '''
        [-l [-m]] [msg]

        Interact with the agent without committing it to memory.

        parameters:
        msg             The message to send to the agent. If missing, use STDIN.

        options:
          -l,--list       Show the recalled chat log.
          -v,--verbose    Show the total context as seen by the agent.
          -q,--quiet      Do not print extra data, just the raw contents.
        '''

        args, opts = argparse(argv, {
            "-l,--list": False,
            "-v,--verbose": False,
            "-q,--quiet": False,
        })
        if len(args) > 1:
            raise ValueError("Expected a single message argument.")
        
        message = args[0] if args else input("<user> ").strip()
        
        config = self.get_config()
        async with self.mcp() as client:
            chatlog = await client.query(
                prompt=IncompleteMemory(
                    data=TextData(content=message),
                ),
                system_prompt=SYSTEM_PROMPT,
                recall_config=config.recall or RecallConfig(),
                chat_config=config.chat or SampleConfig()
            )
            
            # Just dump the response
            if not opts['list']:
                print(chatlog.response)
                return
            
            meta=bool(opts.get('meta'))
            verbose=bool(opts.get('verbose'))
            extra=not opts.get('quiet')

            log = chatlog.chatlog
            self.print_chatlog(
                log, {
                    m.cid: i for i, m in enumerate(log, start=1)
                },
                meta=meta,
                verbose=verbose,
                extra=extra
            )
            if verbose:
                print("<user>", message)
                content = chatlog.response.content
                assert isinstance(content, TextContent)
                print("<assistant>", content.text)

    async def subcmd_recall(self, *argv: str):
        '''
        [-s sona] [search]

        Recall a chat log from the agent's memory.

        parameters:
          search          The search term to use for recalling the chat log.

        options:
          -s,--sona       The sona to recall from. If not specified, the default
                           chat sona is used.
          -v,--verbose    Show the total context as seen by the agent.
          -q,--quiet      Do not print extra data, just the raw contents.
        '''
        args, opts = argparse(argv, {
            "-s,--sona": str,
            "-v,--verbose": False,
            "-q,--quiet": False
        })

        if len(args) > 1:
            raise ValueError("Expected a single CID argument.")
        
        search = args[0] if args else input("search> ").strip()

        config = self.get_config()
        async with self.mcp() as client:
            if (sona := opts.get('sona')) is None:
                sona = self.get_config().sona

            g = await client.recall(
                prompt=IncompleteMemory(
                    data=TextData(content=search),
                    sonas=[str(sona)]
                ),
                recall_config=config.recall or RecallConfig(),
            )
            log = []
            refs: dict[CIDv1, int] = {}
            for cid in g.invert().toposort(key=lambda m: m.timestamp):
                assert g[cid].cid == cid
                log.append(g[cid])
                refs[cid] = len(log)
            
            self.print_chatlog(
                log, refs,
                verbose=bool(opts.get('verbose')),
                extra=not opts.get('quiet')
            )
    
    def subcmd_config(self, *argv: str):
        '''
        ["edit"]

        Show or edit the configuration file.
        
        parameters:
          "edit"          Edit the configuration file using the configured editor.
                           If no arguments are given, the configuration is printed.
        '''
        check_overflow(argv[1:])
        config = self.get_config()
        if argv:
            if argv[0] == 'edit':
                editor = os.environ.get("EDITOR", "nano")
                os.execvp(editor, [
                    editor, os.path.expanduser(self.config)
                ])
            raise CmdError(f"Unknown argument {argv[0]!r} for config command.")

        print(config.source)

    def subcmd_help(self, *argv: str):
        '''
        [cmd]

        Show help for the given command or all commands.
        '''
        check_overflow(argv[1:])
        return self.usage(*argv[:1])

def main(name, *argv):
    try:
        if app := MemoriaApp.argparse(*argv):
            app.run()
        else:
            warn("Expected a command.")
            print(MemoriaApp().usage())
    except BrokenPipeError:
        pass # head and tail close stdout
    except KeyboardInterrupt:
        print()
    except CmdError as e:
        print("Error:", e, file=sys.stderr)

if __name__ == "__main__":
    import sys
    main(*sys.argv)