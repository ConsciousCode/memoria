#!/usr/bin/env python3

from contextlib import asynccontextmanager
import os
from typing import TYPE_CHECKING, Any, Final, Iterable, Iterator, NoReturn, Optional
import inspect

from pydantic import BaseModel

from ipld.cid import CIDv1
from models import Chatlog, OtherMemory, RecallConfig, SelfMemory
from util import ifnone

if TYPE_CHECKING:
    # These can be pretty beefy, so put in a type check to avoid
    # importing them unless needed.
    from fastmcp.client.sampling import SamplingParams
    from fastmcp.utilities.types import MCPContent
    from mcp import CreateMessageResult, SamplingMessage
    from mcp.types import TextContent

SERVER: Final = "http://127.0.0.1:8000/mcp"
CONFIG: Final = "./private/memoria.toml"
CHAT_SONA: Final = "chat"
TEMPERATURE: Final = 0.7

class CmdError(Exception):
    pass

class OpenAIConfig(BaseModel):
    api_key: str

class Config(BaseModel):
    source: str
    '''Original source of the config file.'''
    server: str
    '''MCP server URL to connect to.'''
    sona: Optional[str]
    '''Default sona to use for chat.'''
    temperature: float
    '''Default temperature for chat responses.'''
    recall: Optional[RecallConfig]
    '''Configuration for how to weight memory recall.'''
    openai: Optional[OpenAIConfig]
    '''OpenAI API configuration.'''

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
                    try:
                        opts[name] = int(val)
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

def print_chatlog(chatlog: Chatlog, meta: bool=False):
    """
    Print the chat log in a readable format.
    
    Parameters:
        chatlog: The Chatlog object to print.
        meta: If True, print metadata about the chat log.
    """
    for m in chatlog.chatlog:
        assert isinstance(m, OtherMemory)
        print(f"<{m.kind}> {m.document()}")
    
    print(f"<assistant> {chatlog.response.document()}")

class MemoriaApp:
    def __init__(self, *rest, help=None, config=None):
        self.rest = rest
        self.help = help
        self.config = config or CONFIG
    
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
          query [query]     Talk to the agent without creating memories.
          chat [message]    Single-turn conversation with the agent. (default)
          recall [search]   Recall a chat log from the agent's memory.
          help              Show this help message.
        """
        
        if what == "":
            doc = inspect.cleandoc(self.usage.__doc__ or "")
        elif sub := getattr(self, f"subcmd_{what}", None):
            doc = inspect.cleandoc(sub.__doc__ or "")
            doc = "usage: {name} " + f"{what} {doc}"
        else:
            doc = inspect.cleandoc(self.usage.__doc__ or "")
            doc = f"Unknown subcommand {what!r}\n\n{doc}"
        
        return doc
    
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

                [openai]
                # OpenAI API configuration
                #api_key = "sk-###"
            ''')
            data = {}
        
        # Coerce empty sections to None
        recall = data.get("recall") or None
        openai = data.get("openai") or None
        
        return Config(
            source=source,
            server=data.get("server", SERVER),
            sona=data.get("sona"),
            temperature=data.get("temperature", TEMPERATURE),
            recall=recall and RecallConfig.model_validate(recall),
            openai=openai
        )
    
    async def chatquery(self, tool: str, *argv: str):
        from fastmcp.client.client import Client
        from fastmcp.client.transports import StreamableHttpTransport
        from mcp.types import TextContent

        args, opts = argparse(argv, {
            "-l,--list": False,
            "-m,--meta": False,
        })
        if len(args) > 1:
            raise ValueError("Expected a single message argument.")
        
        message = args[0] if args else input("<user> ").strip()
        
        async with self.mcp() as client:
            contents = await client.call_tool(
                tool, {"sona": None, "message": message}
            )
            
            if not opts['list']:
                print("".join(getattr(c, "text", "") for c in contents))
                return
            
            for c in contents:
                assert isinstance(c, TextContent)
                return Chatlog.model_validate_json(c.text)
            
            raise ValueError("No chat log found in response.")
    
    async def sampling_handler(self, msgs: list['SamplingMessage'], params: 'SamplingParams', ctx):
        import httpx
        from mcp.types import TextContent
        from mcp import CreateMessageResult

        if (api_key := os.getenv("OPENAI_API_KEY")) is None:
            config = self.get_config()
            if config.openai is None or not config.openai.api_key:
                raise RuntimeError("OPENAI_API_KEY is required")
            api_key = config.openai.api_key
        
        prefs = params.modelPreferences
        if isinstance(prefs, list) and prefs:
            model = prefs[0]
        elif isinstance(prefs, str):
            model = prefs
        else:
            model = "gpt-4.1-nano"
        messages = []
        if params.systemPrompt:
            messages.append({"role": "system", "content": params.systemPrompt})
        for m in params.messages:
            assert isinstance(m.content, TextContent), "Only TextContent is supported"
            messages.append({"role": m.role, "content": m.content.text})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": ifnone(params.temperature, 0.7),
            "max_tokens": params.maxTokens,
            "stop": params.stopSequences,
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
            )
        data = r.json()
        choice = data.get("choices", [{}])[0]
        return CreateMessageResult(
            role=choice.get("message", {}).get("role", "assistant"),
            model=data.get("model"),
            content=TextContent(
                type="text",
                text=choice.get("message", {}).get("content", "")
            ),
            stopReason=choice.get("finish_reason"),
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
            yield client

    async def subcmd_chat(self, *argv: str):
        '''
        [-l [-m]] [msg]

        Interact with the agent in a single-turn conversation.

        parameters:
        msg             The message to send to the agent. If missing, use STDIN.

        options:
        -l,--list       Show the recalled chat log.
        -m,--meta       Show the total context as seen by the agent.
        '''
        
        if (chat := await self.chatquery("chat", *argv)) is None:
            return
        
        print_chatlog(chat)

    async def subcmd_query(self, *argv: str):
        '''
        [-l [-m]] [msg]

        Interact with the agent without committing it to memory.

        parameters:
        msg             The message to send to the agent. If missing, use STDIN.

        options:
        -l,--list       Show the recalled chat log.
        -m,--meta       Show the total context as seen by the agent.
        '''
        
        if (chat := await self.chatquery("query", *argv)) is None:
            return
        
        print_chatlog(chat)

    async def subcmd_recall(self, *argv: str):
        '''
        [-s sona] [search]

        Recall a chat log from the agent's memory.

        parameters:
          search          The search term to use for recalling the chat log.

        options:
          -s,--sona       The sona to recall from. If not specified, the default
                           chat sona is used.
        '''
        args, opts = argparse(argv, {
            "-s,--sona": str
        })

        if len(args) > 1:
            raise ValueError("Expected a single CID argument.")
        
        search = args[0] if args else input("search> ").strip()

        async with self.mcp() as client:
            if (sona := opts.get('sona')) is None:
                sona = self.get_config().sona
            
            chatlog = await client.call_tool("recall", {
                "sona": sona,
                "prompt": search
            })
            for item in chatlog:
                assert isinstance(item, TextContent)
                print_chatlog(
                    Chatlog.model_validate_json(item.text)
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