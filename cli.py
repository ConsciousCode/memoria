#!/usr/bin/env python3

'''
CLI for interacting with Memoria systems, particularly via the Memoria Subject.
'''

from contextlib import asynccontextmanager
from datetime import datetime
import os
from typing import Final, Iterable, Optional, Sequence, cast
import inspect
from uuid import UUID
import mimetypes

from tqdm import tqdm
from mcp.shared.context import LifespanContextT, RequestContext
from mcp.types import ModelPreferences, TextContent, SamplingMessage
from mcp import ClientSession, CreateMessageResult
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from fastmcp.client.sampling import SamplingParams

from cid import CIDv1

from memoria.util import argparse, check_overflow, expected, named_value, warn
from memoria.subject.config import Config
from memoria.subject.client import SubjectClient
from memoria.subject._common import AddParameters
from memoria.memory import DraftMemory, Edge, FileData, ImportAdapter, ImportMemory, Memory, MetaData, PartialMemory
from memoria.config import RecallConfig, SampleConfig

SERVER: Final = "http://127.0.0.1:8000/mcp"
CONFIG: Final = "./private/memoria.toml"

SYSTEM_PROMPT = """You are a helpful AI assistant. You can answer questions, provide information, and assist with tasks."""

class CmdError(Exception):
    pass

class Sampler:
    '''
    Sampling handler for MCP using pydantic_ai Agents as the driver. Handles
    loading the configuration and selecting the appropriate model based on
    the provided preferences.
    '''
    def __init__(self, config: Config):
        self.config = config
    
    def convert_history(self, msgs: list[SamplingMessage]) -> Iterable[ModelRequest | ModelResponse]:
        """Convert SamplingMessages to ModelRequest/ModelResponse history."""
        for m in msgs:
            assert isinstance(m.content, TextContent), "Only TextContent is supported"
            text = m.content.text
            if m.role == "user":
                yield ModelRequest.user_text_prompt(text)
            elif m.role == "assistant":
                yield ModelResponse(parts=[TextPart(content=text)])
            else:
                raise ValueError(f"Unknown role {m.role!r}")
    
    def convert_settings(self, params: SamplingParams) -> ModelSettings:
        """Convert SamplingParams to ModelSettings."""
        settings: ModelSettings = {}
        if params.maxTokens and params.maxTokens < 4096:
            settings["max_tokens"] = params.maxTokens
        if params.stopSequences:
            settings["stop_sequences"] = params.stopSequences
        if params.temperature is not None:
            settings["temperature"] = params.temperature
        elif self.config.temperature is not None:
            settings["temperature"] = self.config.temperature
        return settings
    
    async def sampling_handler(self,
            msgs: list[SamplingMessage],
            params: 'SamplingParams',
            ctx: RequestContext[ClientSession, LifespanContextT]
        ) -> CreateMessageResult:
        """Sampling handler for FastMCP using Pydantic AI Agent."""
        config = self.config
        prefs = params.modelPreferences or ModelPreferences()
        for prov, name in config.select_model(prefs):
            provider = config.build_provider(prov)
            model = config.build_model(name, provider) if provider else name

            agent = Agent(model,
                instructions=params.systemPrompt
            )
            result = await agent.run(
                message_history=list(self.convert_history(msgs)),
                model_settings=self.convert_settings(params)
            )
            return CreateMessageResult(
                role="assistant",
                content=TextContent(type="text", text=result.output),
                model=f"{prov}:{name}",
                stopReason="endTurn"
            )
        raise ValueError("No suitable model found for sampling handler.")

class MemoriaApp:
    def __init__(self, *rest, help=None, config=None):
        self.rest = rest
        self.help = help
        self.config = config or CONFIG
    
    def tagname(self, memory: PartialMemory):
        match memory.data.kind:
            case "self": return "assistant"
            case "other": return "user"
            case kind: return kind

    def print_memory(self, memory: PartialMemory, refs: dict[CIDv1, int], verbose=True, extra=True):
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
            
            after = ' ' + ' '.join(parts) if verbose and parts else ""
            print(f"<{self.tagname(memory)}{after}> ", end='')
        print(memory.data.document())
    
    def print_chatlog(self, log: Sequence[PartialMemory], refs: dict[CIDv1, int], meta=False, verbose=True, extra=True):
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
        return Config.from_file(self.config)
    
    @asynccontextmanager
    async def mcp(self):
        from fastmcp.client.client import Client
        from fastmcp.client.transports import StreamableHttpTransport
        
        config = self.get_config()
        sampler = Sampler(config)

        async with Client(
            StreamableHttpTransport(config.server),
            sampling_handler=sampler.sampling_handler
        ) as client:
            yield SubjectClient(client)
    
    async def insert_memory(self,
            config: Config,
            sona: Optional[UUID|str],
            metadata_cid: Optional[CIDv1],
            prev: Optional[CIDv1],
            client: SubjectClient,
            memory: DraftMemory | ImportMemory
        ) -> CIDv1:
        if sona:
            if memory.sonas is None:
                memory.sonas = []
            memory.sonas.append(sona)
        
        if metadata_cid:
            memory.edges.append(Edge(target=metadata_cid, weight=0.0))
        
        if prev:
            memory.edges.append(Edge(target=prev, weight=1.0))
        
        if isinstance(memory, ImportMemory):
            with open(memory.data.path, "rb") as f:
                data = f.read()
            if (filename := memory.data.filename):
                filename = os.path.basename(filename)
            else:
                filename = memory.data.path
            
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
            memory = DraftMemory(
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
                        DraftMemory(
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
                prompt=Memory(
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
                prompt=Memory(
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
                print("<assistant>", chatlog.response.document())

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
                prompt=Memory(
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
