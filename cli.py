#!/usr/bin/env python3

import inspect
import os
import sys
import asyncio
from typing import Iterable, Iterator, Optional

from fastmcp.client.sampling import SamplingParams
from fastmcp.utilities.types import MCPContent
import httpx
from fastmcp.client.client import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp import CreateMessageResult, SamplingMessage
from mcp.types import TextContent
from pydantic import BaseModel

from ipld.cid import CIDv1
from models import Chatlog, OtherMemory, SelfMemory
from util import ifnone

SERVER = "http://127.0.0.1:8000/mcp"

def unpack[*A](args: Iterable[str], *defaults: *A) -> tuple[str, ...]|tuple[*A]:
    '''Unpack rest arguments with defaults and proper typing.'''
    return (*args, *defaults)[:len(defaults)] # type: ignore

def expected(name: str):
    raise ValueError(f"Expected a {name}.")

def warn(msg):
    print(f"Warning: {msg}", file=sys.stderr)

def check_overflow(rest):
    if rest: warn("Too many arguments.")

async def sampling_handler(msgs: list[SamplingMessage], params: SamplingParams, ctx):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
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

def argparse(argv: tuple[str, ...], config: dict[str, type[bool]|type[int]|type[str]]):
    def named_value(arg: str, it: Iterator[tuple[int, str]]):
        try: return next(it)
        except StopIteration:
            raise expected(f"value after {arg}") from None
    
    which = {}
    for aliases, v in config.items():
        als = aliases.split(',')
        match [a for a in als if a.startswith("--")]:
            case []: name = None
            case [name]: name = name.removeprefix('--')
            case long: raise ValueError(f"Multiple long options found: {long}")

        for k in als:
            which[k] = v, name or k.replace('-', "")
    
    pos = []
    opts = {}

    it = iter(enumerate(argv, 1))
    try:
        while True:
            i, arg = next(it)
            if arg.startswith("-"):
                if arg == "--":
                    pos.extend(arg for _, arg in it)
                    break
                
                if (c := which.get(arg)) is None:
                    raise ValueError(f"Unknown option {arg!r}")
                
                t, name = c

                if t is bool:
                    opts[name] = True
                elif t is int:
                    i, val = named_value(arg, it)
                    try:
                        opts[name] = int(val)
                    except ValueError:
                        raise ValueError(f"Expected an integer after {arg!r}") from None
                elif t is str:
                    i, opts[name] = named_value(arg, it)
                else:
                    raise TypeError(f"Unsupported type {t} for option {arg!r}")
            else:
                pos.append(arg)
    except StopIteration:
        pass

    for t, name in which.values():
        if t is bool and name not in opts:
            opts[name] = False
    
    return pos, opts

class ChatModel(BaseModel):
    cid: Optional[CIDv1]
    chatlog: list[MCPContent]
    response: MCPContent

async def chatquery(tool, argv: tuple[str, ...]):
    args, opts = argparse(argv, {
        "-l,--list": bool,
        "-m,--meta": bool,
    })
    if len(args) == 0:
        message = input("<user> ").strip()
    elif len(args) > 1:
        raise ValueError("Expected a single message argument.")
    else:
        message: str = args[0]
    
    async with Client(
        StreamableHttpTransport(SERVER),
        sampling_handler=sampling_handler,
    ) as client:
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

async def chat(argv: tuple[str, ...]):
    '''
    [-l [-m]] [msg]

    Interact with the agent in a single-turn conversation.

    parameters:
      msg             The message to send to the agent. If missing, use STDIN.

    options:
      -l,--list       Show the recalled chat log.
      -m,--meta       Show the total context as seen by the agent.
    '''
    
    if (chat := await chatquery("chat", argv)) is None:
        return
    
    for m in chat.chatlog:
        print(f"<{m.kind}> {m.document()}")
    
    print(f"<assistant> {chat.response.document()}")

async def query(argv: tuple[str, ...]):
    '''
    [-l [-m]] [msg]

    Interact with the agent without committing it to memory.

    parameters:
      msg             The message to send to the agent. If missing, use STDIN.

    options:
      -l,--list       Show the recalled chat log.
      -m,--meta       Show the total context as seen by the agent.
    '''
    
    if (chat := await chatquery("query", argv)) is None:
        return
    
    for m in chat.chatlog:
        print(f"<{m.kind}> {m.document()}")
    
    print(f"<assistant> {chat.response.document()}")

def usage():
    return inspect.cleandoc("""
        usage: python cli.py cmd ...
        Commands:
          query <query>     Talk to the agent without creating memories.
          chat <message>    Single-turn conversation with the agent.
          help              Show this help message.
    """)

async def main(*argv):
    match argv[0]:
        case "chat":
            await chat(argv[1:])
        
        case 'query':
            await query(argv[1:])
        
        case _:
            raise ValueError(f"Unknown command {argv[0]!r}")

if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))  # Skip the script name