'''
An Interpreter is a component which advances the Subject's state by interpreting and responding to context. It's able to use whatever heuristics, models, and tools it needs to accomplish the task. In the production of new memories, side effects may occur which is desirable; this is how the total agent is able to take any actions at all.

Memoria is designed to have one or more interpreters, each managing its own input event stream and manipulations of the Subject's state.
'''

from abc import ABC, abstractmethod
import asyncio
from typing import TYPE_CHECKING, NotRequired, assert_never, override
from collections.abc import AsyncIterable, Iterable
import json

from aioipfs import AsyncIPFS
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, SystemPromptPart, TextPart, ThinkingPart, ToolCallPart, ToolReturnPart, UserPromptPart, BinaryContent

from memoria import memory as M
from memoria.bus import Receiver, Bus

if TYPE_CHECKING:
    from typing_extensions import TypedDict

    from pydantic_ai import Agent

    from cid import CID, CIDv1

    from memoria.memory import PartialMemory, MemoryDAG
    from memoria.util import json_t

    class ToolCallArgs(TypedDict):
        tool_name: str
        args: dict[str, json_t]
        tool_call_id: NotRequired[str]

    class ToolReturnArgs(TypedDict):
        tool_name: str
        content: json_t
        tool_call_id: str

def build_prefix(name: str, refs: dict[CIDv1, int], m: 'PartialMemory') -> str:
    deps = ', '.join(str(refs[e]) for e in m.edges)
    return f"[{name}:{refs[m.cid]}->{deps}]\t"

async def to_messages(ipfs: AsyncIPFS, refs: dict[CIDv1, int], ms: Iterable[PartialMemory]) -> AsyncIterable[ModelMessage]:
    pending_return: list[ToolReturnPart | ModelRequestPart] = []
    for m in ms:
        if m.data.kind == "self":
            res_parts: list[ModelResponsePart] = []
            
            for part in m.data.parts:
                match part:
                    case M.TextPart(content=content):
                        res_parts.append(TextPart(content=content))

                    case M.ThinkPart(
                            content=content, think_id=think_id,
                            signature=signature
                        ):
                        res_parts.append(ThinkingPart(
                            content=content,
                            id=think_id,
                            signature=signature
                        ))

                    case M.ToolPart(
                            name=name,
                            args=args,
                            result=result,
                            call_id=call_id
                        ):
                        tcall: ToolCallArgs = {
                            "tool_name": name,
                            "args": args
                        }
                        if call_id:
                            tcall['tool_call_id'] = call_id
                        tcp = ToolCallPart(**tcall)
                        res_parts.append(tcp)

                        pending_return.append(ToolReturnPart(
                            tool_name=name,
                            content=result,
                            tool_call_id=tcp.tool_call_id
                        ))

            prefix = build_prefix("ref", refs, m)
            match res_parts:
                case [TextPart(content=content) as part, *_]:
                    part.content = prefix + content
                case _:
                    res_parts.insert(0, TextPart(content=prefix))
            
            yield ModelResponse(
                parts=res_parts,
                model_name=m.data.model
            )
        else:
            req_parts, pending_return = pending_return, []
            match d := m.data:
                case M.OtherData(parts=parts):
                    for part in parts:
                        match part:
                            case M.TextPart(content=content):
                                req_parts.append(UserPromptPart(content=content))
                            
                            case M.FilePart(file=file, mimetype=mt):
                                if data := await ipfs.cat(file):
                                    req_parts.append(UserPromptPart(content=[
                                        BinaryContent(data=data, media_type=mt)
                                    ]))
                    
                    prefix = build_prefix("ref", refs, m)
                    match req_parts:
                        case [UserPromptPart(content=str(content)) as part, *_]:
                            part.content = prefix + content
                        case [UserPromptPart(content=content) as part, *_]:
                            part.content = [prefix, *content]
                        case _:
                            req_parts.insert(0, UserPromptPart(content=prefix))

                case M.SystemData(parts=parts):
                    req_parts.extend(
                        SystemPromptPart(content=part.content)
                            for part in parts
                    )
                    prefix = build_prefix("ref", refs, m)
                    match req_parts:
                        case [SystemPromptPart(content=content) as part, *_]:
                            part.content = prefix + content
                        case _:
                            req_parts.insert(0, SystemPromptPart(content=prefix))

                case M.MetaData(metadata=md):
                    req_parts.append(UserPromptPart(
                        content=build_prefix("metadata", refs, m) + json.dumps(md)
                    ))

                case _:
                    assert_never(d)

            yield ModelRequest(req_parts)

    if pending_return:
        yield ModelRequest(pending_return)

class Interpreter(Receiver):
    def __init__(self, agent: 'Agent', ipfs: AsyncIPFS):
        self.agent: 'Agent' = agent
        self.ipfs = ipfs
        self.pending = MemoryDAG()
        self.important = asyncio.Event()

    @override
    async def task(self, bus: 'Bus'):
        while True:
            await self.important.wait()
            await bus.push(self.pending)
            self.pending.clear()
            self.important.clear()

    @abstractmethod
    @override
    async def recv(self, update: MemoryDAG):
        ...

    async def process(self, g: 'MemoryDAG'):
        refs: dict[CIDv1, int] = {}
        ms: list['PartialMemory'] = []
        for cid in g.toposort(lambda m: m.uuid, reverse=True):
            refs[cid] = len(refs)
            ms.append(g[cid])
        
        return await self.agent.run(
            message_history=[
                m async for m in to_messages(self.ipfs, refs, ms)
            ]
        )
