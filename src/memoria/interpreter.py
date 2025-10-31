'''
An Interpreter is a component which advances the Subject's state by interpreting and responding to context. It's able to use whatever heuristics, models, and tools it needs to accomplish the task. In the production of new memories, side effects may occur which is desirable; this is how the total agent is able to take any actions at all.

Memoria is designed to have one or more interpreters, each managing its own input event stream and manipulations of the Subject's state.
'''
from datetime import datetime
from typing import TYPE_CHECKING, NotRequired, assert_never
import json
from typing_extensions import TypedDict
from collections.abc import AsyncIterable, Iterator

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, TextPart, ThinkingPart, ToolCallPart, ToolReturnPart, UserPromptPart, BinaryContent

from ipld import dag

from memoria.memory import FileData, Memory, MemoryContext, MemoryDAG, PartialMemory, SelfData, TextData, MetaData
from memoria.util import json_t

if TYPE_CHECKING:
    from cid import CID, CIDv1
    from pydantic_ai import Agent
    from memoria.subject.client import SubjectClient

    class ToolCallArgs(TypedDict):
        tool_name: str
        args: dict[str, json_t]
        tool_call_id: NotRequired[str]

    class ToolReturnArgs(TypedDict):
        tool_name: str
        content: json_t
        tool_call_id: str

    class TSArgs(TypedDict):
        timestamp: NotRequired[datetime]

class Interpreter:
    def __init__(self, agent: 'Agent', client: 'SubjectClient'):
        self.agent: 'Agent' = agent
        self.client: 'SubjectClient' = client

    async def resolve_cid(self, cid: 'CID') -> bytes | None:
        return await self.client.ipfs(cid)

    async def to_message(self, ms: Iterator[MemoryContext[PartialMemory]]) -> AsyncIterable[ModelMessage]:
        pending_return: list[ToolReturnPart | ModelRequestPart] = []
        for mc in ms:
            m = mc.memory
            ts = mc.timestamp
            tsargs: TSArgs = {
                "timestamp": datetime.fromtimestamp(ts)
            } if ts else {}
            if m.data.kind == "self":
                res_parts: list[ModelResponsePart] = []
                for part in m.data.parts:
                    match part:
                        case SelfData.TextPart(content=content):
                            res_parts.append(TextPart(
                                content=content
                            ))

                        case SelfData.ThinkPart(
                                content=content, think_id=think_id,
                                signature=signature
                            ):
                            res_parts.append(ThinkingPart(
                                content=content,
                                id=think_id,
                                signature=signature
                            ))

                        case SelfData.ToolPart(
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

                            tret: ToolReturnArgs = {
                                "tool_name": name,
                                "content": result,
                                "tool_call_id": tcp.tool_call_id
                            }
                            pending_return.append(ToolReturnPart(**tret))

                yield ModelResponse(
                    parts=res_parts,
                    model_name=m.data.model,
                    **tsargs
                )
                continue

            req_parts: list[ModelRequestPart] = []
            req_parts.extend(pending_return)
            pending_return.clear()
            match d := m.data:
                case TextData(content=content):
                    req_parts.append(UserPromptPart(
                        content=content,
                        **tsargs
                    ))

                case FileData(file=file, filename=fn, mimetype=mt):
                    data = await self.resolve_cid(file)
                    if data is not None:
                        req_parts.append(UserPromptPart(
                            content=[BinaryContent(
                                data=data,
                                media_type=mt,
                                identifier=fn
                            )],
                            **tsargs
                        ))

                case MetaData(metadata=md):
                    req_parts.append(UserPromptPart(
                        content=json.dumps(md),
                        **tsargs
                    ))

                case _:
                    assert_never(d)

            yield ModelRequest(req_parts)

        if pending_return:
            yield ModelRequest(pending_return)

    async def process(self, context: list['CIDv1']):
        g = MemoryDAG()
        for cid in context:
            if data := await self.client.ipfs(cid):
                g[cid] = MemoryContext(
                    memory=Memory.model_validate(
                        dag.unmarshal(cid.codec, data)
                    ).partial(
                        lambda e: e.target in context
                    )
                )

        return await self.agent.run(
            message_history=[
                x async for x in self.to_message(g[m] for m in g.toposort())
            ]
        )
