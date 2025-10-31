from typing import Iterable
import random

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.mcp import ToolResult
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ModelResponsePart, TextPart, ThinkingPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.usage import Usage

from ipld import dag
from cid import CIDv1
from multibase import base58
from memoria.memory import Memory, TextData, PartialMemory, SelfData
from memoria.subject.client import SubjectClient
from memoria.prompts import CHAT_PROMPT

from ._common import InterpretRequest

app = FastAPI()

class SplitImportance(BaseModel):
    """Explicitly label components of importance for clarity."""
    novelty: int
    intensity: int
    future: int
    personal: int
    saliency: int

class Output(BaseModel):
    """Expected output from the agent."""
    relevance: dict[str, int]
    importance: SplitImportance

agent = Agent(
    model="gpt-4o",
    name="InterpreterAgent",
    instructions=CHAT_PROMPT,
    output_type=Output
)

def build_history(refs: dict[CIDv1, int], history: list[PartialMemory]) -> Iterable[ModelMessage]:
    """Build a message history from a list of Memory objects."""
    for ref, memory in enumerate(history):
        match memory.data:
            case SelfData(parts=parts, model=model):
                msg_parts: list[ModelResponsePart] = []
                results: list[ToolReturnPart] = []
                for part in parts:
                    match part:
                        case SelfData.TextPart(content=content):
                            msg_parts.append(TextPart(content=content))

                        case SelfData.ThinkPart(
                                content=content, think_id=think_id,
                                signature=signature
                            ):
                            msg_parts.append(ThinkingPart(
                                content=content,
                                id=think_id,
                                signature=signature
                            ))
                        
                        case SelfData.ToolPart(
                                name=name, args=args, result=result,
                                call_id=call_id
                            ):
                            if call_id is None:
                                bs = random.randbytes(32)
                                call_id = f"memoria-{base58.encode(bs)}"
                            
                            msg_parts.append(ToolCallPart(
                                tool_name=name,
                                args=args,
                                tool_call_id=call_id
                            ))
                            results.append(ToolReturnPart(
                                tool_name=name,
                                content=result,
                                tool_call_id=call_id
                            ))
                
                yield ModelResponse(
                    parts=msg_parts,
                    usage=Usage(),
                    model_name=model
                )
            
            case TextData(content=content):
                yield ModelRequest(
                    parts=[UserPromptPart(content=content)]
                )
            
            case data:
                raise NotImplementedError(f"Memory part {type(data)}")

@app.post("/interpret")
async def interpret(
        ir: InterpretRequest
    ) -> Memory:
    """
    Interpret the provided context and produce a new memory.
    """
    async with httpx.AsyncClient(base_url="http://localhost:8000") as http:
        subject = SubjectClient(http)
        history: list[PartialMemory] = []
        refs: dict[CIDv1, int] = {}
        for ref, cid in enumerate(ir.context):
            if (bs := await subject.ipfs(cid)) is None:
                continue
            
            m = Memory.model_validate(
                dag.unmarshal(cid.codec, bs)
            )
            # Strip edges that aren't in the context subgraph
            m = PartialMemory(
                cid=cid,
                data=m.data,
                edges=[e for e in m.edges if e.target not in ir.context]
            )
            refs[m.cid] = ref
            history.append(m)
    
    response = await agent.run(
        message_history=list(build_history(refs, history))
    )
    response.output

def main():
    import uvicorn
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8001
    )
    server = uvicorn.Server(config)
    try: server.run()
    except KeyboardInterrupt:
        print("Interpreter server stopped by user.")

if __name__ == "__main__":
    main()