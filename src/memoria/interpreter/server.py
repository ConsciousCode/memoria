from typing import Iterable
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

from ipld import dag
from cid import CIDv1
from memoria.memory import Memory, OtherData, PartialMemory, SelfData
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

def build_history(history: list[Memory]) -> Iterable[ModelMessage]:
    """Build a message history from a list of Memory objects."""
    for memory in history:
        match memory.data:
            case SelfData(name=name, parts=parts, stop_reason=stop_reason):
                yield ModelResponse(
                    content=memory.content,
                    metadata={
                        "cid": str(memory.cid),
                        "created_at": memory.created_at.isoformat(),
                        "updated_at": memory.updated_at.isoformat()
                    }
                )
            
            case OtherData(name=name, content=content):
                yield ModelRequest(
                    content=content,
                    metadata={
                        "cid": str(memory.cid),
                        "created_at": memory.created_at.isoformat(),
                        "updated_at": memory.updated_at.isoformat()
                    }
                )

@app.post("/interpret")
async def interpret(
        ir: InterpretRequest
    ) -> Memory:
    """
    Interpret the provided context and produce a new memory.
    """
    async with httpx.AsyncClient(base_url="http://localhost:8000") as http:
        subject = SubjectClient(http)
        memories: dict[CIDv1, PartialMemory] = {}
        for cid in ir.context:
            if (bs := await subject.ipfs(cid)) is None:
                continue
            
            m = Memory.model_validate(
                dag.unmarshal(cid.codec, bs)
            )
            # Strip edges that aren't in the context subgraph
            memories[cid] = PartialMemory(
                cid=cid,
                data=m.data,
                edges=[e for e in m.edges if e.target not in ir.context]
            )
    
    response = await agent.run(
        message_history=list(build_history(history))
    )

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