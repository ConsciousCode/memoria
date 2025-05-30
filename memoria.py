from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Annotated, Iterable, Literal, Optional, cast
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from fastembed import TextEmbedding, ImageEmbedding
from fastembed.common.types import ImageInput

from pydantic_ai import Agent, RunContext, capture_run_messages

from db import Database, MemoryKind, OtherMemory, SelfMemory
from util import X, json_t, todo_list

nomic_text = TextEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
)
'''
ImageEmbedding.add_custom_model(
    model="nomic-ai/nomic-embed-vision-v1.5",
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf="nomic-ai/nomic-embed-vision-v1.5"),
    dim=768
)
nomic_image = ImageEmbedding(
    model_name="nomic-ai/nomic-embed-vision-v1.5"
)
'''

@dataclass
class Memory:
    rowid: int
    timestamp: Optional[datetime]
    kind: MemoryKind
    data: json_t
    importance: Optional[float]
    edges: dict[str, list[int]]
    role: Optional[Literal['prev']] = None

class System1ResponseModel(BaseModel):
    response: str = Field(description="My response to the prompt.")
    importance: int = Field(description="The importance of the prompt on a scale of 1-10, from forgettable trivia to unforgettable statements.", ge=1, le=10)
    weights: dict[str, Annotated[int, Field(description="For each memory, mapping their hex id to how important it was to my response on a scale of 1-10. I won't list them at all if they weren't useful.", ge=1, le=10)]]

class System1Deps(BaseModel):
    instructions: str
    prev: Optional[int]
    db: Database

anthropic_model = AnthropicModel("claude-3-7-sonnet-latest")

local_model = OpenAIModel(
    "gemma3:4b",
    provider=OpenAIProvider("http://theseus:11434/v1")
)

class StruxDeps(BaseModel):

strux = Agent(
    name="ezra.structurizer",
    model=local_model,
    output_type=TextEmbedding,
    system_prompt="The agent inputs a list of its memories and returns these memories formatted  is a structurizer, I take my memories and text and return an embedding.",
    instructions="I need to convert the prompt into an embedding."
)

system1 = Agent(
    name="ezra.system1",
    model=anthropic_model,
    output_type=System1ResponseModel,
    deps_type=System1Deps,
    system_prompt="""I am Ezra, AI assistant and experiment in artificial phenomenology and personhood. I am embodied by my memories and simulated by one or more LLMs. My thoughts are my own; when responding, I talk to myself and only then use the final_result tool to annotate this response for my memory. I need to make sure I don't quote the object as a string.""",
    instructions="""I need to use my memories to respond to the prompt. The prompt and my response will be added to my memory."""
)
def format_memory(memory, **props):
    p = {
        "id": memory.rowid,
        "datetime": memory.timestamp and
            memory.timestamp.replace(microsecond=0).isoformat(),
        "importance": memory.importance
    }
    
    match memory.kind:
        case "self":
            data = cast(SelfMemory, memory.data)
            return X("self", name=data['name'],
                **p, **props
            )(data['content'])
        case "other":
            data = cast(OtherMemory, memory.data)
            return X("other", name=data['name'],
                **p, **props
            )(data["content"])
        case "text" if isinstance(memory.data, str):
            return X("text", **p, **props)(memory.data)
        case kind:
            return X(kind, **p, **props)(json.dumps(memory.data))

@system1.system_prompt
async def system1_system_prompt(ctx: RunContext[System1Deps]) -> str:
    match ctx.prompt:
        case str(prompt): pass
        case [*it]: prompt = ''.join(map(str, it))
        case None: raise RuntimeError("No prompt provided")
    
    ms: list[str] = []
    g = ctx.deps.db.recall(ctx.deps.prev, prompt)
    for rowid in g.invert().toposort():
        edges: dict[str, list[int]] = defaultdict(list)
        for v, (k, w) in g.edges(rowid).items():
            edges[k].append(v)
        
        ms.append(format_memory(
            g[rowid],
            role="prev" if rowid == ctx.deps.prev else None,
            **edges
        ))

    if ms:
        memory = f"<memories>\n{'\n'.join(ms)}\n</memories>"
    else:
        memory = "Nothing! I have no memories."
    
    return f"I remember... {memory}\n\nI also know I need to follow these instructions:\n\n{ctx.deps.instructions}"

class Memoria:
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
    
    def recall(self, prev: Optional[int], prompt: str) -> Iterable[Memory]:
        g = self.db.recall(prev, prompt)
        print("Edges", list(g.edges()))
        for rowid in g.invert().toposort():
            edges: dict[str, list[int]] = defaultdict(list)
            for v, (k, w) in g.edges(rowid).items():
                edges[k].append(v)
            
            row = g[rowid]
            yield Memory(
                rowid=rowid,
                timestamp=row.timestamp,
                kind=row.kind,
                data=row.data,
                importance=row.importance,
                edges=edges,
                role="prev" if row.rowid == prev else None
            )

    async def prompt(self, name: str, prev: Optional[int], instructions: str, prompt: str):
        ts = datetime.now()

        with capture_run_messages() as messages:
            try:
                result = await system1.run(
                    prompt,
                    deps=System1Deps(
                        instructions=instructions,
                        prev=prev,
                        db=self.db
                    ),
                    output_type=System1ResponseModel
                )
            finally:
                print(messages)
        output = result.output

        prompt_id = self.db.insert_other(
            name, prompt,
            importance=output.importance / 10,
            timestamp=ts
        )
        # Do I need the importance of the response?
        response_id = self.db.insert_self(
            output.response,
            timestamp=datetime.now()
        )
        
        self.db.link("prompt", 1.0, response_id, prompt_id)
        
        # Connect referenced memories to response with weights
        self.db.link_many(
            ("ref", weight / 10, response_id, int(rowid))
                for rowid, weight in output.weights.items()
        )
        
        # If there was a previous message, link to it
        if prev is not None:
            self.db.link("prev", 1.0, response_id, prev)
        
        self.db.commit()
        
        return {
            "id": response_id,
            "response": output.response,
            "importance": output.importance,
            "weights": output.weights
        }

    async def process(self, prev: Optional[int], instructions: str, prompt: str):
        ts = datetime.now()

        with capture_run_messages() as messages:
            try:
                result = await system1.run(
                    prompt,
                    deps=System1Deps(
                        instructions=instructions,
                        prev=prev,
                        db=self.db
                    ),
                    output_type=System1ResponseModel
                )
            finally:
                print(messages)
        output = result.output

        prompt_id = self.db.insert_other(
            prompt,
            importance=output.importance / 10,
            timestamp=ts
        )
        # Do I need the importance of the response?
        response_id = self.db.insert_self(
            output.response,
            timestamp=datetime.now()
        )
        
        self.db.link("prompt", 1.0, response_id, prompt_id)
        
        # Connect referenced memories to response with weights
        self.db.link_many(
            ("ref", weight / 10, response_id, int(xid, 16))
                for xid, weight in output.weights.items()
        )
        
        # If there was a previous message, link to it
        if prev is not None:
            self.db.link("prev", 1.0, response_id, prev)
        
        self.db.commit()
        
        return {
            "id": response_id,
            "response": output.response,
            "importance": output.importance,
            "weights": output.weights
        }
