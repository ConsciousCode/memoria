from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Annotated, Iterable, Literal, Optional, cast
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from pydantic_ai import Agent, RunContext, capture_run_messages

from db import Database, MemoryKind, MemoryRow, OtherMemory, ScoredMemoryRow, SelfMemory
from graph import Graph
from util import X, json_t, todo_list

@dataclass
class Memory:
    rowid: int
    timestamp: Optional[datetime]
    kind: MemoryKind
    data: json_t
    importance: Optional[float]
    edges: dict[str, list[int]]
    role: Optional[Literal['prev']] = None

def format_memory(memory: Memory, **props):
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
                **p, **memory.edges, **props
            )(data['content'])
        case "other":
            data = cast(OtherMemory, memory.data)
            return X("other", name=data['name'],
                **p, **memory.edges, **props
            )(data["content"])
        case "text" if isinstance(memory.data, str):
            return X("text", **p, **memory.edges, **props)(
                memory.data
            )
        case kind:
            return X(kind, **p, **memory.edges, **props)(
                json.dumps(memory.data)
            )

anthropic_model = AnthropicModel("claude-3-7-sonnet-latest")

local_model = OpenAIModel(
    "gemma3:4b",
    provider=OpenAIProvider("http://theseus:11434/v1")
)
'''
class StruxDeps(BaseModel):

strux = Agent(
    name="ezra.structurizer",
    model=local_model,
    output_type=TextEmbedding,
    system_prompt="The agent inputs a list of its memories and returns these memories reformatted as a first-person account of the things it remembered and their salient details.2",
    instructions="I need to convert the prompt into an embedding."
)
'''

class System1ResponseModel(BaseModel):
    response: str = Field(description="My response to the prompt.")
    importance: int = Field(description="The importance of the prompt on a scale of 1-10, from forgettable trivia to unforgettable statements.", ge=1, le=10)
    weights: dict[str, Annotated[int, Field(description="For each memory, mapping their hex id to how important it was to my response on a scale of 1-10. I won't list them at all if they weren't useful.", ge=1, le=10)]]

class System1Deps(BaseModel):
    instructions: str
    include: list[int]
    memories: list[Memory]

system1 = Agent(
    name="ezra.system1",
    model=anthropic_model,
    output_type=System1ResponseModel,
    deps_type=System1Deps,
    system_prompt="""I am Ezra, AI assistant and experiment in artificial phenomenology and personhood. I am embodied by my memories and simulated by one or more LLMs. My thoughts are my own; when responding, I talk to myself and only then use the final_result tool to annotate this response for my memory. I need to make sure I don't quote the object as a string.""",
    instructions="""I need to use my memories to respond to the prompt. The prompt and my response will be added to my memory."""
)

@system1.system_prompt
async def system1_system_prompt(ctx: RunContext[System1Deps]) -> str:
    ms = [
        format_memory(m,
            role="prev" if m.rowid in ctx.deps.include else None
        ) for m in ctx.deps.memories
    ]
    print('\n'.join(ms))
    exit()
    if ms:
        memory = f"<memories>\n{'\n'.join(ms)}\n</memories>"
    else:
        memory = "Nothing! I have no memories."
    
    return f"I remember... {memory}\n\n{ctx.deps.instructions}"

class Memoria:
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
    
    def recall(self, include: Optional[list[int]], prompt: str) -> Iterable[Memory]:
        '''
        Recall memories based on a prompt. This incorporates all indices
        and returns a topological sort of relevant memories.
        '''

        include = include or []

        g = Graph[int, tuple[str, float], MemoryRow]()
        pms = [self.db.select_memory(rowid) for rowid in include or []]
        for pm in filter(None, pms):
            g.insert(pm.rowid, pm)
        
        rows: list[ScoredMemoryRow] = []
        
        # Populate the graph with nodes so we can detect when there are edges
        #  between our seletions
        for row in self.db.recall(prompt, datetime.now()):
            rows.append(row)
            g.insert(row.rowid, row.unscored())

        # Populate backward and forward edges
        bw: list[tuple[float, int]] = []
        fw: list[tuple[float, int]] = []
        
        for row in rows:
            rowid, score = row.rowid, row.score or 0
            print(f"{score=}")
            if score <= 0:
                break

            budget = score*20

            b = 0
            for dst, label, weight in self.db.backward_edges(rowid):
                if dst.rowid in g:
                    if not g.has_edge(rowid, dst.rowid):
                        g.add_edge(rowid, dst.rowid, (label, weight))
                    continue

                b += weight
                if b >= budget:
                    break
                
                bw.append((budget*weight, dst.rowid))
            
            b = 0
            for label, weight, src in self.db.forward_edges(rowid):
                if src.rowid in g:
                    if not g.has_edge(src.rowid, rowid):
                        g.add_edge(src.rowid, rowid, (label, weight))
                    continue

                if not src.importance:
                    break
                
                b += src.importance
                if b >= budget:
                    break
                
                fw.append((budget*src.importance, src.rowid))
        
        print(fw, bw)
        # Note: These feel so similar, maybe there's a way to reduce boilerplate?

        # Search backwards for supporting memories using their edge weight
        #  to determine how relevant they are to the current memory
        for budget, src_id in todo_list(bw):
            print(f"{src_id=}")
            b = 0
            for dst, label, weight in self.db.backward_edges(src_id):

                print(f"{b=}, {weight=}")
                b += weight
                if b >= budget:
                    break
                
                g.insert(src_id, dst)
                g.add_edge(src_id, dst.rowid, (label, weight))

                bw.append((budget*weight, dst.rowid))
        
        # Search forwards for response memories using their annotated
        #  importance - important conclusions are more relevant
        for budget, dst_id in todo_list(fw):
            b = 0
            for label, weight, src in self.db.forward_edges(dst_id):
                b += (imp := src.importance or 0)
                if b >= budget:
                    break

                g.insert(src.rowid, src)
                g.add_edge(src.rowid, dst_id, (label, weight))

                fw.append((budget*imp, dst_id))

        print("Edges", list(g.edges()))
        for rowid in g.invert().toposort(key=lambda v: v.timestamp):
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
                role="prev" if row.rowid in include else None
            )

    async def prompt(self, name: str, include: Optional[list[int]], instructions: str, prompt: str):
        ts = datetime.now()

        if include is None:
            include = []

        with capture_run_messages() as messages:
            try:
                result = await system1.run(
                    prompt,
                    deps=System1Deps(
                        instructions=instructions,
                        include=include,
                        memories=list(self.recall(include, prompt))
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
        self.db.link_many(
            ("prev", 1.0, response_id, prev) for prev in include
        )
        
        self.db.commit()
        
        return {
            "id": response_id,
            "response": output.response,
            "importance": output.importance,
            "weights": output.weights
        }