import json
from typing import Annotated, Any, Iterable, Literal, Optional, cast, overload
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent, RunContext, capture_run_messages

import cid

from db import Database, Edge, MemoryKind, MemoryRow, OtherMemory, SelfMemory, Memory
from graph import Graph
from util import X, json_t, todo_list

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
    weights: dict[str, Annotated[int, Field(description="For each memory, mapping their hex id to how important it was to my response on a scale of 1-10. I won't list them at all if they weren't useful.", ge=1, le=10)]]

class System1Deps(BaseModel):
    instructions: str
    #include: list[int]
    memories: list[str]

system1 = Agent(
    name="ezra.system1",
    model=anthropic_model,
    output_type=System1ResponseModel,
    deps_type=System1Deps,
    system_prompt="""I am Ezra, AI assistant and experiment in artificial phenomenology and personhood. I am embodied by my memories and simulated by one or more LLMs. My thoughts are my own; when responding, I talk to myself and only then use the final_result tool to annotate this response for my memory. I need to make sure I don't quote the object as a string.""",
    instructions="""I need to use my memories to respond to the prompt. The prompt and my response will be added to my memory."""
)

def localize_memory(g: Graph[int, tuple[str, float], MemoryRow]) -> Iterable[tuple[Optional[int], MemoryRow, dict[str, int]]]:
    '''
    Serialize the memory graph into a sequence of memories with localized
    references and edges.
    '''
    gv = g.invert()
    refs: dict[int, int] = {} # rowid: ref index

    for rowid in gv.toposort(key=lambda v: v.timestamp):
        # Only include ids if they have references
        if gv.edges(rowid):
            ref = refs[rowid] = len(refs)
        else:
            ref = None
        
        # We don't have to check if rowid is in refs because of toposort
        yield ref, gv[rowid], {
            e: refs[rowid]
                for rowid, (e, w) in gv.edges(rowid).items()
        }

def format_memory(ref: Optional[int], memory: MemoryRow, edges: dict[str, Any]) -> str:
    '''Render memory for the context.'''

    p = {
        "id": ref,
        "datetime": memory.timestamp and datetime
            .fromtimestamp(memory.timestamp)
            .replace(microsecond=0)
            .isoformat(),
        "importance": memory.importance,
        **edges
    }
    
    match memory.kind:
        case "self":
            data = cast(SelfMemory, memory.data)
            return X("self", name=data.name, **p)(data.content)
        case "other":
            data = cast(OtherMemory, memory.data)
            return X("other", name=data.name, **p)(data.content)
        case "text" if isinstance(memory.data, str):
            return X("text", **p)(memory.data)
        case kind:
            return X(kind, **p)(json.dumps(memory.data))

@system1.system_prompt
async def system1_system_prompt(ctx: RunContext[System1Deps]) -> str:
    if ms := ctx.deps.memories:
        memory = f"<memories>\n{'\n'.join(ms)}\n</memories>"
    else:
        memory = "Nothing! I have no memories."
    
    return f"I remember... {memory}\n\n{ctx.deps.instructions}"

class RecallConfig(BaseModel):
    importance: Optional[float]=None
    recency: Optional[float]=None
    fts: Optional[float]=None
    vss: Optional[float]=None
    k: Optional[int]=None

class Memoria:
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
    
    @overload
    def append(self,
        kind: Literal['self'],
        data: SelfMemory,
        timestamp: Optional[datetime],
        description: Optional[str] = None,
        importance: Optional[float] = None,
        edges: dict[str, list[Edge]] = {}
    ) -> Memory: ...
    @overload
    def append(self,
        kind: Literal['other'],
        data: OtherMemory,
        timestamp: Optional[datetime],
        description: Optional[str] = None,
        importance: Optional[float] = None,
        edges: dict[str, list[Edge]] = {}
    ) -> Memory: ...

    def append(self,
            kind: MemoryKind,
            data: Any,
            timestamp: Optional[datetime],
            description: Optional[str] = None,
            importance: Optional[float] = None,
            edges: dict[str, list[Edge]] = {}
        ) -> Memory:
        '''
        Append a memory to the sona file.
        '''

        match kind:
            case "self":
                return self.db.insert_memory(
                    "self", data, description or data, None, timestamp, edges
                )
            
            case "other":
                return self.db.insert_memory(
                    "other",
                    data,
                    description or data['content'],
                    importance,
                    timestamp,
                    edges
                )
            
            case _:
                return self.db.insert_memory(
                    kind, data, description or data, importance, timestamp, edges
                )
    
    def recall(self,
            prompt: str,
            include: Optional[list[cid.CIDv1]]=None,
            timestamp: Optional[datetime]=None,
            config: Optional[RecallConfig]=None
        ) -> Graph[int, tuple[str, float], MemoryRow]:
        '''
        Recall memories based on a prompt. This incorporates all indices
        and returns a topological sort of relevant memories.
        '''
        if include is None:
            include = []
        g = Graph[int, tuple[str, float], MemoryRow]()
        
        for cid in include or []:
            if pm := self.db.select_memory(cid):
                g.insert(pm.rowid, pm)
        
        rows: list[tuple[MemoryRow, float]] = []
        
        if config:
            importance = config.importance
            recency = config.recency
            fts = config.fts
            vss = config.vss
            k = config.k
        else:
            importance = recency = fts = vss = k = None

        memories = self.db.recall(
            prompt, timestamp, importance, recency, fts, vss, k
        )
        # Populate the graph with nodes so we can detect when there are edges
        #  between our seletions
        for rs in memories:
            row, score = rs
            rows.append(rs)
            g.insert(row.rowid, row)

        # Populate backward and forward edges
        bw: list[tuple[float, int]] = []
        fw: list[tuple[float, int]] = []
        
        for rs in rows:
            row, score = rs
            if score <= 0:
                break
            rowid = row.rowid

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
        
        # Note: These feel so similar, maybe there's a way to reduce boilerplate?

        # Search backwards for supporting memories using their edge weight
        #  to determine how relevant they are to the current memory
        for budget, src_id in todo_list(bw):
            b = 0
            for dst, label, weight in self.db.backward_edges(src_id):
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
        
        return g

    async def prompt(self, name: str, include: Optional[list[cid.CIDv1]], instructions: str, prompt: str):
        if include is None:
            include = []
        ts = datetime.now()

        g = self.recall(prompt, include)

        # Serialize the memory graph with localized references and edges.
        gv = g.invert()
        refs: dict[int, tuple[int, MemoryRow]] = {} # rowid: ref index
        memories: list[str] = []

        for rowid in gv.toposort(key=lambda v: v.timestamp):
            # Only include ids if they have references
            if gv.edges(rowid):
                ref = len(refs)
                refs[rowid] = (ref, gv[rowid])
            else:
                ref = None
            
            memories.append(format_memory(
                ref, gv[rowid], {
                    e: refs[rowid]
                        for rowid, (e, w) in gv.edges(rowid).items()
                }
            ))

        with capture_run_messages() as messages:
            try:
                result = await system1.run(
                    prompt,
                    deps=System1Deps(
                        instructions=instructions,
                        memories=memories
                    ),
                    output_type=System1ResponseModel
                )
            finally:
                print(messages)
        output = result.output

        p = self.db.insert_memory(
            "other", {
                "name": name,
                "content": prompt
            },
            prompt, None, ts
        )
        # Do I need the importance of the response?
        r = self.db.insert_memory(
            "self",
            output.response,
            output.response,
            timestamp=datetime.now(),
            edges={
                "prompt": [Edge(1.0, p.cid)],
                "ref": [
                    Edge(weight / 10, cid.from_bytes(refs[int(ref)][1].cid))
                        for ref, weight in output.weights.items()
                ],
                "prev": [
                    Edge(1.0, cid) for cid in include
                ]
            }
        )
        
        self.db.commit()
        
        return {
            "id": r,
            "response": output.response,
            "weights": output.weights
        }