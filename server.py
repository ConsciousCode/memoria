from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
import json
import re
from typing import Annotated, Iterable, Literal, Optional, Sequence, cast
from uuid import UUID

from fastapi import FastAPI, Header, Request, Response
from fastmcp.exceptions import ResourceError, ToolError
from mcp import CreateMessageResult, SamplingMessage
from mcp.types import ModelHint, ModelPreferences, PromptMessage, Role, TextContent
from fastmcp import Context, FastMCP
from fastmcp.prompts.prompt import Message
from pydantic import BaseModel, Field

from ipld import dagcbor, CIDv1
from db import Edge
from memoria import Database, Memoria
from models import AnyMemory, Chatlog, CompleteMemory, DraftMemory, IncompleteMemory, Memory, MemoryDAG, NodeMemory, PartialMemory, RecallConfig, SampleConfig, StopReason

ANNOTATE_EDGES = """\
Respond ***ONLY*** with a JSON object with the following schema:
{
    "relevance": {
        "<index>": score
        "1": 8,
        "2": 5,
    },
    "importance": {
        "novelty": score,
        "intensity": score,
        "future": score,
        "personal": score,
        "saliency": score
    }
}
The "relevance" object identifies which memories ([ref:index] or [final:index]) are relevant to the [response]. Each key is a quoted number (the index indicated by the tag) and the score is a number 1-10 indicating how relevant the memory is to the response. Only the memories that are relevant to the response are included.

The "importance" object scores the assistant's response according to the following dimensions (0-10):
- "novelty": How unique or original the response is.
- "intensity": How emotionally impactful the response is.
- "future": How useful this response might be in future conversations.
- "personal": How relevant the response is to the *assistant's* personal context.
- "saliency": How attention-grabbing or notable the response is.

DO NOT write comments.
DO NOT write anything EXCEPT JSON.
"""

def date_time(ts: Optional[datetime|float]) -> tuple[int, datetime]:
    '''Convert a mixed-type timestamp to a UNIX timestamp and datetime.'''
    if isinstance(ts, datetime):
        dt = ts
        ts = ts.timestamp()
    elif ts:
        dt = datetime.fromtimestamp(ts)
    else:
        dt = datetime.now()
        ts = dt.timestamp()
    
    return int(ts), dt

_gmemoria: Optional[tuple[Memoria, int]] = None

@contextmanager
def lifespan():
    '''Lifespan context for the FastAPI app.'''
    global _gmemoria
    if _gmemoria:
        m, count = _gmemoria
        _gmemoria = (m, count + 1)
        yield m
    else:
        with Database("private/memoria.db", "files") as db:
            m = Memoria(db)
            _gmemoria = (m, 1)
            yield m

    m, count = _gmemoria
    _gmemoria = (m, count - 1) if count > 1 else None

@asynccontextmanager
async def fastapi_lifespan(app: FastAPI):
    '''Lifespan context for the FastAPI app.'''
    with lifespan() as memoria:
        app.state.lifespan_context = memoria
        yield memoria

@asynccontextmanager
async def mcp_lifespan(server: FastMCP):
    '''Lifespan context for the FastAPI app.'''
    with lifespan() as memoria:
        yield memoria

mcp = FastMCP("memoria",
    """Coordinates a "sona" representing a cohesive identity and memory.""",
    lifespan=mcp_lifespan,
    #log_level="DEBUG"
)
mcp_app = mcp.http_app()
app = FastAPI(lifespan=mcp_app.lifespan)
ipfs = FastAPI()
app.mount("/ipfs", ipfs)

def mcp_context(ctx: Context|Request) -> Memoria:
    '''Get memoria from the FastAPI context.'''
    if isinstance(ctx, Context):
        return ctx.request_context.lifespan_context
    else:
        return ctx.app.state.lifespan_context

## IPFS trustless gateway ##

@ipfs.get("/bafkqaaa")
def get_ipfs_empty():
    '''Empty block for PING'''
    return Response()

@ipfs.get("/{path:path}")
def get_ipfs(
        path: str,
        request: Request,
        accept: Annotated[Optional[list[str]], Header()] = None
    ):
    if accept is None:
        accept = []
    cid = CIDv1(path)
    memoria = mcp_context(request)
    if ob := memoria.lookup_memory(cid):
        if "application/cbor" in accept:
            return dagcbor.marshal(ob)
        return ob
    if ob := memoria.lookup_act(cid):
        if "application/cbor" in accept:
            return dagcbor.marshal(ob)
        return ob
    # ipfs_api.lookup(cid) # TODO
    return Response(
        status_code=404,
        content=f"Memory or ACT not found for CID {cid}"
    )

## RESTful API ##

@app.get("/memories/list")
def list_memories(
        request: Request,
        page: Annotated[
            int,
            Header(description="Page number to return.")
        ] = 1,
        perpage: Annotated[
            int,
            Header(description="Number of messages to return per page.")
        ] = 100,
        accept: Annotated[Optional[list[str]], Header()] = None
    ):
    '''List messages in the Memoria system.'''
    memoria = mcp_context(request)
    messages = memoria.list_messages(page, perpage)
    accept = accept or []
    if "application/cbor" in accept:
        return dagcbor.marshal(messages)
    return messages

@app.get("/sona/{uuid}")
def get_sona(
        request: Request,
        uuid: UUID|str,
        accept: Annotated[Optional[list[str]], Header()] = None
    ):
    '''Get a sona by UUID.'''
    try: uuid = UUID(uuid) # type: ignore
    except (ValueError, TypeError):
        pass

    memoria = mcp_context(request)
    if sona := memoria.find_sona(uuid):
        accept = accept or []
        if "application/cbor" in accept:
            return dagcbor.marshal(sona)
        return sona.human_json()
    return Response(
        status_code=404,
        content=f"Sona with UUID {uuid} not found."
    )

@app.get("/sonas/list")
def list_sonas(
        request: Request,
        page: Annotated[
            int,
            Header(description="Page number to return.")
        ] = 1,
        perpage: Annotated[
            int,
            Header(description="Number of sonas to return per page.")
        ] = 100,
        accept: Annotated[Optional[list[str]], Header()] = None
    ):
    '''List sonas in the Memoria system.'''
    memoria = mcp_context(request)
    sonas = memoria.list_sonas(page, perpage)
    accept = accept or []
    if "application/cbor" in accept:
        return dagcbor.marshal(sonas)
    return [sona.human_json() for sona in sonas]

## MCP ##

# Nothing can be mounted after this
app.mount("", mcp_app)

@mcp.resource("ipfs://{cid}")
def ipfs_resource(ctx: Context, cid: CIDv1):
    '''IPFS resource handler.'''
    memoria = mcp_context(ctx)
    if (m := memoria.lookup_memory(cid)) is None:
        raise ResourceError("Memory not found")
    
    if m.data.kind != "file":
        raise ResourceError("Memory is not a file")
    
    return m.data.content

@mcp.resource("memoria://sona/{uuid}")
def sona_resource(ctx: Context, uuid: UUID):
    '''Sona resource handler.'''
    if m := mcp_context(ctx).find_sona(uuid):
        return m
    raise ResourceError("Sona not found")

@mcp.resource("memoria://memory/{cid}")
def memory_resource(ctx: Context, cid: str):
    '''Memory resource handler.'''
    if m := mcp_context(ctx).lookup_memory(CIDv1(cid)):
        return m
    raise ResourceError("Memory not found")

@mcp.tool(
    annotations=dict(
        idempotentHint=True,
        openWorldHint=False
    )
)
def insert_memory(
        ctx: Context,
        sona: Annotated[
            Optional[UUID|str],
            Field(description="Sona to push the memory to.")
        ],
        memory: Annotated[
            Memory,
            Field(description="Memory to insert.")
        ],
        importance: Annotated[
            Optional[float],
            Field(description="Importance of the memory, from 0 to 1.")
        ] = None,
        index: Annotated[
            Optional[str],
            Field(description="Plaintext indexing field for the memory if available.")
        ] = None
    ):
    '''Insert a new memory into the sona.'''
    return mcp_context(ctx).insert(memory, importance, sona)

@mcp.tool(
    annotations=dict(
        openWorldHint=False
    )
)
def act_push(
        ctx: Context,
        sona: Annotated[
            UUID|str,
            Field(description="Sona to push the memory to.")
        ],
        memories: Annotated[
            list[Edge[CIDv1]],
            Field(description="Additional memories to include in the ACT, keyed by label.")
        ]
    ):
    '''
    Insert a new memory into the sona, formatted for an ACT
    (Autonomous Cognitive Thread).
    '''
    if u := mcp_context(ctx).act_push(sona, memories):
        return u
    raise ToolError("Sona not found or prompt memory not found.")

@mcp.tool(
    annotations=dict(
        openWorldHint=False
    )
)
def act_stream(
        ctx: Context,
        sona: Annotated[
            UUID|str,
            Field(description="Sona to push the memory to.")
        ],
        delta: Annotated[
            Optional[str],
            Field(description="Delta to append to the memory.")
        ],
        model: Annotated[
            Optional[str],
            Field(description="Model which generated this delta.")
        ] = None,
        stop_reason: Annotated[
            Optional[StopReason],
            Field(description="Reason for stopping the stream, if applicable.")
        ] = None,
    ):
    '''
    Stream tokens from the LLM to the sona to be committed to memory in case
    the LLM is interrupted or the session ends unexpectedly.
    '''
    memoria = mcp_context(ctx)
    return memoria.act_stream(sona, delta, model, stop_reason)

@mcp.tool(
    annotations=dict(
        readOnlyHint=True,
        openWorldHint=False,
    )
)
def recall(
        ctx: Context,
        sona: Annotated[
            Optional[str],
            Field(description="Sona to focus the recall on.")
        ],
        prompt: Annotated[
            DraftMemory,
            Field(description="Prompt to base the recall on.")
        ],
        config: Annotated[
            RecallConfig,
            Field(description="Configuration for how to weight memory recall.")
        ] = RecallConfig()
    ):
    '''
    Recall memories related to the prompt, including relevant included memories
    and their dependencies.
    '''
    return dict(mcp_context(ctx).recall(sona, prompt, config).items())

def build_tags(tags: list[str], timestamp: Optional[float|datetime]) -> str:
    if timestamp is not None:
        if not isinstance(timestamp, datetime):
            timestamp = datetime.fromtimestamp(timestamp)
        tags.append(timestamp.replace(microsecond=0).isoformat())
    return f"[{'\t'.join(tags)}]\t"

def memory_to_message(ref: int, deps: list[int], memory: AnyMemory, final: bool=False) -> tuple[Role, str]:
    '''Render memory for the context.'''
    
    tags = []
    if final:
        tags.append("final")
    tags.append(
        f"ref:{ref}" +
        (f" -> {','.join(map(str, deps))}" if deps else "")
    )

    if (ts := memory.timestamp) is not None:
        ts = datetime.fromtimestamp(ts)
    
    match memory.data:
        case Memory.SelfData():
            if name := memory.data.name:
                tags.append(f"name:{name}")
            if sr := memory.data.stop_reason:
                if sr != "finish":
                    tags.append(f"stop_reason:{sr}")
            return (
                "assistant",
                build_tags(tags, ts) +
                    ''.join(p.content for p in memory.data.parts)
            )
        case Memory.TextData():
            tags.append("kind:raw_text")
            return (
                "user",
                build_tags(tags, ts) + memory.data.content
            )
        case Memory.OtherData():
            if name := memory.data.name:
                tags.append(f"name:{name}")
            return (
                "user",
                build_tags(tags, ts) + memory.data.content
            )
            '''
        case "file":
            return ConvoMessage(
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri=AnyUrl(f"memory://{memory.cid}"),
                        mimeType=memory.data.mimeType or
                            "application/octet-stream",
                        blob=memory.data.content
                    )
                ),
                role="user"
            )
        '''
        
        case _:
            raise ValueError(f"Unknown memory kind: {memory.data.kind}")

class ConvoMessage(BaseModel):
    '''
    I hate everything about this and would like nothing more than some
    other approach to serializing the conversation that doen't involve
    *REIFYING THE ENTIRE FUCKING ITERATOR STATE*
    '''
    g: MemoryDAG
    ref: int
    refs: dict[CIDv1, int]
    memory: PartialMemory
    
    def message(self, include_final: bool = False):
        cid = self.memory.cid
        return memory_to_message(
            self.ref,
            [self.refs[dst] for dst, _ in self.g.edges(cid)],
            self.memory,
            final=include_final and (self.ref >= len(self.refs))
        )

    def prompt_message(self, include_final: bool = False) -> PromptMessage:
        role, content = self.message(include_final)
        return PromptMessage(
            role=role,
            content=TextContent(
                type="text",
                text=content
            )
        )
    
    def sampling_message(self, include_final: bool = False) -> SamplingMessage:
        role, content = self.message(include_final)
        return SamplingMessage(
            role=role,
            content=TextContent(
                type="text",
                text=content
            )
        )

def serialize_dag(g: MemoryDAG, refs: Optional[dict[CIDv1, int]]=None) -> Iterable[ConvoMessage]:
    '''Convert a memory DAG to a conversation.'''
    if refs is None:
        refs = {}
    for cid in g.invert().toposort(key=lambda v: v.timestamp):
        # We need ids for each memory so their edges can be annotated later
        ref = refs[cid] = len(refs) + 1
        #yield ref, g[cid]
        yield ConvoMessage(
            g=g,
            ref=ref,
            refs=refs,
            memory=g[cid]
        )

@mcp.prompt()
def act_next(
        ctx: Context,
        sona: Annotated[
            str,
            Field(description="Sona to process, either a name or UUID.")
        ],
        timestamp: Annotated[
            Optional[float],
            Field(description="Timestamp to use for the recall recency. If `null`, uses the current time.")
        ] = None,
        chat_config: Annotated[
            RecallConfig,
            Field(description="Configuration for how to weight memory recall.")
        ] = RecallConfig()
    ) -> Optional[list[PromptMessage]]:
    '''Get the prompt for the next step of an ACT (Autonomous Cognitive Thread).'''
    memoria = mcp_context(ctx)
    g = memoria.act_next(
        sona,
        timestamp or datetime.now().timestamp(),
        chat_config
    )
    if g is None:
        return None

    return [
        m.prompt_message(include_final=True)
            for m in serialize_dag(g)
    ]

def parse_edge_ref(k: str):
    # Some models are stupid cunts which can't follow instructions.
    if m := re.search(r"\d+", k):
        return int(m[0])

async def annotate_edges(
        ctx: Context,
        memories: Sequence[CompleteMemory],
        prompt: Optional[NodeMemory],
        response: NodeMemory,
        annotate_config: SampleConfig
    ) -> tuple[dict[int, float], float]:
    '''Use sampling with a faster model to annotate which memories are connected and by how much.'''

    refs: dict[CIDv1, int] = {}
    context = []
    for m in memories:
        refs[m.cid] = index = len(context) + 1
        c = {
            "index": index,
            "role": {
                "self": "assistant",
                "other": "user"
            }[m.data.kind],
            "content": m.data.document()
        }
        if m.edges:
            c['deps'] = [refs[e.target] for e in m.edges]
        context.append(c)
    
    if prompt:
        context.append({
            "index": len(context) + 1,
            "role": "user",
            "content": prompt.data.document(),
        })
    context.append({
        "index": len(context) + 1,
        "deps": None,
        "role": "assistant",
        "content": response.data.document()
    })
    
    result = await ctx.sample(
        [json.dumps(context, indent=2)],
        system_prompt=ANNOTATE_EDGES,
        temperature=0,
        max_tokens=None,
        model_preferences=annotate_config.model_preferences
    )
    if not isinstance(result, TextContent):
        raise ValueError(
            f"Edge annotation response must be text, got {type(result)}: {result}"
        )
    try:
        assert (m := re.match(r"^(?:```(?:json)?)?([\s\S\n]+?)(?:```)?$", result.text))
        data = json.loads(m[1])
        
        if not isinstance(data, dict):
            raise ValueError(f"Edge annotation response must be a JSON object, got {type(result)}: {result}")
        
        importance = data.get("importance", {})
        edges: dict[int, float] = {}
        for k, v in data.get("relevance", {}).items():
            if v < 0: continue
            if ki := parse_edge_ref(k):
                if not isinstance(v, (int, float)):
                    raise ValueError(
                        f"Edge relevance score must be a number, got {type(v)}: {v}"
                    )
                edges[ki] = v

        # Combine importance with self-weighted mean
        imp = sum(x**2 for x in importance.values()) / sum(importance.values()) / 10
        return edges, imp
    except json.JSONDecodeError as e:
        raise ValueError(f"Edge annotation response is not valid JSON: {e}") from e

@mcp.tool(
    annotations=dict(
        openWorldHint=False,
    )
)
async def chat(
        ctx: Context,
        sona: Annotated[
            Optional[str],
            Field(description="Sona to focus the chat on.")
        ],
        prompt: Annotated[
            IncompleteMemory,
            Field(description="Prompt for the chat. If `null`, use only the included memories.")
        ],
        recall_config: Annotated[
            RecallConfig,
            Field(description="Configuration for how to weight memory recall.")
        ] = RecallConfig(),
        chat_config: Annotated[
            SampleConfig,
            Field(description="Configuration for how to sample the response.")
        ] = SampleConfig(),
        annotate_config: Annotated[
            SampleConfig,
            Field(description="Configuration for how to sample the response for edge annotation.")
        ] = SampleConfig()
    ):
    '''
    Single-turn conversation returning the response. This is committed to memory.
    '''
    memoria = mcp_context(ctx)
    ts, dt = date_time(prompt.timestamp)
    g = memoria.recall(sona, prompt, recall_config)

    refs: dict[int, CIDv1] = {}
    chat_memories: list[PartialMemory] = []
    chat_messages: list[SamplingMessage] = []
    
    for m in serialize_dag(g):
        refs[m.ref] = m.memory.cid
        chat_memories.append(m.memory)
        chat_messages.append(m.sampling_message(include_final=not prompt))
    
    lastref = len(refs) + 1
    if prompt:
        # Annotate the final user message with reference tags
        chat_messages.append(
            SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=build_tags(
                        [f"final:{lastref}"], timestamp=dt
                    ) + prompt.document()
                )
            )
        )
    
    # Agent response
    response = cast(CreateMessageResult,
        await ctx.request_context.session.create_message(
            chat_messages,
            system_prompt="I'm talking to a user. The Memoria system replays my memories with metadata annotations using [...]\t prefixes, ending with [final] for the last replayed message. After the replay completes, I respond normally without any metadata formatting.",
            temperature=chat_config.temperature,
            max_tokens=chat_config.max_tokens,
            model_preferences=chat_config.model_preferences
        )
    )
    content = response.content
    assert isinstance(content, TextContent)

    # Add self memory to annotate edges
    self_memory = IncompleteMemory(
        timestamp=ts,
        data=IncompleteMemory.SelfData(
            parts=[IncompleteMemory.SelfData.Part(content=content.text)],
            stop_reason=response.stopReason
        )
    )

    rel, imp = await annotate_edges(
        ctx, chat_memories, prompt, self_memory, annotate_config
    )

    # Now that there's no risk of interruption,

    # Insert the user memory
    other_memory = prompt.complete()
    refs[lastref] = other_memory.cid
    memoria.insert(other_memory, rel.get(lastref, 0) / 10, sona)

    # Finish the response memory for insertion and return
    self_memory.edges = [
        Edge(target=target, weight=weight / 10)
            for ref, weight in rel.items()
                if (target := refs.get(ref))
    ]
    self_memory = self_memory.complete()
    memoria.insert(self_memory, imp, sona)
    return Chatlog(
        chatlog=chat_memories,
        response=self_memory
    )

@mcp.tool
async def insert_interaction(
        ctx: Context,
        sona: Annotated[
            Optional[str],
            Field(description="Sona to focus the chat on.")
        ],
        prompt: Annotated[
            Optional[CIDv1|IncompleteMemory],
            Field(description="The memory which prompted the response.")
        ],
        response: Annotated[
            IncompleteMemory,
            Field(description="The response to insert as a first-person memory.")
        ],
        recall_config: Annotated[
            RecallConfig,
            Field(description="Configuration for how to weight memory recall.")
        ] = RecallConfig(),
        annotate_config: Annotated[
            SampleConfig,
            Field(description="Configuration for how to sample for edge annotation.")
        ] = SampleConfig()
    ):
    '''
    Insert an interaction without calling the chat model. This uses sampling to
    annotate edges from the existing memory to the response as if it were a chat.

    The purpose of this is to load external memories such as LLM chatlogs while
    properly integrating them into the existing memory graph.
    '''
    memoria = mcp_context(ctx)
    g = memoria.recall(sona, response, recall_config)

    refs: dict[int, CIDv1] = {}
    chat_samples: list[PartialMemory] = []

    for m in serialize_dag(g):
        refs[m.ref] = m.memory.cid
        chat_samples.append(m.memory)
    
    lastref = len(refs) + 1
    
    if prompt:
        if isinstance(prompt, CIDv1):
            if (other_memory := memoria.lookup_memory(prompt)) is None:
                raise ToolError(f"Prompt memory {prompt} not found.")
        else:
            other_memory = prompt.complete()
        refs[lastref] = other_memory.cid
    else:
        other_memory = None

    rel, imp = await annotate_edges(
        ctx, chat_samples, other_memory, response, annotate_config
    )

    # Now that there's no risk of interruption,

    # Insert the user memory (if it exists this is a no-op)
    if other_memory:
        memoria.insert(other_memory, rel.get(lastref, 0) / 10, sona)
    
    # Finish the response memory for insertion and return
    response.edges = [
        Edge(target=target, weight=weight / 10)
            for ref, weight in rel.items()
                if (target := refs.get(ref))
    ]
    self_memory = response.complete()
    memoria.insert(self_memory, imp, sona)

    return self_memory.cid

@mcp.tool(
    annotations=dict(
        readOnlyHint=True,
        openWorldHint=False,
    )
)
async def query(
        ctx: Context,
        sona: Annotated[
            Optional[str],
            Field(description="Sona to focus the chat on.")
        ],
        prompt: Annotated[
            IncompleteMemory,
            Field(description="Prompt for the chat. If `null`, use only the included memories.")
        ],
        recall_config: Annotated[
            RecallConfig,
            Field(description="Configuration for how to weight memory recall.")
        ] = RecallConfig(),
        chat_config: Annotated[
            SampleConfig,
            Field(description="Configuration for how to sample the response.")
        ] = SampleConfig(),
    ):
    '''Single-turn conversation returning the response.'''
    memoria = mcp_context(ctx)
    g = memoria.recall(sona, prompt, recall_config)

    # Because this isn't committed to memory, we don't actually need
    # to annotate edges, and thus don't need localized references.

    chatlog: list[PartialMemory] = []
    messages: list[str|SamplingMessage] = []
    for cid in g.invert().toposort(key=lambda v: v.timestamp):
        chatlog.append(m := g[cid])
        messages.append(
            SamplingMessage(
                role="assistant" if m.data.kind == "self" else "user",
                content=TextContent(
                    type="text",
                    text=build_tags([], m.timestamp) + m.data.document()
                )
            )
        )
    
    if prompt:
        messages.append(build_tags([], prompt.timestamp) + prompt.document())

    response = await ctx.sample(
        messages,
        system_prompt="I'm talking to a user. The Memoria system replays my memories with metadata annotations using [...]\t prefixes, ending with [final] for the last replayed message. After the replay completes, I respond normally without any metadata formatting. This is being run as a query so I won't remember it.",
        temperature=chat_config.temperature,
        max_tokens=chat_config.max_tokens,
        model_preferences=chat_config.model_preferences
    )
    assert isinstance(response, TextContent)

    return Chatlog(
        chatlog=chatlog,
        response=Memory(
            timestamp=int(datetime.now().timestamp()),
            data=Memory.SelfData(
                parts=[Memory.SelfData.Part(content=response.text)]
            )
        )
    )

def main():
    import uvicorn
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    main()