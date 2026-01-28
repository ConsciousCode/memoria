#!/usr/bin/env python3
"""
Simple chatbot MVP demonstrating the memory DAG concept.

At each turn:
1. User provides input
2. Query for a foliation over the memory DAG
3. Format as markdown with frontmatter
4. Send to LLM
5. Parse response and extract metadata from frontmatter
6. Store new memories in the DAG
"""

import inspect
import sys
from typing import assert_never, cast
from uuid import UUID
from pathlib import Path
import re
import itertools

import yaml
from uuid_extension import uuid7 as _uuid7

def uuid7():
    return _uuid7().uuid7

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from memoria.util import json_t
from memoria.db import database
from memoria.subject import Subject
from memoria.config import RecallConfig
from memoria import memory
from memoria.memory import (
    Memory, OtherMemory, SelfMemory, TextPart, MemoryDAG
)

from pydantic_ai import Agent, BinaryContent, FilePart, ModelMessage, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, UserContent
from pydantic_ai.messages import (
    SystemPromptPart, UserPromptPart, TextPart, ThinkingPart,
    ToolCallPart, ToolReturnPart
)

FRONTMATTER = re.compile(r'^---\n([\s\S\n]+?)\n---([\s\S\n]*)$')

def convert_metadata(m: Memory, idmap: dict[UUID, int]):
    extra = m.__pydantic_extra__
    metadata = {
        "id": idmap[m.uuid],
        "edges": [idmap[e] for e in m.edges],
        **({} if extra is None else extra)
    }

    frontmatter = yaml.safe_dump(
        metadata, default_flow_style=False, sort_keys=False
    )
    return f"---\n{frontmatter}---"

def load_file(uuid: UUID) -> bytes|None:
    # TODO: Consider directory sharding
    return None

def convert_self(m: SelfMemory, idmap: dict[UUID, int]) -> tuple[list[ModelResponsePart], list[ToolReturnPart]]:
    parts: list[ModelResponsePart] = [
        TextPart(content=convert_metadata(m, idmap))
    ]
    tools: list[ToolReturnPart] = []

    for part in m.parts:
        match part:
            case memory.TextPart(content=content):
                parts.append(TextPart(content=content))
            
            case memory.ThinkPart(content=c, think_id=tid, signature=sig):
                parts.append(ThinkingPart(c, id=tid, signature=sig))
            
            case memory.ToolPart(name=name, args=args, result=result, call_id=cid):
                parts.append(ToolCallPart(name, args, cid))
                tools.append(ToolReturnPart(name, result, cid))
            
            case memory.FilePart(file=uuid, mimetype=mt):
                # This isn't really implemented and requires the agent generate
                # its own multimedia, probably out of scope
                if content := load_file(uuid):
                    parts.append(FilePart(BinaryContent(
                        content, media_type=mt
                    )))
                else:
                    parts.append(TextPart(
                        f"[file failed to load {mt} {uuid}]"
                    ))
            
            case x:
                assert_never(x)
    
    return parts, tools

def convert_other(m: OtherMemory, idmap: dict[UUID, int]) -> UserPromptPart:
    parts: list[UserContent] = [convert_metadata(m, idmap)]
    for part in m.parts:
        match part:
            case memory.TextPart(content=content):
                if isinstance(parts[-1], str):
                    parts[-1] += content
                else:
                    parts.append(content)
            
            case memory.FilePart(file=uuid, mimetype=mt):
                if content := load_file(uuid):
                    parts.append(BinaryContent(content, media_type=mt))
                else:
                    parts.append(f"[file failed to load {mt} {uuid}]")
            
            case x:
                assert_never(x)
    
    return UserPromptPart(parts)

def dag_to_messages(g: MemoryDAG) -> tuple[list[ModelMessage], dict[UUID, int]]:
    """
    Convert memory DAG to pydantic_ai message objects.
    Returns (system_prompts, conversation_messages).
    """
    # Create UUID -> simplified ID mapping
    idmap: dict[UUID, int] = {uuid: i for i, uuid in enumerate(g.keys(), 1)}

    # In memory tool calls and results are paired together, so we have
    # to untether them into request-response with a running list.
    pending: list[ModelRequestPart] = []
    ms: list[ModelMessage] = []

    # Sort by UUID as a tie-breaker for temporal ordering
    for uuid in g.toposort(key=lambda x: x.uuid):
        match m := g[uuid]:
            case SelfMemory():
                # Don't want to let these persist over multiple messages
                if pending:
                    ms.append(ModelRequest(pending))

                parts, tools = convert_self(m, idmap)
                ms.append(ModelResponse(parts))
                pending = cast(list[ModelRequestPart], tools)
            
            case OtherMemory():
                pending.append(convert_other(m, idmap))
                ms.append(ModelRequest(pending))
                pending = []
            
            case x:
                assert_never(x)
    
    if pending:
        ms.append(ModelRequest(pending))

    return ms, idmap

def create_other(text: str, edges: set[UUID], metadata: dict[str, json_t] | None = None) -> Memory:
    """Create a user memory from text input."""
    return OtherMemory(
        uuid=uuid7(),
        metadata={} if metadata is None else metadata,
        parts=[memory.TextPart(content=text)],
        edges=edges or set()
    )

def format_frontmatter(text: str):
    return 

def parse_frontmatter(text: str) -> tuple[dict[str, json_t], str]:
    if (m := FRONTMATTER.match(text)) is None:
        raise ValueError("No frontmatter")

    try:
        data = yaml.safe_load(m[1])
    except yaml.YAMLError:
        raise ValueError("No frontmatter") from None
    
    if not isinstance(data, dict):
        raise ValueError("Frontmatter is not a dictionary")
    
    return data, m[2]

def create_self(text: str, idmap: dict[int, UUID], frontmatter: dict[str, json_t], metadata: dict[str, json_t]|None = None) -> Memory:
    """Create an assistant memory from LLM output."""

    frontmatter.pop('id')
    if not isinstance(es := frontmatter.pop('edges'), list):
        raise TypeError("Edges are not a list")
    edges = set(idmap[e] for e in cast(list[int], es))

    return SelfMemory(
        uuid=uuid7(),
        metadata={} if metadata is None else metadata,
        edges=edges,
        parts=[memory.TextPart(content=text)]
    )

class Session[T]:
    def __init__(self, size: int):
        self.size = size
        self.history: list[T] = []
    
    def add(self, value: T):
        self.history.append(value)
    
    def get(self):
        return self.history[-self.size:]

WINDOW = 10

def chatbot_loop(db_path: str = "memoria.db", model_name: str = "anthropic:claude-3-5-haiku-latest"):
    """Main chatbot REPL loop."""

    # Initialize agent with current system prompt
    agent = Agent(
        model_name,
        system_prompt=inspect.cleandoc("""
            You are a helpful assistant. This conversation was reconstructed from a DAG of your memories, and your responses will be stored in that memory. User memories are linked to every recalled message they saw, while your memories are linked to every message you previously said was a dependency. Every response is in Markdown with frontmatter for metadata. Every response contains its id and dependencies:
            ---
            id: 9
            depends: [2, 5, 7]
            tags: [tag1, tag2]
            custom_field: value
            ---
            Your actual response here.
        """)
    )
    history: list[Memory] = []

    with database(db_path) as db:
        subject = Subject(db)
        recall_config = RecallConfig(refs=2, deps=3, memories=10)

        print("Memoria Chatbot MVP")
        print("=" * 50)
        print("Commands:")
        print("  /quit - Exit the chatbot")
        print("  /clear - Clear conversation history")
        print("=" * 50)
        print()

        while True:
            try:
                # Get user input
                if not (user_input := input("You: ").strip()):
                    continue

                # Handle commands
                if user_input == "/quit":
                    print("Goodbye!")
                    break

                if user_input == "/clear":
                    print("(History cleared - not implemented in MVP)")
                    continue

                # Construct the roots and recall
                roots = set(itertools.chain(
                    subject.list_ids(0, 5),
                    (m.uuid for m in history[-WINDOW:])
                ))
                foliation = subject.recall(roots, recall_config)
                ms, idmap = dag_to_messages(foliation)
                mapid = {i: u for u, i in idmap.items()}

                # Create user memory (after recall)
                om = create_other(
                    user_input, set(m.uuid for m in history)
                )
                idmap[om.uuid] = len(idmap)
                mapid[len(idmap)] = om.uuid
                subject.add_memory(om)
                
                fm = convert_metadata(om, idmap)

                print(ms)

                # Call the agent
                result = agent.run_sync(f"{fm}\n{user_input}", message_history=ms)
                assistant_text = result.output
                print(assistant_text)

                # Parse response for metadata
                frontmatter, content = parse_frontmatter(assistant_text)

                # Create assistant memory
                subject.add_memory(create_self(
                    content,
                    mapid,
                    frontmatter=frontmatter,
                    metadata={
                        "model": model_name
                    }
                ))

                print(f"\nAssistant: {assistant_text}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    chatbot_loop()
