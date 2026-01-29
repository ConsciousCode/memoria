#!/usr/bin/env python3

import inspect
import sys
from typing import Annotated, assert_never, cast
from uuid import UUID
from pathlib import Path
import re
import itertools

from pydantic import BaseModel, Field
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

def load_file(uuid: UUID) -> bytes|None:
    # TODO: Consider directory sharding
    return None

def convert_self(m: SelfMemory, idmap: dict[UUID, int]) -> tuple[list[ModelResponsePart], list[ToolReturnPart]]:
    parts: list[ModelResponsePart] = [
        #TextPart(content=convert_metadata(m, idmap))
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

def create_self(msgs: list[ModelResponse], edges: set[UUID]) -> Memory:
    """Create an assistant memory from LLM output."""

    return SelfMemory(
        uuid=uuid7(),
        metadata={} if metadata is None else metadata,
        edges=edges,
        parts=[memory.TextPart(content=text)]
    )

class MemoriaResult(BaseModel, extra='allow'):
    '''Core memory logged to Memoria. Allows arbitrary metadata.'''
    depends: Annotated[set[int],
        Field(description="Ids of memories this response depends on.")
    ]
    response: Annotated[str,
        Field(description="Response to send to the user.")
    ]

    __pydantic_extra__: dict[str, json_t] = Field(init=False) # pyright: ignore[reportIncompatibleVariableOverride]

WINDOW = 10

def chatbot_loop(db_path: str = "memoria.db", model_name: str = "anthropic:claude-3-5-haiku-latest"):
    """Main chatbot REPL loop."""

    # Initialize agent with current system prompt
    agent = Agent(
        model_name,
        output_type=MemoriaResult,
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
                
                print(ms)

                # Call the agent
                result = agent.run_sync(user_input, message_history=ms)
                output = result.output
                
                print(result.response)

                # Create assistant memory
                subject.add_memory(create_self(
                    output.response,
                    mapid,
                    frontmatter=output.__pydantic_extra__,
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
