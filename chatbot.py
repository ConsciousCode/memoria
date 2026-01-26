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

import os
import sys
from uuid import UUID
from pathlib import Path

from uuid_extension import uuid7 as _uuid7

def uuid7():
    return _uuid7().uuid7

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from memoria.db import database
from memoria.subject import Subject
from memoria.config import RecallConfig
from memoria.memory import (
    Memory, OtherData, SelfData, TextPart, MemoryDAG
)

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("Warning: anthropic package not installed. Install with: pip install anthropic")
    print("Falling back to echo mode (no actual LLM calls)")


def format_memory_frontmatter(uuid: UUID, memory: Memory, simplified_id: int) -> str:
    """Format a memory as markdown with frontmatter."""
    lines = ["---"]
    lines.append(f"id: {simplified_id}")
    lines.append(f"uuid: {uuid}")

    # Add edge references as simplified IDs (if we have them in context)
    if memory.edges:
        edge_ids = [str(e) for e in memory.edges]
        lines.append(f"edges: [{', '.join(edge_ids)}]")

    # Add metadata from memory data
    if memory.data.kind == "other":
        lines.append("role: user")
    elif memory.data.kind == "self":
        lines.append("role: assistant")
        if memory.data.model:
            lines.append(f"model: {memory.data.model}")
    elif memory.data.kind == "system":
        lines.append("role: system")
    elif memory.data.kind == "meta":
        lines.append("role: meta")
        for key, value in memory.data.metadata.items():
            lines.append(f"{key}: {value}")

    lines.append("---")
    lines.append("")

    # Add content
    if hasattr(memory.data, 'parts'):
        for part in memory.data.parts:
            if part.kind == "text":
                lines.append(part.content)

    return "\n".join(lines)


def format_foliation(dag: MemoryDAG) -> str:
    """Format a memory foliation as markdown with frontmatter."""
    # Create UUID -> simplified ID mapping
    uuid_to_id = {uuid: i for i, uuid in enumerate(dag.keys(), 1)}

    sections = []
    for uuid, memory in dag.items():
        simplified_id = uuid_to_id.get(uuid, 0)
        sections.append(format_memory_frontmatter(uuid, memory, simplified_id))

    return "\n\n---\n\n".join(sections)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse markdown frontmatter from text."""
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    frontmatter_text = parts[1].strip()
    content = parts[2].strip()

    metadata = {}
    for line in frontmatter_text.split("\n"):
        line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    return metadata, content


def get_recent_roots(subject: Subject, n: int = 3) -> list[UUID]:
    """Get the N most recent memory UUIDs as roots for recall."""
    roots = []
    for memory in subject.list_memories(page=1, perpage=n):
        roots.append(memory.uuid)
    return roots


def create_user_memory(text: str, edges: set[UUID] | None = None) -> Memory:
    """Create a user memory from text input."""
    return Memory(
        uuid=uuid7(),
        data=OtherData(
            parts=[TextPart(content=text)]
        ),
        edges=edges or set()
    )


def create_assistant_memory(text: str, edges: set[UUID] | None = None, model: str | None = None) -> Memory:
    """Create an assistant memory from LLM output."""
    return Memory(
        uuid=uuid7(),
        data=SelfData(
            parts=[TextPart(content=text)],
            model=model
        ),
        edges=edges or set()
    )


def chatbot_loop(db_path: str = "memoria.db", model: str = "claude-3-5-sonnet-20241022"):
    """Main chatbot REPL loop."""

    # Initialize client if available
    client = None
    if HAS_ANTHROPIC:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            client = Anthropic(api_key=api_key)
        else:
            print("Warning: ANTHROPIC_API_KEY not set. Falling back to echo mode.")

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
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input == "/quit":
                    print("Goodbye!")
                    break

                if user_input == "/clear":
                    print("(History cleared - not implemented in MVP)")
                    continue

                # Create user memory
                user_memory = create_user_memory(user_input)
                subject.add_memory(user_memory)

                # Get recent roots and recall foliation
                roots = get_recent_roots(subject, n=5)
                if not roots:
                    # First message, no history
                    foliation_text = ""
                else:
                    foliation = subject.recall(roots, recall_config)
                    foliation_text = format_foliation(foliation)

                # Prepare context for LLM
                system_prompt = """You are a helpful assistant. Your responses will be stored in a memory DAG.
You can include frontmatter metadata in your responses using YAML frontmatter format:
---
tags: [tag1, tag2]
custom_field: value
---

Your actual response here."""

                if client:
                    # Call LLM
                    messages = []
                    if foliation_text:
                        messages.append({
                            "role": "user",
                            "content": f"Previous context:\n\n{foliation_text}\n\n---\n\nCurrent message: {user_input}"
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": user_input
                        })

                    response = client.messages.create(
                        model=model,
                        max_tokens=1024,
                        system=system_prompt,
                        messages=messages
                    )

                    assistant_text = response.content[0].text
                else:
                    # Echo mode
                    assistant_text = f"Echo: {user_input}"

                # Parse response
                metadata, content = parse_frontmatter(assistant_text)

                # Create assistant memory
                assistant_memory = create_assistant_memory(
                    content,
                    edges={user_memory.uuid},  # Link to the user's message
                    model=model if client else "echo"
                )
                subject.add_memory(assistant_memory)

                # Display response
                print(f"\nAssistant: {content}\n")

                if metadata:
                    print(f"[Metadata: {metadata}]")
                    print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    chatbot_loop()
