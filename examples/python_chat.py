#!/usr/bin/env python3
"""
Python MCP client example using fastmcp.Client to drive the `chat` tool.

Usage:
  export OPENAI_API_KEY=your_openai_api_key
  python3 examples/python_chat.py
"""
import os
import asyncio
from fastmcp.client.sampling import SamplingParams
import httpx
from fastmcp.client.client import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp import ClientSession, CreateMessageResult, SamplingMessage
from mcp.shared.context import RequestContext
from mcp.types import TextContent

SERVER = "http://127.0.0.1:8000/mcp"

async def sampling_handler(msgs: list[SamplingMessage], params: SamplingParams, ctx):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    prefs = params.modelPreferences
    if isinstance(prefs, list) and prefs:
        model = prefs[0]
    elif isinstance(prefs, str):
        model = prefs
    else:
        model = "gpt-4.1-nano"
    messages = []
    if params.systemPrompt:
        messages.append({"role": "system", "content": params.systemPrompt})
    for m in params.messages:
        assert isinstance(m.content, TextContent), "Only TextContent is supported"
        messages.append({"role": m.role, "content": m.content.text})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": params.temperature or 0.7,
        "max_tokens": params.maxTokens,
        "stop": params.stopSequences,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
        )
    data = r.json()
    choice = data.get("choices", [{}])[0]
    return CreateMessageResult(
        role=choice.get("message", {}).get("role", "assistant"),
        model=data.get("model"),
        content=TextContent(
            type="text",
            text=choice.get("message", {}).get("content", "")
        ),
        stopReason=choice.get("finish_reason"),
    )

async def main():
    async with Client(
        StreamableHttpTransport(SERVER),
        sampling_handler=sampling_handler,
    ) as client:
        contents = await client.call_tool(
            "chat", {"sona": None, "message": "Hello, world!"}
        )
        text = "".join(getattr(c, "text", "") for c in contents)
        print("Chat response:", text)

if __name__ == "__main__":
    asyncio.run(main())