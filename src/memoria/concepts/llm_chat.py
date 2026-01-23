import asyncio
from typing import Literal, NotRequired, Required, TypedDict, cast

from litellm import ModelResponse, acompletion

from memoria.hypersync import Concept, action, value_t
from memoria.util import json_t

type iso8601 = str

class Timer:
    task: asyncio.Task
    reset: asyncio.Event
    
    def __init__(self, task: asyncio.Task):
        super().__init__()
        self.task = task
        self.reset = asyncio.Event()

class ToolCall(TypedDict):
    name: str
    arguments: dict[str, json_t]

class Message(TypedDict):
    role: Literal['user', 'assistant', 'function', 'system', 'tool']
    content: str | None
    name: NotRequired[str]
    tool_calls: NotRequired[list[ToolCall]]

class Tool(TypedDict):
    type: Literal['function', 'mcp']
    function: json_t

class Config(TypedDict, total=False):
    model: Required[str]
    api_key: str | None

    api_base: str | None
    api_version: str | None
    '''(Azure-specific) the api version for the call'''
    num_retries: int | None
    #context_window_fallback_dict: dict[str, dict]
    #fallbacks: list[Config]
    #metadata: dict

    temperature: float | None
    top_p: float | None
    n: int | None
    stop: str | list[str] | None
    max_completion_tokens: int | None
    max_tokens: int | None
    presence_penalty: float | None
    seed: int | None
    tools: list[Tool] | None
    tool_choice: str | Literal['auto', 'none', 'required', 'any'] | None
    parallel_tool_calls: bool | None
    frequency_penalty: bool | None
    logit_bias: dict[str, float] | None
    timeout: float | None
    logprobs: bool | None
    top_logprobs: Literal[0, 1, 2, 3, 4, 5] | None
    safety_identifier: str | None
    headers: dict[str, json_t] | None
    extra_headers: dict[str, json_t] | None

class Chat(Concept):
    """Chat completions from LLM providers."""

    @action
    async def completion(self, *,
            messages: list[Message],
            config: Config
        ) -> dict[str, value_t]:
        res = cast(ModelResponse, await acompletion(
            messages=messages,
            **config,
            stream=False
        ))
        return res.json()