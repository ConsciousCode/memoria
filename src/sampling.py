from typing import Iterable
from mcp.shared.context import LifespanContextT, RequestContext
from mcp.types import ModelPreferences, TextContent, SamplingMessage
from mcp import ClientSession, CreateMessageResult
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from fastmcp.client.sampling import SamplingParams

from .config import Config

class Sampler:
    '''
    Sampling handler for MCP using pydantic_ai Agents as the driver. Handles
    loading the configuration and selecting the appropriate model based on
    the provided preferences.
    '''
    def __init__(self, config: Config):
        self.config = config
    
    def convert_history(self, msgs: list[SamplingMessage]) -> Iterable[ModelRequest | ModelResponse]:
        """Convert SamplingMessages to ModelRequest/ModelResponse history."""
        for m in msgs:
            assert isinstance(m.content, TextContent), "Only TextContent is supported"
            text = m.content.text
            if m.role == "user":
                yield ModelRequest.user_text_prompt(text)
            elif m.role == "assistant":
                yield ModelResponse(parts=[TextPart(content=text)])
            else:
                raise ValueError(f"Unknown role {m.role!r}")
    
    def convert_settings(self, params: SamplingParams) -> ModelSettings:
        """Convert SamplingParams to ModelSettings."""
        settings: ModelSettings = {}
        if params.maxTokens and params.maxTokens < 4096:
            settings["max_tokens"] = params.maxTokens
        if params.stopSequences:
            settings["stop_sequences"] = params.stopSequences
        if params.temperature is not None:
            settings["temperature"] = params.temperature
        elif self.config.temperature is not None:
            settings["temperature"] = self.config.temperature
        return settings
    
    async def sampling_handler(self,
            msgs: list[SamplingMessage],
            params: 'SamplingParams',
            ctx: RequestContext[ClientSession, LifespanContextT]
        ) -> CreateMessageResult:
        """Sampling handler for FastMCP using Pydantic AI Agent."""
        config = self.config
        prefs = params.modelPreferences or ModelPreferences()
        for prov, name in config.select_model(prefs):
            provider = config.build_provider(prov)
            model = config.build_model(name, provider) if provider else name

            agent = Agent(model,
                instructions=params.systemPrompt
            )
            result = await agent.run(
                message_history=list(self.convert_history(msgs)),
                model_settings=self.convert_settings(params)
            )
            return CreateMessageResult(
                role="assistant",
                content=TextContent(type="text", text=result.output),
                model=f"{prov}:{name}",
                stopReason="endTurn"
            )
        raise ValueError("No suitable model found for sampling handler.")
