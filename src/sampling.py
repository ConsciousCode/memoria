from typing import Any, Iterable
from mcp.shared.context import LifespanContextT, RequestContext
from mcp.types import ModelPreferences, TextContent, SamplingMessage
from mcp import ClientSession, CreateMessageResult
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import CallToolFunc, MCPServer, MCPServerSSE
from pydantic_ai.settings import ModelSettings
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from fastmcp.client.sampling import SamplingParams

from cli import Config, SYSTEM_PROMPT

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
    
    async def memoria_process_tool(self,
            ctx: RunContext[str|int],
            call_tool: CallToolFunc,
            tool_name: str,
            args: dict[str, Any]
        ):
        """Process a tool call for Memoria."""
        # For now we just add the related_request_id
        return await call_tool(tool_name, args, {'related_request_id': ctx.deps})
    
    def mcp_servers(self) -> list[MCPServer]:
        """Return a list of MCP server URLs."""
        memoria = MCPServerSSE(
            self.config.server,
            process_tool_call=self.memoria_process_tool
        )
        return [memoria]

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
                instructions=params.systemPrompt or SYSTEM_PROMPT,
                mcp_servers=self.mcp_servers(),
                deps_type=str|int
            )
            result = await agent.run(
                message_history=list(self.convert_history(msgs)),
                model_settings=self.convert_settings(params),
                deps=ctx.related_request_id
            )
            return CreateMessageResult(
                role="assistant",
                content=TextContent(type="text", text=result.output),
                model=f"{prov}:{name}",
                stopReason="endTurn"
            )
        raise ValueError("No suitable model found for sampling handler.")
