'''
Pydantic models for OpenAI's JSON export format exports.json
'''

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter

class AnthropicConvo(BaseModel):
    """Anthropic conversations.json export."""
    class Account(BaseModel):
        """Anthropic account information."""
        uuid: UUID
    
    class ChatMessage(BaseModel):
        """Anthropic chat message."""
        class Content(BaseModel):
            """Content of an Anthropic chat message."""
            class Input(BaseModel):
                id: Optional[str] = None
                type: Optional[Literal[
                    'application/vnd.ant.mermaid',
                    'application/vnd.ant.react',
                    'application/vnd.ant.code'
                ]] = None
                title: Optional[str] = None
                command: Optional[str] = None
                content: Optional[str] = None
                language: Optional[str] = None
                version_uuid: Optional[UUID] = None
                code: Optional[str] = None
            
            class Part(BaseModel):
                """A part of the content in an Anthropic chat message."""
                type: Literal['text']
                text: str
                uuid: UUID
            
            class Summary(BaseModel):
                """A summary of the chat message."""
                summary: str

            start_timestamp: Optional[datetime] = None
            stop_timestamp: Optional[datetime] = None
            type: Literal['tool_use', 'thinking', 'tool_result', 'text']
            text: Optional[str] = None
            citations: list[dict[str, str]] = Field(default_factory=list) # ?
            name: Optional[Literal['repl', 'artifacts']] = None
            input: Optional[Input] = None
            message: Optional[str] = None
            integration_name: Optional[str] = None
            integration_icon_url: Optional[str] = None
            context: None = None
            display_content: Optional[str] = None
            content: list[Part] = Field(default_factory=list)
            is_error: bool = False
            thinking: Optional[str] = None
            summaries: list[Summary] = Field(default_factory=list)
            cut_off: bool = False

        class Attachment(BaseModel):
            """Attachment in an Anthropic chat message."""
            file_name: str
            file_size: int
            file_type: str
            extracted_content: str
        
        class File(BaseModel):
            """File in an Anthropic chat message."""
            file_name: str

        uuid: UUID
        text: str
        content: list[Content]
        sender: Literal['assistant', 'human']
        created_at: datetime
        updated_at: datetime
        attachments: list[Attachment] = Field(default_factory=list)
        files: list[File] = Field(default_factory=list)

    uuid: UUID
    name: str
    created_at: datetime
    updated_at: datetime
    account: Account
    chat_messages: list[ChatMessage]

AnthropicExport = TypeAdapter(list[AnthropicConvo])