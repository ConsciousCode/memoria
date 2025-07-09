'''
Pydantic models for OpenAI's JSON export format exports.json
'''

from datetime import datetime
from typing import Annotated, Literal, Optional
from uuid import UUID
import sqlite3
import os

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic_core import Url

from .schema import Invalid, Unknown
from src.ipld import CID

with open(os.path.join(os.path.dirname(__file__), "anthropic.sql"), "r") as f:
    SCHEMA = f.read()

class AnthropicDatabase:
    """Database for Anthropic conversations."""

    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def __enter__(self):
        conn = self.conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        self.cursor().executescript(SCHEMA)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.rollback() if exc_type else self.commit()
        self.close()
        del self.conn
        return False
    
    def cursor(self): return self.conn.cursor()
    def commit(self): self.conn.commit()
    def rollback(self): self.conn.rollback()
    def close(self): self.conn.close()

    def load_convo(self, convo: UUID) -> Optional[CID]:
        cur = self.cursor()
        row = cur.executemany("""
            SELECT memory
            FROM anthropic_convos WHERE convo_uuid = ?
        """, (convo.bytes,)).fetchone()
        if row is not None:
            return CID(row[0])
    
    def load_message(self, message: UUID) -> Optional[CID]:
        cur = self.cursor()
        row = cur.execute("""
            SELECT uuid, memory
            FROM anthropic_messages WHERE message_uuid = ?
        """, (message.bytes,)).fetchone()
        if row is not None:
            return CID(row[0])

class ANT(BaseModel):
    """
    Anthropic doesn't publish their export schema everywhere and it's almost
    as messy as OpenAI.
    """
    model_config = ConfigDict(
        extra='allow',
        populate_by_name=True,
        validate_by_name=True,
        validate_by_alias=True
    )

class Summary(ANT):
    """A summary of the chat message."""
    summary: str

class AnthropicConvo(ANT):
    """Anthropic conversations.json export."""
    class Account(ANT):
        """Anthropic account information."""
        uuid: UUID
    
    class ChatMessage(ANT):
        """Anthropic chat message."""
        class ToolUseContent(ANT):
            class AntCodeInput(ANT):
                type: Literal[
                    'application/vnd.ant.react',
                    'application/vnd.ant.mermaid',
                    'application/vnd.ant.code'
                ]
                id: str
                title: str
                version_uuid: UUID
                language: Optional[str]
                command: Literal['create', 'update', 'rewrite']
                content: str
            
            class MarkdownInput(ANT):
                class Citation(ANT):
                    class Source(ANT):
                        url: Url
                        uuid: UUID
                        title: str
                        source: str
                        icon_url: Url
                        subtitles: Optional[Unknown]
                        content_body: Optional[Unknown]
                        resource_type: Optional[Unknown]
                    
                    class GenericMetadata(ANT):
                        type: Literal['generic_metadata']
                        uuid: UUID
                        source: str
                        icon_url: Url
                        preview_title: str

                    url: Url
                    uuid: UUID
                    title: str
                    sources: list[Source]
                    metadata: GenericMetadata | Unknown
                    end_index: int
                    start_index: int
                    origin_tool_name: Literal[
                        'web_search', 'web_fetch'
                    ] | Invalid[str]
                
                type: Literal['text/markdown']
                id: str
                title: str
                source: str
                command: Literal['create', 'update', 'rewrite']
                content: str
                language: Optional[str]
            
            class Code(ANT):
                code: str
            
            class Command(ANT):
                command: str
            
            class Query(ANT):
                query: str
            
            type Input = Annotated[
                AntCodeInput | MarkdownInput,
                Field(discriminator='type')
            ]
            
            type: Literal['tool_use']
            approval_key: Optional[Unknown]
            approval_options: Optional[Unknown]
            context: Optional[Unknown]
            display_content: Optional[str] = None
            input: Input | Command | Code | Query
            integration_icon_url: Optional[Unknown]
            integration_name: Optional[Unknown]
            message: Optional[str]
            name: Optional[Literal[
                "artifacts"
                "launch_extended_search_task"
                "repl"
                "web_search"
            ] | str]
            start_timestamp: Optional[datetime] = None
            stop_timestamp: Optional[datetime] = None
        
        class ToolResultContent(ANT):
            class TextResult(ANT):
                type: Literal['text']
                text: str
                uuid: UUID
            
            class KnowledgeResult(ANT):
                class WebpageMetadata(ANT):
                    type: Literal['webpage_metadata']
                    site_domain: str
                    favicon_url: Url
                    site_name: str
                
                class PromptContextMetadata(ANT):
                    url: Url
                
                type: Literal['knowledge']
                title: str
                url: Url
                metadata: WebpageMetadata
                is_missing: bool
                text: str
                is_citable: bool
                prompt_context_metadata: PromptContextMetadata
            
            type Result = Annotated[
                TextResult | KnowledgeResult,
                Field(discriminator='type')
            ]

            type: Literal['tool_result']
            content: list[Result]
            start_timestamp: Optional[datetime] = None
            stop_timestamp: Optional[datetime] = None
            display_content: Optional[Unknown] = None
            integration_icon_url: Optional[Unknown] = None
            integration_name: Optional[Unknown] = None
            is_error: bool = False
            message: Optional[str] = None
            name: Optional[Literal[
                'launch_extended_search_task',
                'artifacts', 'repl', 'web_search'
            ] | Invalid[str]] = None
        
        class ThinkingContent(ANT):
            type: Literal['thinking']
            cut_off: bool = False
            start_timestamp: Optional[datetime] = None
            stop_timestamp: Optional[datetime] = None
            summaries: Optional[list[Summary]] = None
            thinking: Optional[str] = None
        
        class TextContent(ANT):
            class Citation(ANT):
                class WebSearch(ANT):
                    type: Literal['web_search_citation']
                    url: Url
                
                uuid: UUID
                start_index: int
                end_index: int
                details: WebSearch | Unknown
            
            type: Literal['text']
            text: str
            citations: list[Citation]
        
        type Content = Annotated[
            ToolUseContent | ToolResultContent | ThinkingContent | TextContent,
            Field(discriminator='type')
        ]

        class Attachment(ANT):
            """Attachment in an Anthropic chat message."""
            file_name: str
            file_size: int
            file_type: str
            extracted_content: str
        
        class CompleteImageFile(ANT):
            '''
            Anthropic exports deliberately exclude files from the export.
            This is the alternate of incomplete files which includes that
            missing information.
            '''
            class Asset(ANT):
                url: str
                primary_color: str
                image_width: int
                image_height: int
            
            class ThumbnailAsset(ANT):
                file_variant: Literal['thumbnail']
            
            class PreviewAsset(ANT):
                file_variant: Literal['preview']
            
            file_kind: Literal['image']
            file_uuid: UUID
            file_name: str
            created_at: datetime
            thumbnail_url: str
            preview_url: str
            thumbnail_asset: ThumbnailAsset
            preview_asset: PreviewAsset
        
        type CompleteFile = Annotated[
            CompleteImageFile,
            Field(discriminator='file_kind')
        ]

        class IncompleteFile(ANT):
            """Incomplete file as seen in chat exports."""
            file_name: str
        
        type File = CompleteFile | IncompleteFile

        uuid: UUID
        text: str
        content: Optional[list[Content]] = None
        sender: Literal['assistant', 'human']
        created_at: datetime
        updated_at: datetime
        attachments: list[Attachment] = Field(default_factory=list)
        files: list[File] = Field(default_factory=list)

    uuid: UUID
    name: str
    created_at: datetime
    updated_at: datetime
    account: Optional[Account] = None
    chat_messages: list[ChatMessage]

AnthropicExport = TypeAdapter(list[AnthropicConvo])
