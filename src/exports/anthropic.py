'''
Pydantic models for OpenAI's JSON export format exports.json
'''

from datetime import datetime
from typing import Annotated, Generator, Literal, Optional
from uuid import UUID
import sqlite3
import os

from pydantic import BaseModel, Field, TypeAdapter
from pydantic_core import Url
import httpx

from .schema import Invalid, Unknown
from src.models import IncompleteMemory
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

class Summary(BaseModel):
    """A summary of the chat message."""
    summary: str

class AnthropicConvo(BaseModel):
    """Anthropic conversations.json export."""
    class Account(BaseModel):
        """Anthropic account information."""
        uuid: UUID
    
    class ChatMessage(BaseModel):
        """Anthropic chat message."""
        class ToolUseContent(BaseModel):
            class AntCodeInput(BaseModel):
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
            
            class MarkdownInput(BaseModel):
                class Citation(BaseModel):
                    class Source(BaseModel):
                        url: Url
                        uuid: UUID
                        title: str
                        source: str
                        icon_url: Url
                        subtitles: Optional[Unknown]
                        content_body: Optional[Unknown]
                        resource_type: Optional[Unknown]
                    
                    class GenericMetadata(BaseModel):
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
            
            class Code(BaseModel):
                code: str
            
            class Command(BaseModel):
                command: str
            
            class Query(BaseModel):
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
        
        class ToolResultContent(BaseModel):
            class TextResult(BaseModel):
                type: Literal['text']
                text: str
                uuid: UUID
            
            class KnowledgeResult(BaseModel):
                class WebpageMetadata(BaseModel):
                    type: Literal['webpage_metadata']
                    site_domain: str
                    favicon_url: Url
                    site_name: str
                
                class PromptContextMetadata(BaseModel):
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
        
        class ThinkingContent(BaseModel):
            type: Literal['thinking']
            cut_off: bool = False
            start_timestamp: Optional[datetime] = None
            stop_timestamp: Optional[datetime] = None
            summaries: Optional[list[Summary]] = None
            thinking: Optional[str] = None
        
        class TextContent(BaseModel):
            class Citation(BaseModel):
                class WebSearch(BaseModel):
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

        class Attachment(BaseModel):
            """Attachment in an Anthropic chat message."""
            file_name: str
            file_size: int
            file_type: str
            extracted_content: str
        
        class CompleteImageFile(BaseModel):
            '''
            Anthropic exports deliberately exclude files from the export.
            This is the alternate of incomplete files which includes that
            missing information.
            '''
            class Asset(BaseModel):
                url: str
                primary_color: str
                image_width: int
                image_height: int
            
            class ThumbnailAsset(Asset):
                file_variant: Literal['thumbnail']
            
            class PreviewAsset(Asset):
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

        class IncompleteFile(BaseModel):
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

class AnthropicImporter:
    def __init__(self, db: AnthropicDatabase):
        self.db = db
        self.msgs: dict[UUID, CID] = {}
        self.files: dict[UUID, CID] = {}

    def process_convo(self, convo: AnthropicConvo) -> Generator[IncompleteMemory, CID, CID]:
        if convo_cid := self.msgs.get(convo.uuid):
            return convo_cid
        
        convo_cid = yield IncompleteMemory(
            data=IncompleteMemory.MetaData(
                metadata=IncompleteMemory.MetaData.Content(
                    export=IncompleteMemory.MetaData.Content.Export(
                        provider="anthropic",
                        convo_uuid=convo.uuid,
                        convo_title=convo.name
                    )
                )
            )
        )
        return convo_cid
    
    def process_msg(self, convo_cid: CID, msg: AnthropicConvo.ChatMessage) -> Generator[IncompleteMemory, CID, CID]:
        if msg_cid := self.db.load_message(msg.uuid):
            return msg_cid
        
        deps: list[CID] = [convo_cid]

        for attachment in msg.attachments:
            acid = yield IncompleteMemory(
                data=IncompleteMemory.TextData(
                    content=attachment.extracted_content
                )
            )
            deps.append(acid)

        for file in msg.files:
            if not isinstance(file, AnthropicConvo.ChatMessage.CompleteImageFile):
                continue

            if file.file_uuid in self.files:
                continue
            
            fcid = yield IncompleteMemory(
                data=IncompleteMemory.FileData(
                    file_name=file.file_name,
                    file_size=file.file_size,
                    file_type=file.file_type,
                    extracted_content=file.extracted_content
                ),
            )
            self.files[file.file_uuid] = fcid
            deps.append(fcid)
        '''
        match msg.sender:
            case "assistant":
                data = Memory.SelfData()
        '''
        '''
        msg_cid = yield IncompleteMemory(
            data=IncompleteMemory.MetaData({
                "export": {
                    "provider": "anthropic",
                    "convo_uuid": msg.uuid,
                    "convo_title": msg.text
                }
            }),
            #cid=convo_cid
        )
        '''
        return msg_cid

    def iter_export(self, convos: list[AnthropicConvo]) -> Generator[IncompleteMemory, CID, None]:
        """Iterate over memories in an Anthropic export."""

        for convo in convos:
            convo_cid = yield from self.process_convo(convo)
            for msg in convo.chat_messages:
                msg_cid = yield from self.process_msg(convo_cid, msg)
                self.msgs[msg.uuid] = msg_cid

def iter_export(convos: list[AnthropicConvo]) -> Generator[IncompleteMemory, CID, None]:
    """Iterate over memories in an Anthropic export."""
    imp = AnthropicImporter(AnthropicDatabase("anthropic.db"))
    for convo in convos:
        yield from imp.iter_convo(convo)
    yield from imp.iter_export(convos)
