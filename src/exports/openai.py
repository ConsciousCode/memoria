'''
Pydantic models for OpenAI's JSON export format conversations.json
'''

from datetime import datetime
from typing import Annotated, Any, Literal, Optional
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter, GetCoreSchemaHandler
from pydantic_core import Url, core_schema

type MessageId = Literal['client-created-root']|UUID

class Unknown:
    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        # Define a validator that prints a debug message and returns the value unchanged
        def debug_validator(value: Any, info: core_schema.ValidationInfo) -> Any:
            print("Unknown:", info.field_name)
            return value
        return core_schema.with_info_plain_validator_function(
            debug_validator, field_name=handler.field_name
        )

class OAI(BaseModel):
    """OpenAI maliciously overengineers and underspecifies their exports."""
    model_config = ConfigDict(
        extra='allow',
        populate_by_name=True,
        validate_by_name=True,
        validate_by_alias=True
    )

######################
## Multimodal parts ##
######################

class ImagePart(OAI):
    class Metadata(OAI):
        class Dalle(OAI):
            gen_id: str
            prompt: str
            seed: Optional[int] = None
            parent_gen_id: Optional[Unknown] = None
            edit_op: Optional[Unknown] = None
            serialization_title: str
        
        class Generation(OAI):
            gen_id: str
            gen_size: Literal['image', 'xlimage'] | str
            seed: Optional[int] = None
            parent_gen_id: Optional[Unknown] = None
            edit_op: Optional[Unknown] = None
            serialization_title: str

        dalle: Optional[Dalle] = None
        generation: Optional[Generation] = None

        # Worthless
        gizmo: Optional[Unknown] = None
        container_pixel_height: Optional[int] = None
        container_pixel_width: Optional[int] = None
        emu_omit_glimpse_image: Optional[Unknown] = None
        emu_patches_override: Optional[Unknown] = None
        sanitized: bool
        asset_pointer_link: Optional[Unknown] = None
        watermarked_asset_pointer: Optional[Unknown] = None
    
    content_type: Literal['image_asset_pointer']
    asset_pointer: Url
    size_bytes: int
    width: int
    height: int
    fovea: Optional[int] = None
    metadata: Optional[Metadata] = None

class AudioAssetPointer(OAI):
    class Metadata(OAI):
        start: float
        end: float

        # Worthless
        start_timestamp: Optional[float] = None
        end_timestamp: Optional[float] = None
        pretokenized_vq: Optional[Unknown] = None
        interruptions: Optional[Unknown] = None
        original_audio_source: Optional[Unknown] = None
        transcription: Optional[Unknown] = None
        word_transcription: Optional[Unknown] = None
    
    content_type: Literal['audio_asset_pointer']
    expiry_datetime: Optional[datetime] = None
    asset_pointer: Url
    '''URI to local asset, file-service://file-<slug> or sediment://file_<hex>'''
    size_bytes: int
    format: Literal['wav'] | str
    metadata: Metadata

class AudioTranscriptionPart(OAI):
    content_type: Literal['audio_transcription']
    text: str
    direction: Literal['in', 'out']
    decoding_id: Optional[Unknown] = None

class RealTimeUserAudioVideoAssetPointerPart(OAI):
    content_type: Literal['real_time_user_audio_video_asset_pointer']
    audio_asset_pointer: AudioAssetPointer
    audio_start_timestamp: float

    # Trash
    expiry_datetime: Optional[datetime] = None
    frames_asset_pointers: list[Unknown]
    video_container_asset_pointer: Optional[Unknown] = None

type Part = Annotated[
    # For some reason AudioAssetPointer is available as a Part despite
    # also being embedded in RealTimeUserAudioVideoAssetPointerPart
    ImagePart | AudioTranscriptionPart | AudioAssetPointer |
    RealTimeUserAudioVideoAssetPointerPart,
    Field(discriminator='content_type')
]

###################
## Content types ##
###################

class TextContent(OAI):
    content_type: Literal['text']
    parts: list[str]

class MultimodalTextContent(OAI):
    content_type: Literal['multimodal_text']
    parts: list[str|Part]

class ThoughtContent(OAI):
    class Thought(OAI):
        summary: str
        content: str
    
    content_type: Literal['thoughts']
    thoughts: list[Thought]
    source_analysis_msg_id: UUID

class ReasoningRecapContent(OAI):
    content_type: Literal['reasoning_recap']
    content: str

class UserEditableContextContent(OAI):
    content_type: Literal['user_editable_context']
    user_profile: str
    user_instructions: str

class Tether(OAI):
    tether_id: Optional[Unknown] = None

class TetherQuoteContent(Tether):
    content_type: Literal['tether_quote']
    url: str
    domain: str
    text: str
    title: str

class TetherBrowsingDisplayContent(Tether):
    content_type: Literal['tether_browsing_display']
    result: str
    summary: Optional[str] = None
    assets: Optional[list[Unknown]] = None
    tether_id: Optional[Unknown] = None

class CodeContent(OAI):
    content_type: Literal['code']
    language: str
    response_format_name: Optional[Unknown] = None
    text: str

class SonicWebpageContent(OAI):
    content_type: Literal["sonic_webpage"]
    url: Url
    domain: str
    title: str
    text: str
    snippet: str
    pub_date: Optional[Unknown] = None
    crawl_date: Optional[Unknown] = None
    pub_timestamp: Optional[float] = None
    ref_id: str

class ExecutionOutputContent(OAI):
    content_type: Literal['execution_output']
    text: str

class SystemErrorContent(OAI):
    content_type: Literal["system_error"]
    name: str
    text: str

type Content = Annotated[
    UserEditableContextContent | ExecutionOutputContent |
    TextContent | MultimodalTextContent | CodeContent |
    TetherBrowsingDisplayContent | TetherQuoteContent |
    SonicWebpageContent |
    ThoughtContent | ReasoningRecapContent | SystemErrorContent,
    Field(discriminator='content_type')
]

class MessageMetadata(OAI):
    class UserContextMessageData(OAI):
        about_user_message: Optional[str] = None
        about_model_message: Optional[str] = None
    
    class FinishDetails(OAI):
        type: Literal['stop', 'interrupted', 'max_tokens', 'unknown']
        stop_tokens: Optional[list[int]] = None
    
    class SerializationMetadata(OAI):
        custom_symbol_offsets: Optional[list[Unknown]] = None
    
    class Citation(OAI):
        class Webpage(OAI):
            class Extra(OAI):
                cited_message_idx: Optional[int] = None
                search_result_idx: Optional[int] = None
                evidence_text: Optional[str] = None
                cloud_doc_url: Optional[Url] = None
            
            type: Literal['webpage']
            title: str
            url: Url
            text: str
            pub_date: Optional[datetime] = None
            extra: Optional[Extra] = None
            og_tags: Optional[Unknown] = None
        
        citation_format_type: Optional[
            Literal['tether_og', 'tether_markdown'] | str
        ] = None
        start_idx: int = Field(
            validation_alias=AliasChoices("start_idx", "start_ix")
        )
        end_idx: int = Field(
            validation_alias=AliasChoices("end_idx", "end_ix")
        )
        invalid_reason: Optional[str] = None
        metadata: Optional[Webpage] = None
    
    class InvokedPlugin(OAI):
        type: Literal['remote', 'local']
        namespace: str
        plugin_id: str
        http_response_status: int
    
    class JITPluginData(OAI):
        class FromServer(OAI):
            class Body(OAI):
                class AllowAction(OAI):
                    class Allow(OAI):
                        target_message_id: UUID
                    
                    name: Literal['allow'] # ???
                    type: Literal['allow']
                    allow: Allow
                
                class AlwaysAllowAction(OAI):
                    class AlwaysAllow(OAI):
                        target_message_id: UUID
                        operation_hash: str
                    
                    # No name??
                    type: Literal['always_allow']
                    always_allow: AlwaysAllow
                
                class DenyAction(OAI):
                    class Deny(OAI):
                        target_message_id: UUID
                    
                    name: Literal['deny']
                    type: Literal['deny']
                    deny: Deny
                
                type Action = Annotated[
                    AllowAction | AlwaysAllowAction | DenyAction,
                    Field(discriminator='type')
                ]

                domain: str
                is_consequential: bool
                privacy_policy: Url
                method: Literal['get', 'post']
                path: str
                operation: str
                params: dict[str, Any]
                actions: list[Action]
            
            type: Literal['preview']
            body: Body
    
    class AggregateResult(OAI):
        class InKernelException(OAI):
            name: str
            traceback: list[str]
            args: list[Any]
            notes: list[Any]
        
        class ImageMessage(OAI):
            message_type: Literal['image']
            time: float
            sender: Literal['server'] | str
            image_payload: Optional[Unknown] = None
            image_url: Url
            width: int
            height: int
        
        class StreamMessage(OAI):
            message_type: Literal['stream']
            time: float
            stream_name: Literal['stdout', 'stderr'] | str
            sender: Literal['server'] | str
            text: str
        
        type Message = Annotated[
            ImageMessage | StreamMessage,
            Field(discriminator='message_type')
        ]

        class BaseJupyterMessage(OAI):
            class ParentHeader(OAI):
                msg_id: str
                version: str
            
            parent_header: ParentHeader

        class JupyterStatus(BaseJupyterMessage):
            class Content(OAI):
                execution_state: Literal['busy', 'idle'] | str
            
            msg_type: Literal['status']
            content: Content

        class JupyterExecuteInput(BaseJupyterMessage):
            msg_type: Literal['execute_input']
        
        class JupyterExecuteResult(BaseJupyterMessage):
            class Content(OAI):
                data: dict[str, str]
            
            msg_type: Literal['execute_result']
            content: Content
        
        class JupyterDisplayData(BaseJupyterMessage):
            class Content(OAI):
                data: dict[str, str]
            
            msg_type: Literal['display_data']
            content: Content
        
        type JupyterMessage = Annotated[
            JupyterStatus | JupyterExecuteInput |
            JupyterExecuteResult | JupyterDisplayData,
            Field(discriminator='msg_type')
        ]

        status: Literal[
            'success', 'cancelled', 'failed_with_in_kernel_exception'
        ] | str
        run_id: UUID
        start_time: float
        update_time: float
        code: str
        end_time: float
        final_expression_output: str
        in_kernel_exception: Optional[InKernelException] = None
        system_exception: Optional[Unknown] = None
        messages: list[Message]
        jupyter_messages: list[JupyterMessage]
        timeout_triggered: Optional[Unknown] = None
    
    class MetadataListFile(OAI):
        type: Literal['file']
        name: str
        id: str
        source: Literal['my_files'] | str
        extra: Optional[Unknown] = None
    
    class MetadataListWebpage(OAI):
        type: Literal['webpage']
        title: str
        url: Url
        text: str
        pub_date: Optional[Unknown] = None
        extra: Optional[Unknown] = None
    
    type MetadataListItem = Annotated[
        MetadataListFile | MetadataListWebpage,
        Field(discriminator='type')
    ]

    class SearchResultGroup(OAI):
        class SearchResult(OAI):
            class RefId(OAI):
                turn_index: int
                ref_type: Literal['search'] | str
                ref_index: int
            
            type: Literal['search_result']
            url: Url
            title: str
            snippet: str
            ref_id: RefId
            content_type: Optional[Unknown] = None
            pub_date: Optional[float] = None
            attributions: Optional[list[Unknown]] = None
            attribution: Optional[str] = None
    
    class BaseContentReferences(OAI):
        matched_text: str
        start_idx: int = Field(
            validation_alias=AliasChoices("start_idx", "start_ix")
        )
        end_idx: int = Field(
            validation_alias=AliasChoices("end_idx", "end_ix")
        )
        alt: Optional[str] = None
    
    class ContentReferencesHidden(BaseContentReferences):
        type: Literal['hidden']
        invalid: bool = False
    
    class ContentReferencesVisible(BaseContentReferences):
        refs: Optional[list[Literal['hidden'] | str]] = None
        safe_urls: Optional[list[Url]] = None
        prompt_text: Optional[str] = None
    
    class ContentReferencesAttribution(ContentReferencesVisible):
        type: Literal['attribution']
        attributable_index: str
        attributions: Optional[list[Unknown]] = None
        attributions_debug: Optional[list[Unknown]] = None
    
    class ContentReferencesImageV2(BaseContentReferences):
        class Ref(OAI):
            turn_index: int
            ref_type: Literal['image'] | str
            ref_index: int
        
        class Image(OAI):
            class Size(OAI):
                width: int
                height: int
            
            url: Url
            content_url: Url
            thumbnail_url: Url
            title: str
            content_size: Size
            thumbnail_size: Size
            attribution: str
        
        type: Literal['image_v2']
        refs: list[Ref]
        images: list[Image]

    class ContentReferencesWebpage(ContentReferencesVisible):
        type: Literal['webpage']
        title: str
        url: Url
        pub_date: Optional[float] = None
        snippet: str
        attributions: Optional[Unknown] = None
        attributions_debug: Optional[Unknown] = None
        attribution: Optional[str] = None

    class ContentReferencesGroupedWebpages(ContentReferencesVisible):
        class Item(OAI):
            class Refs(OAI):
                ref_type: Literal['search']
                turn_index: int
                ref_index: int
            
            class Website(OAI):
                title: str
                url: Url
                pub_date: Optional[float] = None
                snippet: str
                attribution: str
            
            title: str
            url: Url
            pub_date: Optional[float] = None
            snippet: str
            attribution_segments: Optional[list[str]] = None
            '''String of index ranges eg 0-3'''
            supporting_websites: Optional[list[Website]] = None
            refs: Optional[list[Annotated[
                Refs|Refs, Field(discriminator='ref_type')
            ]]] = None
            hue: Optional[Unknown] = None
            attributions: Optional[list[Unknown]] = None
            attribution: Optional[str] = None
            '''Plaintext attribution like "Wikipedia"'''
        
        type: Literal['grouped_webpages', 'grouped_webpages_model_predicted_fallback']
        items: list[Item]
        status: Optional[Literal['done', 'error'] | str] = None
        error: Optional[Unknown] = None
        style: Optional[Literal['v2'] | str] = None
    
    class ContentReferencesWebpageExtended(ContentReferencesVisible):
        type: Literal['webpage_extended']
        title: str
        url: Url
        pub_date: Optional[datetime] = None
        snippet: str
    
    class SourcesFootnote(ContentReferencesVisible):
        class Source(OAI):
            title: str
            url: Url
            attribution: str
        
        type: Literal['sources_footnote']
        sources: list[Source]
        has_images: bool

    type ContentReferences = Annotated[
        ContentReferencesHidden | ContentReferencesAttribution |
        ContentReferencesGroupedWebpages | ContentReferencesImageV2 |
        ContentReferencesWebpage | ContentReferencesWebpageExtended |
        SourcesFootnote,
        Field(discriminator='type')
    ]

    class Canvas(OAI):
        class Selection(OAI):
            class Range(OAI):
                start: int
                end: int
            
            selection_type: Literal['selection']
            selection_position_range: Range
        
        textdoc_id: str
        textdoc_type: str
        version: int
        textdoc_content_length: int
        user_message_type: Literal['ask_chatgpt'] | str
        selection_metadata: Selection

    is_user_system_message: bool = False
    is_visually_hidden_from_conversation: bool = False
    is_complete: bool = True
    user_context_message_data: Optional[UserContextMessageData] = None
    timestamp_: Literal['absolute', 'relative'] = 'absolute'
    message_type: Optional[Literal['next'] | str] = None
    model_slug: Optional[str] = None
    default_model_slug: Optional[str] = None
    finish_details: Optional[FinishDetails] = None
    selected_github_repos: Optional[list[str]] = None
    serialization_metadata: Optional[SerializationMetadata] = None
    request_id: Optional[str] = None
    message_source: Optional[Unknown] = None
    voice_mode_message: bool = False
    real_time_audio_has_video: bool = False
    citations: Optional[list[Citation]] = None
    content_references: Optional[list[ContentReferences]] = None
    gizmo_id: Optional[str] = None
    pad: Optional[str] = None
    parent_id: Optional[MessageId] = None
    command: Optional[str] = None
    jit_plugin_data: Optional[JITPluginData] = None
    reasoning_status: Optional[Literal['reasoning_ended', 'is_reasoning']] = None
    metadata_list: Optional[list[MetadataListItem]] = None
    original_query: Optional[Unknown] = None
    search_result_groups: Optional[list[SearchResultGroup]] = None

class Message(OAI):
    class Author(OAI):
        class Metadata(OAI):
            real_author: Optional[str] = None
        
        role: Literal['system', 'assistant', 'user', 'tool'] | str
        name: Optional[str] = None
        metadata: Optional[Metadata] = None
    
    id: MessageId
    author: Author
    create_time: Optional[float] = None
    update_time: Optional[float] = None
    content: Content
    status: Literal[
        'cancelled', 'done', 'failed',
        'failed_with_in_kernel_exception', 'finished',
        'finished_partial_completion', 'finished_successfully',
        'in_progress', 'running', 'success'
    ] | str
    end_turn: Optional[bool] = None
    weight: float
    metadata: MessageMetadata
    recipient: str
    channel: Optional[Literal['final', 'commentary'] | str] = None

class Node(OAI):
    id: MessageId
    message: Optional[Message] = None
    parent: Optional[MessageId] = None
    children: list[UUID] = Field(default_factory=list)

class OpenAIConvo(OAI):
    title: str
    '''Title of the conversation.'''
    create_time: float
    '''Timestamp of when the conversation was created.'''
    update_time: float
    '''Timestamp of the last update to the conversation.'''
    mapping: dict[MessageId, Node]
    '''Mapping of UUIDs to their messages.'''
    voice: Optional[str] = None
    '''Name of the voice being used.'''
    id: UUID
    '''UUID of the conversation.'''

    # Vague utility
    is_archived: bool = False
    is_starred: Optional[bool] = None
    memory_scope: Literal['global_enabled'] | str

    # All of the rest of this is probably worthless
    moderation_results: list[Unknown]
    current_node: UUID
    plugin_ids: Optional[Unknown] = None
    conversation_id: UUID
    conversation_template_id: Optional[str] = None
    gizmo_id: Optional[str] = None
    gizmo_type: Optional[Literal['gpt'] | str] = None
    safe_urls: Optional[list[str]] = None
    blocked_urls: Optional[list[Unknown]] = None
    default_model_slug: Optional[str] = None
    conversation_origin: Optional[Unknown] = None
    async_status: Optional[int] = None # Literal[1, 2]?
    disabled_tool_ids: Optional[list[Unknown]] = None
    is_do_not_remember: Optional[bool] = None
    sugar_item_id: Optional[Unknown] = None

OpenAIExport = TypeAdapter(list[OpenAIConvo])

import os
with open("/home/consciouscode/data/Data/openai/export/conversations.json", 'r') as f:
    OpenAIExport.validate_json(f.read())