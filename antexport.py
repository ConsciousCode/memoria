'''
Anthropic exports don't include attached files. This utility generates code
to paste into the dev tools console to fetch them.
'''
import inspect
import os
import re
import json
import sys

from deepmerge import always_merger

from models import DraftMemory, ImportConvo, ImportFileData, ImportMemory, IncompleteMemory, OtherData, TextData
from src.exports.anthropic import AnthropicConvo, AnthropicExport

CHAT_ORGID = "70da662e-fdd6-42d6-9cf2-edfbaa98a782"
'''
Anthropic's organization ID for chat conversations. We use this to fetch
files which aren't exposed in the data export.
'''

CLAUDE_ROOT = "https://claude.ai"

def genconvos(orgid: str, convo_path: str):
    # 1. Load the exported conversations.json
    with open(os.path.expanduser(convo_path)) as f:
        export = AnthropicExport.validate_json(f.read())
    
    # 2. Build a set of conversation UUIDs that have files
    convos = {
        str(c.uuid)
        for c in export
            for msg in c.chat_messages
                if msg.files and any(
                    not f.file_name.endswith(('.txt', '.md'))
                        for f in msg.files
                )
    }

    # 3. Generate the console command
    print(inspect.cleandoc('''
        await (async convos => {
            const out = [];
            for(const convoId of convos.split(' ')) {
                const res = await fetch(`https://claude.ai/api/organizations/%s/chat_conversations/${convoId}`);
                if(!res.ok) {
                    console.error(`Failed to fetch conversation ${convoId}:`, res.statusText);
                    continue;
                }
                out.push(await res.json());
            }
            return out;
        })("%s").then(console.log).catch(console.error);
    ''' % (orgid, ' '.join(convos))))

def genfiles(orgid: str, convo_path: str):
    # 1. Load the exported conversations.json
    with open(os.path.expanduser(convo_path)) as f:
        export = AnthropicExport.validate_json(f.read())

    # 2. Build set of file UUIDs 
    files = {
        m[1] for c in export
            for msg in c.chat_messages
                for f in msg.files
                    if isinstance(f, AnthropicConvo.ChatMessage.CompleteImageFile)
                        # Extracting the UUID lets us better compress it
                        if (m := re.match(".+/([^/]+)/preview", f.preview_url))
    }

    # 3. Generate the URLs
    print(inspect.cleandoc('''
        (files => {
            for(const file of files.split(' ')) {
                const a = document.createElement('a');
                a.href = `https://claude.ai/api/%s/files/${file}/preview`;
                a.download = `claude-${file}`;
                document.body.appendChild(a);
                a.click();
            }
        })("%s");
    '''%(orgid, ' '.join(files))))

def build_memory(file_path: str, msg: AnthropicConvo.ChatMessage) -> ImportMemory:
    match msg.sender:
        case "human":
            # Assume humans only ever have text content
            for c in msg.content or []:
                if isinstance(c, AnthropicConvo.ChatMessage.TextContent):
                    continue
                raise NotImplementedError(
                    f"Unsupported content type: {type(c)}"
                )
            
            # Include dependencies (attachments and files)
            deps: list[DraftMemory] = []

            for f in msg.attachments or []:
                if f.file_type in {"pdf", "txt", "text/markdown", "text/x-python"}:
                    deps.append(IncompleteMemory(
                        data=TextData(content=f.extracted_content)
                    ))
                else:
                    print("Unsupported attachment type:", f.file_type, file=sys.stderr)
            
            for f in msg.files or []:
                if not isinstance(f, AnthropicConvo.ChatMessage.CompleteImageFile):
                    raise NotImplementedError(type(f))
                
                path = os.path.join(file_path, f"{f.file_uuid}.webp")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"File {f.file_uuid} not found at {path}."
                    )
                ts = f.created_at
                deps.append(IncompleteMemory(
                    data=ImportFileData(
                        file=path,
                        filename=f.file_name,
                        mimetype="image/webp"
                    ),
                    timestamp=ts and int(ts.timestamp())
                ))
            
            # Construct the memory
            ts = msg.created_at
            return ImportMemory(
                data=OtherData(content=msg.text),
                timestamp=ts and int(ts.timestamp()),
                deps=deps
            )
        
        case "assistant":
            pass
        
        case _:
            raise NotImplementedError(msg.sender)
    '''
    uuid: UUID
    text: str
    content: Optional[list[Content]] = None
    sender: Literal['assistant', 'human']
    created_at: datetime
    updated_at: datetime
    attachments: list[Attachment] = Field(default_factory=list)
    files: list[File] = Field(default_factory=list)
    '''

def combine(convo_path: str, file_path: str):
    # 1. Load the exported conversations.json
    with open(os.path.expanduser(convo_path)) as f:
        export = json.load(f)

    # 2. Load the files.json
    with open(os.path.expanduser(file_path)) as f:
        files = json.load(f)

    combined = AnthropicExport.validate_python(
        always_merger.merge(export, files)
    )
    convos: list[ImportConvo] = []
    for convo in combined:
        convos.append(ImportConvo(
            metadata=ImportConvo.Metadata(
                timestamp=convo.created_at,
                provider="anthropic",
                uuid=convo.uuid,
                title=convo.name
            ),
            chatlog=[build_memory(msg) for msg in convo.chat_messages]
        ))

    '''

    sona: Optional[UUID|str] = Field(
        default=None,
        description="Sona to insert the memories into."
    )
    metadata: Optional[Metadata] = Field(
        description="Metadata about the conversation being inserted."
    )
    prev: Optional[CIDv1] = Field(
        default=None,
        description="CID of the previous memory in the thread, if any."
    )
    chatlog: list[ImportMemory[AnyMemoryData]] = Field(
        description="Chatlog to insert into the system."
    )'''

def usage():
    print(inspect.cleandoc('''
        Usage: python antexport.py <command> [args]

        Anthropic exports don't include attached files. This utility generates code
        to paste into the dev tools console to fetch them. Anthropic uses Cloudflare,
        so to get any of this requires an active session with JS. We could use
        Puppeteer or Playwright, but this is simpler for now as it's mostly one-time.
        
        orgid = The organization UUID used by the user's account. Haven't found an
          easy way to get this, try dev tools > Networking and look for JSON.
        cj = conversations.json, the raw export from Anthropic.
        fj = files.json, recovered data which includes the file UUIDs exports lack.
          This is designed to only include conversations with attached files.

        Commands:
          genconvos <orgid> <cj>  Generate code to fetch file UUIDs.
          genfiles <orgid> <fj>   Generate code to fetch file URLs from conversations.
          combine <cj> <fj>       Combine conversations and files into a single JSON.
          help                    Show this help message.
        
        The general workflow is:
        - genconvos
        - genfiles
        - combine
        - cli import combined.json files/
    '''))

def main(*argv: str):
    try:
        match argv:
            case ["genconvos", orgid, cj]: genconvos(orgid, cj)
            case ["genfiles", orgid, fj]: genfiles(orgid, fj)
            case ["combine", cj, fj]: combine(cj, fj)
            case ["help"]: usage()
            case _: usage()
    except TypeError:
        usage()
        sys.exit(1)

if __name__ == "__main__":
    main(*sys.argv[1:])