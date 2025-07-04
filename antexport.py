'''
Anthropic exports don't include attached files. This utility generates code
to paste into the dev tools console to fetch them.
'''
import inspect
import os
import re

from src.exports.anthropic import AnthropicConvo, AnthropicExport

CHAT_ORGID = "70da662e-fdd6-42d6-9cf2-edfbaa98a782"
'''
Anthropic's organization ID for chat conversations. We use this to fetch
files which aren't exposed in the data export.
'''

CLAUDE_ROOT = "https://claude.ai"

def genconvo(convo_path: str):
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
                const res = await fetch(`https://claude.ai/api/organizations/70da662e-fdd6-42d6-9cf2-edfbaa98a782/chat_conversations/${convoId}`);
                if(!res.ok) {
                    console.error(`Failed to fetch conversation ${convoId}:`, res.statusText);
                    continue;
                }
                out.push(await res.json());
            }
            return out;
        })("%s").then(console.log).catch(console.error);
    ''' % (' '.join(convos),)))

def genfile(convo_path: str):
    # 1. Load the exported conversations.json
    with open(os.path.expanduser(convo_path)) as f:
        export = AnthropicExport.validate_json(f.read())

    # 2. Build set of file UUIDs 
    files = {
        m[1] for c in export
            for msg in c.chat_messages
                for f in msg.files
                    if isinstance(f, AnthropicConvo.ChatMessage.CompleteImageFile)
                        if (m := re.match(".+/([^/]+)/preview", f.preview_url))
    }

    # 3. Generate the URLs
    print(inspect.cleandoc('''
        (files => {
            for(const file of files.split(' ')) {
                const a = document.createElement('a');
                a.href = `https://claude.ai/api/70da662e-fdd6-42d6-9cf2-edfbaa98a782/files/${file}/preview`;
                a.download = `claude-${file}`;
                document.body.appendChild(a);
                a.click();
            }
        })("%s");
    '''%(' '.join(files),)))

def usage():
    print("Usage: python antexport.py <command> [args]")
    print("Commands:")
    print("  gencode <conversations.json> - Generate code to fetch files from conversations.")
    print("  urls <conversations.json> <file_path> - Print URLs for files in conversations.")

def main(*argv: str):
    match argv[0]:
        case "genconvo": genconvo(argv[1])
        case "genfile": genfile(argv[1])
        case "help": usage()
        case _: usage()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)
    main(*sys.argv[1:])