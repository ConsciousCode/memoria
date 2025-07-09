'''
Anthropic exports don't include attached files. This utility generates code
to paste into the dev tools console to fetch them.
'''
import inspect
import os
import re
import json

from deepmerge import always_merger

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

def combine(convo_path: str, file_path: str):
    # 1. Load the exported conversations.json
    with open(os.path.expanduser(convo_path)) as f:
        export = json.load(f)

    # 2. Load the files.json
    with open(os.path.expanduser(file_path)) as f:
        files = json.load(f)

    print(json.dumps(
        always_merger.merge(export, files)
    ))

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
    import sys
    main(*sys.argv[1:])