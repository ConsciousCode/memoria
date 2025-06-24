import os

__all__ = (
    'ANNOTATE_EDGES',
)

def read_prompt(name: str) -> str:
    d = os.path.dirname(__file__)
    with open(os.path.join(d, f'{name}.md'), 'r') as f:
        return f.read()

ANNOTATE_EDGES = read_prompt('annotation')
QUERY_PROMPT = read_prompt('query')
CHAT_PROMPT = read_prompt('chat')