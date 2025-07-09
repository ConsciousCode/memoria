# Memoria
Memoria is a memory-first cognitive architecture for autonomous AI agents. It treats large language models (LLMs) as interchangeable simulators of a singular "self" split into multiple sub-personas ("sonas"). By organizing interactions as a DAG of memories and dynamically recalling relevant context, Memoria enables coherent, long-term reasoning across multiple concurrent threads.

## Overview
Memoria reframes AI agents as static bundles of memory which can be *advanced* by an LLM-based emulator. Instead of linear chat logs, every input, output, and internal reasoning step is recorded as a **Memory** node in a graph. Memories are grouped into **Sonas** (context-specific sub-personas) to isolate perspectives. Sonas prevent confusing context from leaking between threads while still allowing inter-sona recall as needed.

## Theory
Part of the design considerations for Memoria as compared to other memory-based architectures such as "MemGPT" is the observation that mutable context causes model instability. Causal autoregressive models cannot think outside of the objective, so *even if they're told a modification has occurred*, they struggle to maintain the necessary meta-level awareness of mutability. As an example, suppose the model says "the time is now 12:00" and the system prompt then changes to reflect 12:01. This rewrites the model's only sense of causality. Future iterations interpret this to mean the agent it's simulating was *demonstrably incorrect* in its previous statement and continues this behavior to maintain consistency. Or, at best, apologizes profusely for being "incorrect".

Topologicall3y-sorted memory subgraphs get around this by providing a consistent, immutable, *contextual* view of the past, better accomodating the weaknesses and strengths of causal autoregressive architectures. Context (the memory) becomes *append-only* like the model's actual objective. It ultimately resembles the attention mechanism, selectively highlighting relevant information from a global context which it's blinded to. If eg a memory is retroactively added, immutability guarantees that no memories between the memory's timestamp and the current time reference it; the model can see this. "I don't remember" remains a valid response, and future recollections in which it *does* remember are consistent with the agent "suddenly" remembering rather than having been incorrect in the past.

## Relevant Concepts
**MCP**
: (Model Context Protocol) Fully compliant with Anthropic's MCP for interoperability. MCP is actually extremely well-suited to Memoria's architecture since LLMs live with the **Host** separated from the identity held in the **Server**.
**IPLD**
: (InterPlanetary Linked Data) Used to encode memory graph nodes, enabling *immutable* content-addressed storage of memories. IPLD is used as a system-level constraint to strongly encourage the use of immutable data structures, which are essential for the Memoria architecture.

## Key Concepts
**Agent**
: The complete system of one or more LLMs emulating a self using Memoria.
**Self**
: The aggregate of an agent’s memories across all sonas, stored persistently in a database and multimodal file store.
**Sona**
: A context or persona defined by a collection of memories. Sonas act as centroids in embedding space, enabling fuzzy sub-personas and compartmentalization of concurrent threads.
**ACT**
: (Autonomous Cognitive Thread) A chain of continuations of a single sona. Every sona has *at most one* ACT chain, with each ACT being associated with one memory. Because of this restriction, they are always linear and act as a recombinate chain of thought.
**Memory**
: A node in a directed acyclic graph (DAG) representing a single meaningful event (message, observation, or reasoning step).

## Architecture
### Memory Graph
Memoria records each interaction as a **Memory** node:
```
Memory {
    data: union {
        "self" {
            name: string,
            parts: [{content: string, model?: string}],
            stop_reason?: "endTurn" | "stopSequence" | "maxTokens"
        },
        "other" {
            name?: string,
            content: string
        },
        "text" {
            content: string
        },
        "file" {
            name?: string,
            mimeType?: string,
            content: CID // CID of the file in IPFS
        }
    } discriminated by "kind",
    timestamp?: int?, // Unix timestamp in seconds
    edges: [{target: CID, weight: float}] // sorted by "target"
}
```
Edges encode dependencies: supporting memories contribute context in future recalls. Memoria also keeps track of mutable *importance* values outside of the IPLD model which are used to weight recall and forgetting. Importance is backpropagated through edges, allowing memories to influence each other over time.

### Recall Mechanism
For each new input, Memoria scores and selects relevant memories based on:
- **Recency**: newer memories score higher and more important memories decay slower.
- **Relevance**: semantic similarity (embeddings) and full-text search.
- **Importance**: an EWMA weighting of a memory's past impact.
- **Sona Similarity**: cosine similarity between the current sona name and sonas a memory belongs to.

Top‐K scoring memories are expanded recursively along weighted edges until a budget is exhausted. The resulting subgraph is topologically sorted (timestamp as tiebreaker) to reconstruct a concise causal context for the LLM.

### Autonomous Cognitive Threads (ACTs)
Memoria models sona simulation with a state‑monad pattern:
```haskell
type Sona a = State MemoryDAG a

advance :: LLM -> UserInput -> Sona Response
advance llm input = do
  appendNodeM input
  response <- recall input >>= llm
  appendNodeM response
  pure response
```

## API Endpoints
### Rest
- **GET /ipfs/{cid}[/...path]** - Retrieve data by its CID as a Trustless Gateway.
- **GET /file/{cid}** - Fetch a file by its CID.
- **GET /memory/{cid}** - Retrieve a memory by its CID.
- **GET /memories/list** - List all memories in the system using paging.
- **GET /sona/{uuid}** - Retrieve a sona by its UUID.
- **GET /sonas/list** - List all sonas in the system using paging.

### MCP
- **RESOURCE ipfs://{cid}** - Retrieve data by its CID using the IPFS.
- **RESOURCE memoria://memory/{cid}** - Retrieve a memory by its CID.
- **RESOURCE memoria://file/{cid}** - Fetch a file by its CID.
- **RESOURCE memoria://sona/{uuid}** - Retrieve a sona by its UUID.
- **TOOL recall** - Call the `recall` endpoint to retrieve relevant memories for a given prompt and sona.
- **TOOL insert** - Insert a new memory into memoria, linking it to other relevant memories.
- **TOOL query** - Recall memories and respond to the prompt, but do not store the result in the agent's memory.
- **TOOL chat** - Single-turn chat with the agent, replaying relevant memories and allowing the agent to respond with a new message. The result is stored in the agent's memory.

## References
1. Generative Agents: Interactive Simulacra of Human Behavior. https://arxiv.org/abs/2304.03442  
2. Memory‑Augmented Neural Networks. https://arxiv.org/abs/1605.06065  

## TODO
- Eventually need to implement bitswap to give IPFS nodes access to memories.