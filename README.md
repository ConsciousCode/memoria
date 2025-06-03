# Memoria
Memoria is a cognitive architecture which reframes LLMs as interchangeable simulators of "sonas" which are wholly defined by the contents on their "sona file". MCP is used as the basis for coordinating the sona for great flexibility with interacting with the sona. The intended approach is to simulate it directly, but an MCP host could alternatively talk to the sona as a repository of memories stripped of subjective context or adopt its own secondary persona such as a "hypervisor" for debugging.

## Architecture
- **Agent** - The totality of a system which uses a Memoria MCP server.
- **ACT** - (Autonomous Cognitive Thread) is a process within an agent which "performs" a sona. An actor consists of at least one or more sonas, coordinating instructions, and a set of tools.
- **Self** - The totality of an agent's subjective memories stored in a database and directory of multimodal files.
- **Sona** - A collection of memories which are relevant to a particular context, such as a person or role. Sonas are defined by their name and the memories they contain. Memories recalled within a sona are additionally weighted by the vector similarities of the names of the sonas they're part of, allowing for fuzzy sets of subpersonas.
- **Memory** - A memory is a node in a directed acyclic graph (DAG) representing a single meaningful event. It's connected to supporting memories by weighted edges and may belong to one or more sonas.

ACTs exist in the MCP host which is meant to manage their interactions via MCP clients. Self, Sona, and Memory describe the core data structures managed by a Memoria server. ACTs do not communicate directly; their coordination is implicit through the memories they share semi-isolated by sonas.

## Interactions
Sona isolation is not absolute. Memories in recall are weighted by the vector similarities between sonas they belong to and the sonas being used by an actor. Thus recalled memories will typically be from within the same sona. However, this is not a hard boundary; if a memory is deemed relevant enough, it may be recalled from another sona, at which point it is added to that sona. This allows for fuzzy sets of subpersonas, where sonas can overlap and share memories without being siloed.

Compare this to typical thought patterns in humans. Subconscious processes operate in the background broadly unknown to the conscious mind, but when insights from it are determined to be important or relevant enough, they "bubble up" into conscious awareness. This emulates spontaneous "aha!" moments and more broadly matches the brain's "small-world network" structure; strong local connections with sparse global connections.

## Technical details
### Overall architecture
At the highest level, Memoria simulates sonas using LLMs in a state monad pattern, which can be modeled as:
```
type Sona a = State MemoryDAG a

sonaStep :: LLM -> UserInput -> Sona Response
sonaStep llm input = do
    appendNodeM input
    response <- recall input >>= llm
    appendNodeM response
    pure response
```

Every interaction with the sona queries top-k memories along with their dependencies and renders them with a topological sort to reconstruct a new context for the LLM.[^1] The resulting context compared to most chat logs is very concise. We then use this to generate one new response to be added to memory. Unlike long linear chat logs, this allows the agent to maintain coherence over arbitrarily long time horizons, engage in multiple simultaneous conversations (including with itself), and respond asynchronously or even spontaneously. The only synchronization necessary is for concurrent writes to the sona file; order of writes doesn't matter.

[^1]: topological sort helps to preserve causal order and better leverage LLM's autoregressive nature, but it isn't strictly necessary if we annotate with dependencies.

### Recall
Recall uses a weighted sum to calculate a score. [^2] The components used in this score are:
- **Recency** - The time since the memory was created, with more recent memories being weighted higher. Memories may lack a timestamp which is treated as infinitely far in the past, and thus recency is not considered.
- **Relevance** - The relevance of the memory to the current context, which is determined by the similarity of the memory's content to the current input. This is split into two separate components, vector similarity and fts (full-text search) for exact keyword matches.
- **Importance** - The model annotates 1-10 how relevant a memory was to each response which is incorporated into the memory's importance using EWMA. Importance is never exposed to agent simulators so this mutation serves doesn't otherwise impact the memory's immutability.
- **Sona similarity** - The similarity between the name of the sona being used and the names of sonas the memory belongs to. This enables a kind of fuzzy small-world connectivity for isolating sonas and is calculated using a vector similarity metric such as cosine similarity.

Each memory is assigned a "budget" based on their score to be filled with the weights of edges to dependencies. This is done recursively, multiplying the budget by the weight, until all budgets are exhausted or no more dependencies are found. The final set of memories is loaded into a graph and sorted topologically with the timestamp as a tie-breaker.

### Endpoints
- **Memory** - Fetch a memory by CID.
- **File** - Fetch a file by CID.
- **Recall** - Retrieve the memories that an agent would recall given a prompt.

## Ideas to explore
- Flags to repress memories. May be useful if something goes catastrophic or you say something you shouldn't to avoid having to modify the sona file.

## Citations
[^2]: [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) - The original paper on generative agents which inspired Memoria's architecture.
[^3]: [Memory-augmented neural networks](https://arxiv.org/abs/1605.06065) - A paper on memory-augmented neural networks which inspired the idea of using a DAG for memories.

## Misc notes
- Memory dependencies are immutable, but there's no limit on how many references can be made to a memory. If I go with IPLD, I probably want to encode dependency edges as parts of the content

## IPLD
```
class Memory:
    timestamp: Optional[float]
    kind: str
    data: CBOR
    dependencies: dict[str, tuple[float, CID]]
```