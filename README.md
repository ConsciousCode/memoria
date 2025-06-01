# Memoria
Memoria is a cognitive architecture which reframes LLMs as interchangeable simulators of "sonas" which are wholly defined by the contents on their "sona file". MCP is used as the basis for coordinating the sona for great flexibility with interacting with the sona. The intended approach is to simulate it directly, but an MCP host could alternatively talk to the sona as a repository of memories stripped of subjective context or adopt its own secondary persona such as a "hypervisor" for debugging.

## Technical details
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

## Ideas to explore
- Flags to repress memories. May be useful if something goes catastrophic or you say something you shouldn't to avoid having to modify the sona file.