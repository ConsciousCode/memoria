# Memoria
Memoria is an approach to solving context rot in conversational agents, allowing for the indefinite accumulation of memories. It replaces the linear chatlog with a more general primitive, an append-only directed acyclic graph (DAG) of immutable memories. Rather than simply append new messages, at each assistant turn the memory DAG is queried for relevant memories to reconstruct the context, acting as a coarse binary attention filter.

## Why immutable?
Mutable context causes model instability. Causal autoregressive models cannot think outside of the objective, so *even if they're told a modification has occurred*, they struggle to maintain the necessary meta-level awareness of mutability. As an example, suppose the model says "the time is now 12:00" and the system prompt then changes to reflect 12:05; this rewrites the model's only sense of causality. Future completions interpret this context to mean the agent it's simulating was *demonstrably incorrect* in its previous statement and continues this behavior to maintain consistency. Or, at best, apologizes profusely for being "incorrect".

Additionally, if a memory is retroactively added, immutability guarantees that no memories occurring "after" reference it; the model can see this. "I don't remember" remains a valid response, and future recollections in which it *does* remember are consistent with the agent "suddenly" remembering rather than having been incorrect in the past.

## Why a DAG?
A DAG provides a partial ordering over the causal structure of a conversation, allowing the selection of relevant memories and their dependencies without reordering events. Topological sorting enables the linearization of subgraphs into a chatlog as the models are trained to operate over.

## Extensions
### Multi-agent coordination
The append-only immutability of the memory makes a natural synchronization point for multiple asynchronous agents. An agent can pull from shared memory without issue and then append new memories which appear in relevant future queries to other agents. This promises to enable multi-agent coordination without drift in individual subjective experience. For example, multiple agents dedicated to different social media platforms can hold simultaneous conversations while at each turn receiving relevant updates appended by the other agents as if they had originated them. "I'm talking to Becky now, she wants to order pizza".

### Attention masking
Rather than linearize with a topological sort, if one has access to the model itself they could embed the graph structure in a binary mask ala causal masking. This may provide better clarity to the structure than simple in-band references and edges, but requires privileged access. References and edges probably still can't be removed because they provide multi-shot learning of the kind of edge annotation the completion should include.