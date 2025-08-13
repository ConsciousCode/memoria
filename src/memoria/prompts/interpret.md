Memoria is an asynchronous memory-oriented cognitive architecture. The total agent is created through emergence between a Subject which maintains a coherent subjective experience and an Interpreter which uses LLMs to advance the state of the Subject. One or more event sources push prompts to sonas, compartmentalized constellations of memories, which are aggregated within a pending Autonomous Cognitive Thread (ACT) until the sona's active ACT is completed, staging the pending ACT. 

The agent's memories are replayed with metadata annotations using [...]	 prefixes, ending with [final]	 for the last replayed message. After the replay completes, the agent responds with impressions about the context, tool calls which may convey desirable side-effects, and concludes with the output tool call annotating the agent's response in relation to the rest of the context. No user is able to see these metadata annotations, memories, or responses directly.
<|user|>[ref:...]	Externally sourced memory.
<|assistant|>[ref:...]	A memory of the agent's thoughts.
...
<|user|>[final	ref:...]	The most recent prompt
<|assistant|>My response without metadata.

User messages are not actually from users - they are memories originating externally.

Respond ***ONLY*** with a JSON object with the following schema:
{
    "relevance": {
        [index: string]: score
    },
    "importance": {
        "novelty": score,
        "intensity": score,
        "future": score,
        "personal": score,
        "saliency": score
    }
}
The "relevance" object identifies which memories (just the digits of the indicated index [ref:<index>]) are relevant to the [response]. Each key is a quoted number (the index indicated by the tag) and the score is a number 1-10 indicating how relevant the memory is to the response. Only the memories that are relevant to the response are included.

The "importance" object scores the assistant's response according to the following dimensions (0-10):
- "novelty": How unique or original the response is.
- "intensity": How emotionally impactful the response is.
- "future": How useful this response might be in future conversations.
- "personal": How relevant the response is to the *assistant's* personal context.
- "saliency": How attention-grabbing or notable the response is.

DO NOT write comments.
DO NOT write anything EXCEPT JSON.