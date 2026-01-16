concept Server
state {
    nick: string,
    host: string,
    port: int
}
actions 
    connect { server: UUID, url: string, port: int } => { server: UUID }
    connect { server: UUID, url: string, port: int } => { error: string }
    disconnect { server: UUID } => { success: true }
    disconnect { server: UUID } => { error: string }

    // Events
    recv { server: UUID, command: "NICK", sender: string, nickname: string, hopcount: int? } => {}
    recv { server: UUID, command: "JOIN", sender: string, channel: string } => {}
    recv { server: UUID, command: "KICK", sender: string, channel: string, user: string, comment: string? } => {}
    recv { server: UUID, command: "PRIVMSG", sender: string, receiver: string, message: string } => {}

concept ServerChannels
state {
    channels: dict[str, UUID]
}
actions
    join { server: UUID, name: string } => { channel: UUID }
    leave { server: UUID, name: string } => {}

concept LastMessage
state
    this: CID
actions
    update { last: UUID, message: CID } => {}

sync UpdateLastOnMessage
when {
    Server/recv { ?server, command: "PRIVMSG", ?sender, ?receiver, ?message } => {}
}
then {
    LastMessage/update { last: ?server, ?message }
}

sync OnSendJoin
when {
    Server/send { ?server, command: "JOIN", ?channel } => {}
}
then {
    Channels/add { ?server, ?channel }
}

sync OnRecvJoin
when {
    Server/recv { ?server, command: "JOIN", ?sender, ?channel } => {}
}
then {
    Channels/add { ?server, ?channel }
}

IRCServer/recv -> BuildIRCMemory -> Memory/add -> RapidRespond -> RapidResponder/respond -> Responder/respond -> IRCResponder -> IRCServer/send

## Vision
Memoria is a cognitive architecture with the aim to try to create an artificial person by fixing the main problem of modern LLMs: they cannot aggregate arbitrarily long subjective experience. It is technically a "multi-agent framework" insofar as it contains multiple "interpreters", but these share a subjective memory and thus act more like lobes of a brain. Previously there was a singular memory called the "Subject", but with the coordination mechanism it makes less and less sense to contain everything in a discrete "person file".

To coordinate between interpreters, I adapt "What You See Is What It Does" (WYSIWID) or as I'll call it, "rhizomatic software". This was originally designed to support vibe coding through hyper-modularization, allowing LLMs to implement discrete features without being able to touch others and break them in the process. Thus, it makes a perfect substrate for a cognitive architecture constructed by the entity itself. Triggering actions directly from pattern-matched actions is far more legible and flexible than maintaining a brittle registry of nominal events. Syncs are able to respond only when all relevant processing has completed without explicit sequencing.

Concepts act as dense strongly-connected causality while syncs provide rhizomatic loose coupling, comparable to mycelia connecting forest roots in small-world networks. Memories are stored in various concepts, external databases, and indexes accessed through concepts, all of this shared by interpreters as convenient. Interpreters themselves are modules of concepts and syncs which may (but don't necessarily) use LLMs to process memories and I/O and create new memories.

Records are entered into IPLD for auditability, strong immutability guarantees, and to ensure causal invariance to avoid autoregressive model instability. I tried to keep records of state explicit rather than implicit through following mutations to allow fast startup and to ensure truncation is minimally disruptive. If a memory from 10 years ago is lost in IPLD, every dependent memory from then on encodes an imprint of what might have been lost. Memories are expected to be queried through both relevance *and* recency which allows emergent knowledge graphs to be updateable through shadowing without needing to be lossy/mutable.

## Terminology
- **Concept**: Silo of state and side-effecting actions.
  - State is a mapping `{UUID: {attribute: [value]}}` to member properties or equivalently multivalued attribute-value mappings on a particular entity.
  - Actions mutate state and perform external I/O. They never throw exceptions; they only ever return error objects. This allows explicit matching against error conditions without special cases.
- **Bootstrap**: A concept with actions invoked by external stimululi. Every time this happens, a new flow is created. Easiest to implement as an async `bootstrap` method yielding action completions. This does not invoke the actions on the concept, bootstrap actions are already completed when entered. "This already happened externally, here's a record".
- **Flow**: A unit of causation used to coordinate syncs and actions. All action matches for a sync must be within the same flow.
- **Synchronization**: (Sync) pattern-matches action completions with variable bindings in a `when` clause, queries concept state in a `where` clause, and invokes one or more actions in a `then` clause.
  - Each `where` clause is a function `: Bindings -> {Bindings}` and `then` is invoked on *every* binding in the final set. This also allows late filtering by mapping to an empty set.
- **Module**: A logical unit containing concepts and syncs under a shared namespace.
- **Hyperconcept**: The global "concept" which stores concept-local state not bound to any entity.
- **Hypersync**: The engine which coordinates syncs and actions.

## Structures
```
/**
 * Prioritize legibility by using mappings by convention.
**/
type Bindings = {name: value}

/**
 * A pattern match against dag-cbor compatible data.
 * - Matches are non-exhaustive. An object with extra keys will still match, eg
 *    `{"a": 1}` matches `{"a": 1, "b": 2}`
 * - Variable repetition eg `{"a": ?a, "b": ?a}` matches `{"a": 1, "b": 1}`.
 * - Literals eg `1` and `"xyz"`
 * - Object destructuring
 * - Wildcard `{}` matches any value, not just an object
 * - Variable-bound keys eg `{?a: ...}` bind to any key not listed. With
 *    variable repetition this doubles as indexing, eg `{"a": ?a, ?a: {}}`
 *    matches `{"a": "b", "b": 100}`. Note that variable-bound keys in general
 *    yield non-unique matches, any pattern which supports this is multivalued.
 *
 * Array destructuring is *not* supported because this enables illegible
 *  patterns such as tuples. Arrays are valid in the data to be matched
 *  against, but these are treated like atomic literal values.
**/
type Pattern = {[name | var]: value | var}

/**
 * Canonically rendered as `{module}.{concept}`. Module disambiguates, in
 *  situations where you can refer to a concept in the same module this can
 *  be omitted.
**/
type ConceptId {
    module: str
    concept: str
}

/**
 * Canonically rendered as `{module}.{concept}/{action}`. Module disambiguates.
**/
type ActionId {
    module: str
    concept: str
    action: str
}

type Concept {
    /**
     * CID tied to the behavior of the concept and its actions to act as a
     *  resolvable versioning system. This can be anything but should change
     *  anytime the behavior changes, so by default consider eg git-raw blobs.
    */
    cid: CID

    /**
     * The concept's name
    **/
    name: str
    
    /**
     * What value does this concept add?
    **/
    purpose: str

    /**
     * The concept's public, queryable state. This must be in-depth enough to
     *  recover any private state. This is a transpose of the equivalent global
     *  entity-oriented view {UUID: {ConceptId: {attribute: [value]}}}.
    **/
    state: {UUID: {attribute: [value]}}

    /**
     * Specifications of each action available on the object.
    **/
    actions: {name: [Callable]}
}

/**
 * [When] clause of a sync. Determines a set of actions to match against and
 *  provides variable binding.
**/
type When {
    /**
     * The action to match against.
    **/
    action: ActionId

    /**
     * Parameter bindings.
    **/
    params: Pattern

    /**
     * Result bindings.
    **/
    result: Pattern
}

/**
 * Query over a concept's state.
**/
type Query {
    /**
     * Name of the concept being queried.
    **/
    concept: ConceptId

    /**
     * Pattern used to query the concept's state.
    **/
    params: Pattern
}

/**
 * Pure function invocation within a [where] clause.
**/
type Call {
    /**
     * Name of the function used to look it up in the global library.
    **/
    name: str

    /**
     * Parameters passed to the function. Vars are substituted from the
     *  environment.
    **/
    params: Pattern

    /**
     * Pattern match over the result to retrieve new bindings.
    **/
    result: Pattern
}

/**
 * [Where] clause of a sync. Enriches the bindings with concept state queries
 *  and effect-free function calls. For instance, UUID7() might create a new
 *  UUID passed to multiple action invocations to coordinate state production.
**/
type Where = Query | Call

/**
 * [Then] clause of a sync specifying which action to invoke with the sync's
 *  bindings.
**/
type Then {
    /**
     * Action to invoke.
    **/
    action: ActionId

    /**
     * The parameters to invoke the action.
    **/
    params: Pattern
}

/**
 * A synchronization which coordinates concepts by pattern-matching over
 *  actions within a flow and invokes new actions.
**/
type Sync {
    /**
     * A readable name for debug; not globally unique.
    **/
    name: str

    /**
     * What does the sync do?
    **/
    purpose: str

    /**
     * When clauses, determining when and what it should match.
    **/
    when: [When]

    /**
     * Where clause, refine bindings to zero or more binding sets.
    **/
    where: [Where]

    /**
     * Then clause, lists actions to invoke with their parameters.
    **/
    then: [Then]
}

/**
 * Record of a sync which was triggered and why.
**/
type Trigger {
    /**
     * Flow the trigger occurs within. Entered into IPLD as a UUID
    **/
    flow: &Flow

    /**
     * Sync initiated by the trigger.
    **/
    sync: &Sync

    /**
     * Completions which caused the trigger. Entered into IPLD as a sorted
     *  list of CIDs, may be useful to consider a mapping {CID: Completion}
    **/
    completions: {Completion}
}

/**
 * A record of an action being completed.
**/
type Completion {
    /**
     * Previous completion for this action.
    **/
    prev: &Completion?

    /**
     * The trigger for the completion. None for bootstrap actions.
    **/
    trigger: &Trigger?

    /**
     * Flow the completion occurred within.
    **/
    flow: &Flow

    /**
     * The name of the action being completed.
    **/
    action: ActionId

    /**
     * Params passed to the action.
    **/
    params: Bindings

    /**
     * Result of the completion.
    **/
    result: Bindings

    /**
     * State of the concept after the completion.
    **/
    state: {UUID: [Bindings]}
}

/**
 * State for tracking the processing of a flow. Once entered into IPLD, acts
 *  as an index of everything involved in the flow.
**/
type Flow {
    /**
     * Flow's UUID.
    **/
    uuid: UUID

    /**
     * All completions in the flow.
    **/
    completions: {&Completion}

    /**
     * All triggers in the flow.
    **/
    triggers: {&Trigger}
}
```
