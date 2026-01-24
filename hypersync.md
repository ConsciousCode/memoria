## Vision
Memoria is a cognitive architecture with the aim to try to create an artificial person by fixing the main problem of modern LLMs: they cannot aggregate arbitrary subjective experience. It is technically a "multi-agent framework" insofar as it contains multiple "interpreters", but these share a subjective memory and thus act more like lobes of a brain. Previously there was a singular memory called the "Subject", but with the coordination mechanism it makes less and less sense to contain everything in a discrete "person file".

To coordinate between interpreters, I adapt "What You See Is What It Does" (WYSIWID) or as I'll call it, "rhizomatic programming". This was originally designed to support vibe coding through hyper-modularization, allowing LLMs to implement discrete features without being able to touch others and break them in the process. Thus, it makes a perfect substrate for a cognitive architecture constructed by the entity itself. Triggering actions directly from pattern-matched actions is far more legible and flexible than maintaining a brittle registry of nominal events. Syncs are able to respond only when all relevant processing has completed without explicit sequencing.

Concepts act as dense strongly-connected causality while syncs provide rhizomatic loose coupling, comparable to mycelia connecting forest roots in small-world networks. Memories are stored in various concepts, external databases, and indexes accessed through concepts, all of this shared by interpreters as convenient. Interpreters themselves are clusters of concepts and syncs which may (but don't necessarily) use LLMs to process memories and I/O and create new memories.

Records are entered into IPLD for auditability, strong immutability guarantees, and to ensure causal invariance to avoid autoregressive model instability. I tried to keep records of state explicit rather than implicit through following mutations to allow fast startup and to ensure truncation is minimally disruptive. If a memory from 10 years ago is lost in IPLD, every dependent memory from then on encodes an imprint of what might have been lost. Memories are expected to be queried through both relevance *and* recency which allows emergent knowledge graphs to be updateable through shadowing without needing to be lossy/mutable.

## Terminology
- **Concept**: Hyper-modular silo of state and actions over that state. They cannot refer to other concepts by construction.
- **Action**: Effectful behavior on a concept.
- **Stimulus**: Something external which starts a new flow. These cannot be invoked internally.
- **Pivot**: An action which does nothing on the concept, acting as an event for syncs to match against.
- **Flow**: A unit of causation which starts with a stimulus and ends when a fixed point is reached where there are no action completions which could trigger a sync. All action matches for a sync must be within the same flow.
- **Sync**: Coordinating mechanism which declaratively matches action completions in a `when` clause, queries concept state in a `where` clause, and issues new action invocations in a `then` clause. These clauses support variable binding.
- **Trigger**: The sync and the actions which triggered it, this links all downstream action invocations to their causes.
- **Static state**: The content-addressable state associated with a concept and not any entity which belongs to it.
- **Local state**: The state of a concept associated with an entity.
- **Persisted state**: The content-addressable state persisted to IPLD.
- **Ephemeral state**: Extra state which lives in application memory and must be rehydrated from the content-addressable state.
- **Hyperconcept**: A concept pertaining to the system itself, such as `Concept` whose entities are themselves concepts.
- **Hypersync**: More hypothetical than real, syncs which operate over hyperconcepts to implement the system. Syncs are ordinarily only observable in the causal trace within a flow, so it doesn't matter that these are implemented in code provided relevant hyperconcept actions are emitted.

## Coding standards
### Entity id
All concepts are able to store data associated with entity UUIDs, though many opt not to. This often involves "constructor"-style operations and passing parameters to refer to an entity. When only one is required, it should generally be the lowercase name of the concept. Constructors should *optionally* take an entity id and *always* return the id with the same property name. This allows syncs to parallelize entity construction between different concepts, eg `User` and `Profile` may each receive the same UUID created in the sync, but without downstream syncs needing to know if the constructor received its id or it was created.

### Results
- `{}` when an action is used as a pivot, ie it does nothing except act as a hook for syncs to match against. These cannot error, so there's nothing to disambiguate.
- `{"done": true}` for purely side-effecting actions with no result in principle. These can error, so we need a way to ensure they completed successfully.
- Otherwise return as much information as may be relevant. Don't return state because this can always be queried.

### Stimuli
Stimuli sometimes gets generated because of events happening within the system. For instance, a `Cron` concept gets schedules to produce multiple3 `Cron/job` stimuli. In these cases, because stimuli always start a new flow there is no implicit way to communicate information from the original cause. Thus, these constructor actions should grab whatever extra parameters are provided (name, ids, updates) and forward these to the stimulus invocation wherever possible.

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

type ConceptId = str
type ActionId = "{concept}/{action}"
type FlowId = str(UUID)

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
     *  recover any ephemeral state. This is a transpose of the equivalent global
     *  entity-oriented view `entity concept:attribute value`
    **/
    state: &{UUID: {attribute: [value]}}

    /**
     * Method names of each action available in the concept.
    **/
    actions: {name: [str]}

    /**
     * Method names of each stimulus in the concept.
    **/
    stimuli: {name: [str]}
}

/**
 * [When] clause of a sync. Determines a set of actions to match against and
 *  provides variable binding.
**/
type When = `action {...params} => {...result}`

/**
 * Query over a concept's state.
**/
type Query = `concept {...pattern}`

/**
 * Function invocation within a [where] clause. Name is looked up in an
 *  engine library. It should be non-effectful even if it isn't pure (eg RNG).
**/
type Call = `name {params} => {...result}`

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
type Then = `action {...params}`

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
     * Flow the trigger occurs within.
    **/
    flow: FlowId

    /**
     * Sync initiated by the trigger.
    **/
    sync: &Sync

    /**
     * Completions which caused the trigger. Entered into IPLD as a sorted
     *  list of CIDs for a canonical representation, may be useful to consider
     *  a mapping {CID: Completion}
    **/
    completions: {&Completion}
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
     * The trigger for the completion. None for stimuli.
    **/
    trigger: &Trigger?

    /**
     * Flow the completion occurred within.
    **/
    flow: FlowId

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
    uuid: FlowId

    /**
     * All triggers in the flow.
    **/
    triggers: {&Trigger}

    /**
     * All completions in the flow.
    **/
    completions: {&Completion}
}
```