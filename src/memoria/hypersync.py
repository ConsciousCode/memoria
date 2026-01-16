from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence, MutableMapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, AsyncIterable, Callable, ChainMap, ClassVar, Mapping, Protocol, TypedDict, cast, get_type_hints, override
import itertools

from pydantic import BaseModel, Field, TypeAdapter
from uuid_extension import UUID7

from cid import CID
from memoria.util import IPLDModel, IPLDRoot, json_t

type value_t = (
    None | bool | int | float | str | bytes | CID |
    Sequence[value_t] | MutableMapping[str, value_t]
)

@dataclass
class Var:
    name: str

    def __str__(self):
        return f"?{self.name}"

type bind_t = (
    None | bool | int | float | str | bytes | CID |
    Sequence[bind_t] | MutableMapping[str | Var, bind_t] | Var
)

type Pattern = MutableMapping[str | Var, bind_t]
type Template = MutableMapping[str | Var, bind_t]
type Bindings = MutableMapping[str, value_t]

def dump_pattern(x: Pattern | Template | Bindings):
    return {
        str(k): ({"?": v.name} if isinstance(v, Var) else v)
            for k, v in x.items()
    }

def mutable_hash(value: value_t):
    '''Hash mutable values to quickly check for identity.'''
    match value:
        case list():
            return hash(tuple(map(mutable_hash, value)))
        case dict():
            return hash(tuple(
                (k, mutable_hash(v)) for k, v in value.items()
            ))
        case _:
            return hash(value)

class ImmutablePromise[T: value_t]:
    '''
    A wrapper which promises that for the time it's alive, the mutable object
    it wraps won't change. This is used to build collections of unique
    mutable objects like `Bindings`.
    '''

    def __init__(self, value: T):
        self.value = value
    
    def __hash__(self):
        return mutable_hash(self.value)
    
    def __eq__(self, other):
        if isinstance(other, ImmutablePromise):
            return self.value == other.value
        return False

class HyperSyncError(RuntimeError): pass

class ConceptConflictError(HyperSyncError):
    '''Loaded more than one concept with the same name.'''
class NoSuchAction(LookupError, HyperSyncError):
    '''Attempted to invoke a nonexistent action.'''
class NoSuchConcept(LookupError, HyperSyncError):
    '''Attempted to reference a nonexistent concept.'''

class UnloadedConcept(NoSuchConcept):
    '''
    Attempted to invoke an action on a concept in the process of being
    unloaded.
    '''

@dataclass(frozen=True)
class ConceptId:
    '''An identifier for a concept.'''
    module: str
    concept: str

    @classmethod
    def parse(cls, s: str):
        m, c = s.split('.', 1)
        return cls(m, c)
    
    def __str__(self):
        return f"{self.module}.{self.concept}"
    
    def ipld_model(self):
        return {
            "module": self.module,
            "concept": self.concept
        }

@dataclass(frozen=True)
class ActionId:
    '''An identifier for an action.'''
    module: str
    concept: str
    action: str

    @property
    def concept_id(self):
        return ConceptId(module=self.module, concept=self.concept)

    @classmethod
    def parse(cls, s: str):
        mc, a = s.split('/', 1)
        m, c = mc.split('.', 1)
        return cls(m, c, a)
    
    def __str__(self):
        return f"{self.module}.{self.concept}/{self.action}"
    
    def ipld_model(self):
        return {
            "module": self.module,
            "concept": self.concept,
            "action": self.action
        }

@dataclass(frozen=True)
class LocalActionId:
    '''An action reference local to a module.'''
    concept: str
    action: str

    @classmethod
    def parse(cls, s: str):
        c, a = s.split('/', 1)
        return cls(c, a)
    
    def __str__(self):
        return f"{self.concept}/{self.action}"
    
    def qualify(self, module: str):
        return ActionId(module, self.concept, self.action)

'''
Not implemented for match_pattern/state:
- Key-bound variables eg {?a: ...}
- EAV-style attribute multivalues

I found these difficult to implement procedurally. The best way to think about it
is probably as a chain of monads. It already implements this using Optional/Maybe
'''

def match_pattern(
        pattern: bind_t,
        data: value_t,
        bindings: Bindings
    ) -> Bindings | None:
    '''Attempt to match a pattern to data.'''
    
    bss = ChainMap({}, bindings)

    match pattern:
        # Variable binding
        case Var(name):
            if name in bindings:
                if data != bindings[name]:
                    return None
            else:
                bss[name] = data
        
        # Deconstruction
        case dict() if isinstance(data, dict):
            for k, p in pattern.items():
                if isinstance(k, Var):
                    if k.name in bss:
                        k = bss[k.name]
                        if not isinstance(k, str):
                            # Only string keys are allowed
                            return None
                    else:
                        raise NotImplementedError("Key-bound variables")
                
                if k not in data:
                    # Key must exist to match
                    break
                
                d = data[k]
                if (bs := match_pattern(p, d, bss)) is None:
                    return None
                
                bss.update(bs)
        
        # Non-empty dict doesn't match non-dict
        case dict() if pattern:
            return None
        
        # Sequence match
        case list():
            if not isinstance(data, list) or len(pattern) != len(data):
                return None
            
            for p, d in zip(pattern, data):
                if (bs := match_pattern(p, d, bss)) is None:
                    return None
                
                bss.update(bs)
        
        # Literal match
        case _ if pattern != data:
            return None
    
    return bss.maps[0]

def match_state(
        pattern: dict[Var, Pattern],
        state: dict[str, Bindings],
        bindings: Bindings
    ) -> Bindings | None:
    '''Matches a concept's state, always {UUID: {attr: value}}.'''
    
    bss = ChainMap({}, bindings)

    for var, pat in pattern.items():
        if (ent := state.get(var.name)) is None:
            return None
        
        if (bs := match_pattern(pat, ent, bss)) is None:
            return None
        
        bss.update(bs)
    
    return bss.maps[0]

class When(IPLDModel):
    '''
    [When] clause of a sync. Determines a set of actions to match against and
    provides variable binding.
    '''
    
    which: ActionId
    '''The action to match against.'''
    params: Pattern
    '''Parameter bindings.'''
    result: Pattern
    '''Result bindings.'''

    def __init__(self, which: str, params: Pattern, result: Pattern):
        super().__init__(
            which=ActionId.parse(which),
            params=params,
            result=result
        )

    def __str__(self):
        return f"{self.which}: {self.params} => {self.result}"
    
    def ipld_model(self):
        return {
            "which": self.which.ipld_model(),
            "params": dump_pattern(self.params),
            "result": dump_pattern(self.result)
        }

    def match(self,
            completion: 'Completion',
            bindings: Bindings
        ) -> Bindings | None:
        '''Attempt to match a completion with bindings, None on failure.'''

        bss = ChainMap({}, bindings)

        if self.which != completion.which:
            return None
        
        if (bs := match_pattern(self.params, completion.params, bss)) is None:
            return None
        
        bss.update(bs)
        
        if self.result:
            if (bs := match_pattern(self.result, completion.result, bss)) is None:
                return None
            
            bss.update(bs)
        
        return bss.maps[0]

class Query(IPLDModel):
    '''Query over a concept's state.'''

    which: ConceptId
    '''Name of the concept being queried.'''
    pattern: dict[Var, Pattern]
    '''Pattern used to query the concept's state.'''

    def __init__(self, which: str, pattern: dict[Var, Pattern]):
        super().__init__(
            which=ConceptId.parse(which),
            pattern=pattern
        )

    def __str__(self):
        return f"{self.which}: {self.pattern}"
    
    def ipld_model(self):
        return {
            "kind": "query",
            "which": self.which.ipld_model(),
            "pattern": dump_pattern(cast(Pattern, self.pattern))
        }

class Call(IPLDModel):
    '''Pure function invocation within a [where] clause.'''

    name: str
    '''Name of the function used to look it up in the global library.'''
    params: Template
    '''Parameters passed to the function.'''
    result: Pattern
    '''Pattern match over the result to retrieve new bindings.'''

    def __str__(self):
        params = ', '.join(
            f"{name}={value if isinstance(value, Var) else repr(value)}"
                for name, value in self.params.items()
        )
        return f"{self.name}({params}) => {self.result}"
    
    def ipld_model(self):
        return {
            "kind": "call",
            "name": self.name,
            "params": dump_pattern(self.params),
            "result": dump_pattern(self.result)
        }

type Where = Query | Call

class Then(IPLDModel):
    '''
    [Then] clause of a sync specifying which action to invoke with the sync's
    bindings.
    '''

    which: ActionId
    '''Action to invoke.'''
    params: Template
    '''The parameters to invoke the action.'''

    def __init__(self, which: str, params: Template):
        super().__init__(
            which=ActionId.parse(which),
            params=params
        )
    
    def ipld_model(self):
        return {
            "which": self.which.ipld_model(),
            "params": dump_pattern(self.params)
        }

class Signature(TypedDict):
    '''Signature of an action or function.'''

    params: dict[str, json_t]
    '''Json schema for the parameters to pass to an invocation.'''
    result: dict[str, json_t]
    '''Json schema for the result of invocation.'''

def action[**P](name: Callable[P, Mapping[str, value_t]] | str):
    def inner(func: Callable[P, Mapping[str, value_t]]) -> Callable[P, Mapping[str, value_t]]:
        func._action_name = name # pyright: ignore [reportFunctionMemberAccess]
        return func
    
    if isinstance(name, str):
        return inner
    name._action_name = name.__name__  # pyright: ignore [reportFunctionMemberAccess]
    return name

class ConceptMeta(type):
    @override
    def __new__(cls, name: str, bases: tuple, dct: dict):
        def schema(f: Callable):
            hints = get_type_hints(f)
            input = TypedDict("input", {
                k: v for k, v in hints.items() if k != "return" # pyright: ignore [reportGeneralTypeIssues]
            })
            output = hints.get('return')
            return {
                "name": name,
                "input": TypeAdapter(input).json_schema(),
                "output": TypeAdapter(output).json_schema()
            }
        dct['actions'] = {
            n: schema(v) for v in dct.values()
                if (n := getattr(v, "_action_name", None))
        }
        return super().__new__(cls, name, bases, dct)

class Concept(metaclass=ConceptMeta):
    '''Stateful locus of behavior.'''

    cid: CID
    '''
    CID tied to the behavior of the concept and its actions to act as a
    resolvable versioning system. This can be anything but should change
    anytime the behavior changes, so by default consider eg git-raw blobs.
    '''
    name: ClassVar[str]
    '''The concept's name.'''
    purpose: ClassVar[str]
    '''What value does this concept add?'''
    actions: ClassVar[dict[str, Signature]]
    '''Specifications of each action available on the object.'''

    state: dict[str, Bindings]
    '''
    The concept's public, queryable state. This must be in-depth enough to
    recover any private state. {UUID: {attr: [value]}}
    '''

    def __init__(self, state: dict[str, Bindings]):
        super().__init__()
        self.state = state

class Sync(IPLDRoot):
    '''
    A synchronization which coordinates concepts by pattern-matching over
    actions within a flow and invokes new actions.
    '''

    name: str
    '''A readable name for debug; not globally unique.'''
    purpose: str
    '''What does the sync do?'''
    when: list[When]
    '''When clauses, determining when and what it should match.'''
    where: list[Where]
    '''Where clause, refine bindings to zero or more binding sets.'''
    then: list[Then]
    '''Then clause, lists actions to invoke with their parameters.'''

    def __init__(self, name: str, purpose: str, *, when: list[When] | None = None, where: list[Where] | None = None, then: list[Then] | None = None):
        super().__init__(
            name=name,
            purpose=purpose,
            when=[] if when is None else when,
            where=[] if where is None else where,
            then=[] if then is None else then
        )
    
    def ipld_model(self):
        return {
            "name": self.name,
            "purpose": self.purpose,
            "when": [when.ipld_model() for when in self.when],
            "where": [where.ipld_model() for where in self.where],
            "then": [then.ipld_model() for then in self.then]
        }

class Module(IPLDRoot):
    name: str
    '''Name of the module.'''
    concepts: dict[ConceptId, type[Concept]]
    '''List of concepts in the module.'''
    syncs: dict[str, Sync]
    '''List of syncs in the module.'''

    def __init__(self, name: str, concepts: list[type[Concept]], syncs: list[Sync]):
        super().__init__(
            name=name,
            concepts={ConceptId(name, c.name): c for c in concepts},
            syncs={s.name: s for s in syncs}
        )

    async def bootstrap(self) -> AsyncIterable[tuple[str, Bindings, Bindings]]:
        '''Bootstrap entry of a module which yields completions.'''
        if False:
            yield
    
    def ipld_model(self):
        return {
            "name": self.name,
            "concepts": {k: v.cid for k, v in self.concepts.items()},
            "syncs": {k: v.cid for k, v in self.syncs.items()}
        }

class Trigger(IPLDRoot):
    '''Record of a sync which was triggered and why.'''
    
    flow: 'Flow'
    '''Flow the trigger occurs within.'''
    sync: 'Sync'
    '''Sync initiated by the trigger.'''
    completions: dict[CID, 'Completion']
    '''Completions which caused the trigger.'''

    def ipld_model(self):
        return {
            "flow": self.flow.cid,
            "sync": self.sync.cid,
            "completions": list(sorted(self.completions))
        }

class Completion(IPLDRoot):
    '''A record of an action being completed.'''

    trigger: Trigger | None
    '''The trigger for the completion. None for bootstrap actions.'''
    flow: 'Flow'
    '''Flow the completion occurred within.'''
    which: ActionId
    '''The name of the action being completed.'''
    params: Bindings
    '''Params passed to the action.'''
    result: Bindings
    '''Result of the completion.'''
    state: dict[str, Bindings]
    '''State of the concept after the completion.'''

    def ipld_model(self):
        return {
            "trigger": None if self.trigger is None else self.trigger.cid,
            "flow": self.flow.cid,
            "which": self.which.ipld_model(),
            "params": self.params,
            "result": self.result,
            "state": self.state
        }

class Flow(IPLDRoot):
    '''State for tracking the processing of a flow.'''

    uuid: str
    '''Flow's UUID.'''
    completions: dict[CID, Completion]
    '''All completions in the flow.'''
    triggers: dict[CID, Trigger]
    '''All triggers in the flow.'''
    matches: defaultdict[CID, Annotated[defaultdict[int, set[CID]], Field(default_factory=defaultdict)]]
    '''Mapping between syncs and the completions available for matching.'''

    def __init__(self):
        super().__init__(
            uuid=str(UUID7().uuid7),
            completions={},
            triggers={},
            matches=defaultdict(lambda: defaultdict(set))
        )
    
    def ipld_model(self):
        return {
            "uuid": self.uuid,
            "completions": list(sorted(self.completions)),
            "triggers": list(sorted(self.triggers))
        }

@dataclass
class ConceptEntry:
    '''The engine's model of the concept.'''

    concept: Concept
    '''The concept itself.'''

async def queue_consumer[T](q: asyncio.Queue[T]):
    '''
    Async generator which waits until something is pushed to the queue
    and then yields everything in it.
    '''
    while True:
        yield await q.get()
        while not q.empty():
            yield q.get_nowait()

@dataclass
class ModuleEntry:
    module: Module
    '''The actual module.'''
    bootstrap: asyncio.Task | None
    '''Task which processes bootstrap actions. None if this closed.'''

class Engine:
    modules: dict[str, ModuleEntry]
    '''Modules loaded by the engine.'''
    concepts: dict[ConceptId, ConceptEntry]
    '''All loaded concepts.'''
    syncs: dict[CID, Sync]
    '''All loaded syncs.'''
    syncdeps: defaultdict[ActionId, dict[CID, Sync]]
    '''An index of syncs by their action dependencies.'''
    funcs: dict[str, Callable]
    '''Functions loaded by the engine.'''
    flows: dict[str, Flow]
    '''Flows being processed.'''
    queue: asyncio.Queue[Completion]
    '''Queue of completions which have not yet been processed.'''
    tg: asyncio.TaskGroup
    '''Group of all running bootstrap tasks.'''

    def __init__(self):
        super().__init__()
        self.modules = {}
        self.concepts = {}
        self.syncs = {}
        self.syncdeps = defaultdict(dict)
        self.funcs = {}
        self.flows = {}
        self.queue = asyncio.Queue()
        self.tg = asyncio.TaskGroup()

    async def _process_bootstrap(self, name: str):
        entry = self.modules[name]
        async for act, inp, out in entry.module.bootstrap():
            which = LocalActionId.parse(act).qualify(name)
            await self.queue.put(Completion(
                trigger=None,
                flow=self.new_flow(),
                which=which,
                params=inp,
                result=out,
                state=self.concepts[which.concept_id].concept.state
            ))
        
        entry.bootstrap = None
    
    def load_state(self, concept: ConceptId):
        return {}

    def load(self, module: Module):
        '''Load a new module.'''
        mn = module.name
        self.modules[mn] = ModuleEntry(
            module, self.tg.create_task(self._process_bootstrap(mn))
        )
        
        for id, concept in module.concepts.items():
            self.concepts[id] = ConceptEntry(
                concept(self.load_state(id))
            )
        
        for sync in module.syncs.values():
            self.syncs[sync.cid] = sync
            for when in sync.when:
                self.syncdeps[when.which][sync.cid] = sync

    def new_flow(self):
        flow = Flow()
        self.flows[flow.uuid] = flow
        return flow

    async def invoke(self,
            flow: Flow,
            action: ActionId,
            **params: value_t
        ):
        '''Invoke an action and return its result.'''
        try:
            if ce := self.concepts.get(action.concept_id):
                c = ce.concept
                if action.action not in c.actions:
                    raise NoSuchAction(action)
                
                if a := getattr(c, action.action, None):
                    result = a(**params)
                else:
                    raise NoSuchAction(action)
            else:
                raise NoSuchConcept(action.concept)
            
            await self.queue.put(Completion(
                trigger=None,
                flow=flow,
                which=action,
                params=params,
                result=result,
                state=c.state
            ))
            return result
        except Exception as err:
            #await self.uncaught_error(flow, err)
            raise

    async def bootstrap(self,
            action: str,
            **params: value_t
        ) -> Bindings:
        '''Invoke an action ex nihilo.'''
        return await self.invoke(
            self.new_flow(), ActionId.parse(action), **params
        )
    
    async def trigger(self,
            flow: Flow,
            sync: Sync,
            candidate: Iterable[CID]
        ) -> bool:
        '''
        Attempt to trigger a sync with a candidate match, return if
        successful.
        '''
        bindings: Bindings = {}

        # [when]

        for candi, when in zip(candidate, sync.when):
            if (bs := when.match(flow.completions[candi], bindings)) is None:
                # Match failed
                return False
            
            bindings.update(bs)
        
        # [where]

        bss: set[ImmutablePromise[Bindings]] = {ImmutablePromise(bindings)}
        for where in sync.where:
            # For each binding we have thus far, we want to expand them
            # into the sets of bindings
            nbss: list[set[ImmutablePromise[Bindings]]] = []
            for base in bss:
                match where:
                    # Query concept state
                    case Query(which=concept, pattern=pattern):
                        c = self.concepts[concept].concept
                        bs = {**base.value}
                        if not match_state(pattern, c.state, bs):
                            # Query failed, treat as an empty set
                            continue
                        # Queries are 1:1 - wait no they're not wtf?
                        nbss.append({
                            ImmutablePromise(
                                {**bindings, **bs} if bs else bindings
                            )
                        })
            
            bss = {bs for nbs in nbss for bs in nbs}
        
        # [then]

        for bs in bss:
            for then in sync.then:
                await self.invoke(flow, then.which, **bs.value)
        
        return True

    @asynccontextmanager
    async def run(self):
        '''Start the engine to process flows.'''
        async with self.tg:
            yield
            async for cmp in queue_consumer(self.queue):
                flow = cmp.flow
                flow.completions[cmp.cid] = cmp
                # Check all syncs dependent on this action
                for dep in self.syncdeps[cmp.which].values():
                    # Find candidate matches
                    ms = flow.matches[dep.cid]
                    for i, when in enumerate(dep.when):
                        if when.match(cmp, {}) is not None:
                            # Add the completion as a candidate
                            ms[i].add(cmp.cid)
                    
                    # Not every clause has a candidate
                    if len(ms) != len(dep.when):
                        continue
                    
                    # Possible match, check every combination in clause order
                    for candidate in itertools.product(*(ms[i] for i in range(len(ms)))):
                        # Completions can only be matched once but may be a
                        # candidate in more than one clause.
                        if len(candidate) != len(set(candidate)):
                            continue

                        if await self.trigger(flow, dep, candidate):
                            # Successfully processed, don't try any more
                            break
