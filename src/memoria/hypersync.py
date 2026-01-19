import asyncio
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property
import inspect
from typing import AsyncIterable, Awaitable, Callable, ClassVar, Mapping, MutableMapping, MutableSequence, Self, cast, overload, override
import itertools
import traceback
from uuid import UUID

from uuid_extension import UUID7

from cid import CID, CIDv1
from ipld import dagcbor

from .util import IPLDRoot

@dataclass
class Var:
    name: str

    def __str__(self):
        return f"?{self.name}"

type simple_t = None | bool | int | float | str | bytes | CID
type composite_t[K, V] = simple_t | Sequence[V] | Mapping[K, V]
type var_t[K] = composite_t[K, var_t[K]] | Var

type mutable_t = simple_t | MutableSequence[mutable_t] | MutableMapping[str, mutable_t]
type value_t = composite_t[str, value_t]
type bind_t = var_t[str]
type multibind_t = var_t[str | Var]

type State = MutableMapping[UUID, MutableMapping[str, mutable_t]]
type Pattern = Mapping[str, bind_t]
type Multipattern = Mapping[str | Var, multibind_t]
type Template = Mapping[str | Var, multibind_t]
type Bindings = Mapping[str, value_t]

def dump_pattern[K: str | Var](x: var_t[K]):
    match x:
        case dict():
            return {
                str(k): dump_pattern(v)
                    for k, v in x.items()
            }
        
        case list():
            return list(map(dump_pattern, x))
        
        case Var(name):
            return {"?": name}
        
        case _:
            return x

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
class EventInvoked(HyperSyncError):
    '''Attempted to invoke an event.'''

class UnloadedConcept(NoSuchConcept):
    '''
    Attempted to invoke an action on a concept in the process of being
    unloaded.
    '''

type ConceptId = str
type ActionId = str

'''
Not implemented for match_multi/state:
- EAV-style attribute multivalues

I found these difficult to implement procedurally. The best way to think about it
is probably as a chain of monads.
'''

def concat[T](vss: Iterable[Iterable[T]]) -> list[T]:
    return [v for vs in vss for v in vs]

def match_subpattern[K: str | Var, V: multibind_t](
        k: K,
        p: V,
        pattern: Mapping[K, V],
        data: Bindings,
        bs: Bindings
    ) -> Iterator[Bindings]:
    match k:
        case str():
            if k in data:
                yield from match_multi(p, data[k], bs)
        
        case Var(kn):
            if kn in bs:
                # Variable key
                vk = bs[kn]
                # Ideas:
                # - k could be a list allowing multiple matches
                # - k could be an int and data a list (probably illegible)
                if isinstance(vk, str) and vk in data:
                    yield from match_multi(p, data[vk], bs)
            else:
                # Key-bound variable binds to all keys not in the pattern
                for dk, dv in data.items():
                    if dk in pattern:
                        continue
                    yield from match_multi(p, dv, {**bs, kn: dk})

def match_multi(
        pattern: multibind_t,
        data: value_t,
        bindings: Bindings
    ) -> Iterator[Bindings]:
    '''Attempt to match a pattern to data. Yields all possible bindings.'''

    match pattern:
        # Variable binding
        case Var(name):
            if name in bindings:
                if data == bindings[name]:
                    yield bindings
            else:
                yield {**bindings, name: data}
        
        # Deconstruction
        case dict():
            # {} matches anything, need this for typing
            if not isinstance(data, dict):
                if not pattern:
                    yield bindings
                return

            # For each candidate binding, expand it with submatches and iterate
            # over the whole pattern.
            bss = [bindings]
            for k, p in pattern.items():
                if not (bss := concat(match_subpattern(k, p, pattern, data, bs) for bs in bss)):
                    break
            
            yield from bss
        
        # Sequence match
        case list():
            if not isinstance(data, list) or len(pattern) != len(data):
                return
            
            bss = [bindings]
            for p, d in zip(pattern, data):
                if not (bss := concat(match_multi(p, d, bs) for bs in bss)):
                    break
            
            yield from bss
        
        # Literal match
        case _ if pattern == data:
            yield bindings

def match_state(
        pattern: dict[Var, Multipattern],
        state: dict[str, Bindings],
        bindings: Bindings
    ) -> Iterator[Bindings]:
    '''Matches a concept's state, always {UUID: {attr: value}}.'''
    
    # Currently overkill but this will allow future EAV multivalues
    bss = [bindings]

    for var, p in pattern.items():
        if (ent := state.get(var.name)) is None:
            break
        
        if not (bss := concat(match_multi(p, ent, bs) for bs in bss)):
            break
    else:
        yield from bss

def match_pattern(
        pattern: bind_t,
        data: value_t,
        bindings: Bindings
    ) -> Bindings | None:
    '''
    Match a simple binary pattern, as in the [when] clause. Return the
    bindings on match, else None. This intentionally does not support key-bound
    variables.
    '''

    match pattern:
        # Variable binding
        case Var(name=name):
            if name in bindings:
                if data == bindings[name]:
                    return bindings
            else:
                return {**bindings, name: data}
        
        # Deconstruction
        case dict():
            # {} matches anything, need this for typing
            if not isinstance(data, dict):
                if not pattern:
                    return bindings
                return

            for k, p in pattern.items():
                if k not in data:
                    return
                if (bs := match_pattern(p, data[k], bindings)) is None:
                    return
                bindings = bs
            
            return bindings
        
        # Sequence match
        case list():
            if not isinstance(data, list) or len(pattern) != len(data):
                return
            
            for p, d in zip(pattern, data):
                if (bs := match_pattern(p, d, bindings)) is None:
                    return
                bindings = bs
            return bindings
        
        # Literal match
        case _ if pattern == data:
            return bindings

@dataclass(frozen=True)
class When:
    '''
    [When] clause of a sync. Determines a set of actions to match against and
    provides variable binding.
    '''
    
    action: ActionId
    '''The action to match against.'''
    params: Pattern
    '''Parameter bindings.'''
    result: Pattern
    '''Result bindings.'''

    def ipld_model(self):
        return {
            "action": self.action,
            "params": dump_pattern(self.params),
            "result": dump_pattern(self.result)
        }
    
    def match(self,
            completion: 'Completion',
            bindings: Bindings
        ) -> Bindings | None:
        '''Attempt to match a completion with bindings, None on failure.'''

        if self.action != completion.action:
            return
        
        if (bs := match_pattern(self.params, completion.params, bindings)) is None:
            return
        bindings = bs
        
        if self.result:
            if (bs := match_pattern(self.result, completion.result, bindings)) is None:
                return
            bindings = bs
        
        return bindings

@dataclass(frozen=True)
class Query:
    '''Query over a concept's state.'''

    concept: ConceptId
    '''Name of the concept being queried.'''
    pattern: dict[Var, Multipattern]
    '''Pattern used to query the concept's state.'''

    def ipld_model(self):
        return {
            "concept": self.concept,
            "pattern": {
                str(k): dump_pattern(v)
                    for k, v in self.pattern.items()
            }
        }

@dataclass(frozen=True)
class Call:
    '''Pure function invocation within a [where] clause.'''

    name: str
    '''Name of the function used to look it up in the global library.'''
    params: Template
    '''Parameters passed to the function.'''
    result: Multipattern
    '''Pattern match over the result to retrieve new bindings.'''

    def ipld_model(self):
        return {
            "name": self.name,
            "params": dump_pattern(self.params),
            "result": dump_pattern(self.result)
        }

type Where = Query | Call

@dataclass(frozen=True)
class Then:
    '''
    [Then] clause of a sync specifying which action to invoke with the sync's
    bindings.
    '''

    action: ActionId
    '''Action to invoke.'''
    params: Template
    '''The parameters to invoke the action.'''

    def ipld_model(self):
        return {
            "action": self.action,
            "params": dump_pattern(self.params)
        }

def action[**P](name: Callable[P, Awaitable[Mapping[str, object]]] | str):
    def inner(func: Callable[P, Awaitable[Mapping[str, object]]]):
        func._action_name = name # pyright: ignore [reportFunctionMemberAccess]
        return func
    
    if isinstance(name, str):
        return inner
    
    func, name = name, name.__name__
    return inner(func)

def event[**P](name: Callable[P, Awaitable[Mapping[str, object] | None]] | str):
    def inner(func: Callable[P, Awaitable[Mapping[str, object] | None]]):
        func._event_name = name # pyright: ignore [reportFunctionMemberAccess]
        return func
    
    if isinstance(name, str):
        return inner
    
    func, name = name, name.__name__
    return inner(func)

class ConceptMeta(type):
    '''Aggregates actions and events from decorators.'''

    @override
    def __new__(cls, name: str, bases: tuple, dct: dict):
        dct['actions'] = actions = {}
        dct['events'] = events = {}
        for k, v in dct.items():
            if (n := getattr(v, "_action_name", None)):
                actions[n] = k
                del v._action_name
            if (n := getattr(v, "_event_name", None)):
                events[n] = k
                del v._event_name
        
        if (source := dct.get('source')) is None:
            try:
                fn = inspect.getfile(cls)
                with open(fn, "r") as f:
                    source = f.read()
                dct['source'] = source
            except FileNotFoundError:
                pass
        
        if source is not None and 'cid' not in dct:
            dct['cid'] = CIDv1.hash(source.encode('utf-8'), codec='raw')
        
        if 'name' not in dct:
            dct['name'] = name
        
        if 'purpose' not in dct:
            dct['purpose'] = dct['__doc__']

        return super().__new__(cls, name, bases, dct)

class Concept(metaclass=ConceptMeta):
    '''Stateful locus of behavior.'''

    cid: CID
    '''
    CID tied to the behavior of the concept and its actions to act as a
    resolvable versioning system. This can be anything but should change
    anytime the behavior changes. For now, ConceptMeta uses dag-raw on
    the file a concept is defined within.
    '''
    source: str
    '''The source code which created the CID.'''
    actions: ClassVar[dict[str, str]]
    '''Specifications of each action available on the object.'''
    events: ClassVar[dict[str, str]]
    '''Specifications of each event available on the object.'''

    name: ClassVar[str]
    '''The concept's name.'''
    purpose: ClassVar[str]
    '''What value does this concept add?'''

    state: State
    '''
    The concept's public, queryable state. This must be in-depth enough to
    recover any private state. {UUID: {attr: [value]}}
    '''
    
    async def __aenter__(self) -> Self:
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        pass

    @property
    def static(self):
        return self.state[UUID(int=0)]

    async def bootstrap(self) -> AsyncIterable[tuple[str, Bindings, Bindings | None]]:
        return
        yield

@dataclass(frozen=True)
class Sync:
    '''
    A synchronization which coordinates concepts by pattern-matching over
    actions within a flow and invokes new actions.
    '''

    name: str
    '''A readable name for debug; not globally unique.'''
    purpose: str
    '''What does the sync do?'''

    _: KW_ONLY

    when: list[When] = field(default_factory=list)
    '''When clauses, determining when and what it should match.'''
    where: list[Where] = field(default_factory=list)
    '''Where clause, refine bindings to zero or more binding sets.'''
    then: list[Then] = field(default_factory=list)
    '''Then clause, lists actions to invoke with their parameters.'''

    def ipld_model(self):
        return {
            "name": self.name,
            "purpose": self.purpose,
            "when": [when.ipld_model() for when in self.when],
            "where": [where.ipld_model() for where in self.where],
            "then": [then.ipld_model() for then in self.then]
        }
    
    def ipld_block(self) -> bytes:
        '''Return the object as an IPLD block.'''
        return dagcbor.marshal(self.ipld_model())
    
    @cached_property
    def cid(self):
        return CIDv1.hash(self.ipld_block())

class Trigger(IPLDRoot):
    '''Record of a sync which was triggered and why.'''
    
    flow: UUID
    '''Flow the trigger occurs within.'''
    sync: CID
    '''Sync initiated by the trigger.'''
    completions: dict[CID, 'Completion']
    '''Completions which caused the trigger.'''

    def ipld_model(self):
        return {
            "flow": str(self.flow),
            "sync": self.sync,
            "completions": list(sorted(self.completions))
        }

class Completion(IPLDRoot):
    '''A record of an action being completed.'''

    trigger: Trigger | None
    '''The trigger for the completion. None for bootstrap actions.'''
    flow: UUID
    '''Flow the completion occurred within.'''
    action: ActionId
    '''The name of the action being completed.'''
    params: Bindings
    '''Params passed to the action.'''
    result: Bindings
    '''Result of the completion.'''
    state: Mapping[str, Bindings]
    '''State of the concept after the completion.'''

    def ipld_model(self):
        return {
            "trigger": None if self.trigger is None else self.trigger.cid,
            "flow": str(self.flow),
            "action": self.action,
            "params": self.params,
            "result": self.result,
            "state": self.state
        }

class Flow:
    '''State for tracking the processing of a flow.'''

    uuid: UUID
    '''Flow's UUID.'''
    completions: dict[CID, Completion]
    '''All completions in the flow.'''
    triggers: dict[CID, Trigger]
    '''All triggers in the flow.'''
    matches: defaultdict[CID, defaultdict[int, set[CID]]]
    '''Mapping between syncs and the completions available for matching.'''

    def __init__(self):
        super().__init__()
        self.uuid = UUID7().uuid7
        self.completions = {}
        self.triggers = {}
        self.matches = defaultdict(lambda: defaultdict(set))

@dataclass
class ConceptEntry:
    concept: Concept
    '''The concept itself.'''
    bootstrap: asyncio.Task | None = None
    '''The task for processing bootstrap events.'''

async def queue_consumer[T](q: asyncio.Queue[T]):
    '''
    Async generator which waits until something is pushed to the queue
    and then yields everything in it.
    '''
    while True:
        yield await q.get()
        while not q.empty():
            yield q.get_nowait()

@overload
def freeze_value(value: Mapping[str, value_t] | MutableMapping[str, mutable_t]) -> Mapping[str, value_t]: ...
@overload
def freeze_value(value: Sequence[value_t] | MutableSequence[mutable_t]) -> Sequence[value_t]: ...
@overload
def freeze_value[T: simple_t](value: T) -> T: ...

def freeze_value(value: value_t | mutable_t) -> value_t:
    match value:
        case dict():
            return {k: freeze_value(v) for k, v in value}
        case list():
            return [freeze_value(v) for v in value]
        case _:
            return value

def freeze_state(value: State) -> Mapping[str, Bindings]:
    return {str(k): freeze_value(v) for k, v in value.items()}

class Engine:
    concepts: dict[ConceptId, ConceptEntry]
    '''All loaded concepts.'''
    syncs: dict[CID, Sync]
    '''All loaded syncs.'''
    syncdeps: defaultdict[ActionId, list[Sync]]
    '''An index of syncs by their action dependencies.'''
    funcs: dict[str, Callable]
    '''Functions loaded by the engine.'''
    flows: dict[UUID, Flow]
    '''Flows being processed.'''
    queue: asyncio.Queue[Completion]
    '''Queue of completions which have not yet been processed.'''
    tg: asyncio.TaskGroup
    '''Group of all running bootstrap tasks.'''
    state: dict[str, State]
    '''App state loaded by the engine.'''

    def __init__(self,
            state: dict[str, State],
            concepts: list[Concept],
            syncs: list[Sync]
        ):
        super().__init__()
        self.concepts = {c.name: ConceptEntry(c) for c in concepts}
        self.syncs = {}
        self.syncdeps = defaultdict(list)
        self.funcs = {}
        self.flows = {}
        self.queue = asyncio.Queue()
        self.tg = asyncio.TaskGroup()
        self.state = state

        for sync in syncs:
            self.load_sync(sync)

    def resolve_action(self, action: ActionId) -> tuple[Concept, ActionId]:
        c, a = action.rsplit('/', 1)
        if ce := self.concepts.get(c):
            con = ce.concept
            if act := con.actions.get(a):
                return con, act
            elif a in con.events:
                raise EventInvoked(action)
            else:
                raise NoSuchAction(action)
        else:
            raise NoSuchConcept(c)

    async def _concept_bootstrap(self, ent: ConceptEntry):
        '''Run a concept's bootstrap task which yields completions.'''

        async with ent.concept as c:
            cn = c.name
            c.state = self.state.get(cn, {})
            async for evt, inp, out in c.bootstrap():
                try:
                    res: Bindings | None = await getattr(c, evt)(**inp)
                    if res is None:
                        result = {} if out is None else out
                    elif out is None:
                        result = {} if res is None else res
                    else:
                        result = {**out, **res}
                except Exception as e:
                    result = {"error": {
                        "type": type(e).__name__,
                        "message": e.args[0],
                        "traceback": traceback.format_tb(e.__traceback__)
                    }}
                
                await self.queue.put(Completion(
                    trigger=None,
                    flow=self.new_flow().uuid,
                    action=f"{cn}/{evt}",
                    params=inp,
                    result=result,
                    state=freeze_state(c.state)
                ))
        ent.bootstrap = None

    def load_concept(self, concept: Concept):
        '''Load a new concept.'''

        self.concepts[concept.name] = ent = ConceptEntry(concept)
        ent.bootstrap = self.tg.create_task(self._concept_bootstrap(ent))
    
    def load_sync(self, sync: Sync):
        '''Load a new sync.'''

        self.syncs[sync.cid] = sync
        for act in set(when.action for when in sync.when):
            self.syncdeps[act].append(sync)

    def new_flow(self):
        flow = Flow()
        self.flows[flow.uuid] = flow
        return flow

    async def invoke(self,
            trigger: Trigger | None,
            flow: Flow,
            action: ActionId,
            **params: value_t
        ):
        '''Invoke an action and return its result.'''
        try:
            con, act = self.resolve_action(action)
            try:
                result = await getattr(con, act)(**params)
            except Exception as e:
                result = {"error": {
                    "type": type(e).__name__,
                    "message": e.args[0],
                    "traceback": traceback.format_tb(e.__traceback__)
                }}
            
            await self.queue.put(Completion(
                trigger=trigger,
                flow=flow.uuid,
                action=action,
                params=params,
                result=result,
                state=freeze_state(con.state)
            ))
            return result
        except Exception as err:
            #await self.uncaught_error(flow, err)
            raise

    async def bootstrap(self,
            action: ActionId,
            **params: value_t
        ) -> Bindings:
        '''Invoke an action ex nihilo.'''
        return await self.invoke(None, self.new_flow(), action, **params)
    
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

        completions = dict[CID, Completion]()

        for candi, when in zip(candidate, sync.when):
            comp = flow.completions[candi]
            if (bs := when.match(comp, bindings)) is None:
                # Match failed
                return False
            
            bindings.update(bs)
            completions[candi] = comp
        
        # [where]

        bss: set[ImmutablePromise[Bindings]] = {ImmutablePromise(bindings)}
        for where in sync.where:
            # For each binding we have thus far, we want to expand them
            # into the sets of bindings
            nbss: list[set[ImmutablePromise[Bindings]]] = []
            for base in bss:
                match where:
                    # Query concept state
                    case Query(concept=concept, pattern=pattern):
                        c = self.concepts[concept].concept
                        bs = {**base.value}
                        if not match_state(pattern, cast(dict[str, Bindings], c.state), bs):
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

        trigger = Trigger(
            flow=flow.uuid,
            sync=sync.cid,
            completions=completions
        )

        for bs in bss:
            for then in sync.then:
                # Completion already registered, we don't need the result
                await self.invoke(trigger, flow, then.action, **bs.value)
        
        return True

    async def run(self):
        '''Start the engine to process flows.'''
        
        async with self.tg:
            for ent in self.concepts.values():
                ent.bootstrap = self.tg.create_task(self._concept_bootstrap(ent))
            
            async for cmp in queue_consumer(self.queue):
                flow = self.flows[cmp.flow]
                flow.completions[cmp.cid] = cmp
                # Check all syncs dependent on this action
                for dep in self.syncdeps[cmp.action]:
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