import asyncio
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property, wraps
import inspect
from typing import Any, AsyncIterable, Awaitable, Callable, ClassVar, Mapping, MutableMapping, MutableSequence, Self, cast, overload, override
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

type State = MutableMapping[str, MutableMapping[str, mutable_t]]
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
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return repr(self.value)
    
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
class StimInvoked(HyperSyncError):
    '''Attempted to invoke a stimulus.'''

class UnloadedConcept(NoSuchConcept):
    '''
    Attempted to invoke an action on a concept in the process of being
    unloaded.
    '''

type ConceptId = str # concept
type ActionId = str # concept/action
type StrUUID = str
type FlowId = StrUUID

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

    concept: ConceptId | Var
    '''Name of the concept being queried.'''
    pattern: dict[Var, Multipattern]
    '''Pattern used to query the concept's state.'''

    def ipld_model(self):
        return {
            "concept": dump_pattern(self.concept),
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

    action: ActionId | Var
    '''Action to invoke.'''
    params: Template
    '''The parameters to invoke the action.'''

    def resolve(self, env: Bindings):
        '''Resolve the parameters with the environment.'''

        @overload
        def inner(bob: Template) -> Bindings: ...
        @overload
        def inner(bob: multibind_t) -> value_t: ...

        def inner(bob: multibind_t) -> value_t:
            match bob:
                case Var(name):
                    return env[name]
                case dict():
                    return {
                        str(env[k.name] if isinstance(k, Var) else k): inner(v)
                            for k, v in bob.items()
                    }
                case list():
                    return [inner(v) for v in bob]
                case None|bool()|int()|float()|str()|bytes()|CID():
                    return bob
                case _:
                    assert False
        
        return inner(self.params)

    def ipld_model(self):
        return {
            "action": dump_pattern(self.action),
            "params": dump_pattern(self.params)
        }

def ignore_extra(func: Callable):
    '''Process the function as if it has a dummy **kwargs parameter.'''
    sig = inspect.signature(func)
    accepted = set[str]()
    
    for name, param in sig.parameters.items():
        if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
            accepted.add(name)
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            # Don't need to ignore if there's a kwargs
            return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **{
            k: v for k, v in kwargs.items() if k in accepted
        })

    return wrapper

def action[**P](name: Callable[P, Awaitable[Mapping[str, object]]] | str):
    def inner(func: Callable[P, Awaitable[Mapping[str, object]]]):
        func._action_name = name # pyright: ignore [reportFunctionMemberAccess]
        return ignore_extra(func)
    
    if isinstance(name, str):
        return inner
    
    func, name = name, name.__name__
    return inner(func)

def stimulus[**P](name: Callable[P, Awaitable[Mapping[str, object] | None]] | str):
    def inner(func: Callable[P, Awaitable[Mapping[str, object] | None]]):
        func._stim_name = name # pyright: ignore [reportFunctionMemberAccess]
        return ignore_extra(func)
    
    if isinstance(name, str):
        return inner
    
    func, name = name, name.__name__
    return inner(func)

class ConceptMeta(type):
    '''Aggregates actions and events from decorators.'''

    @override
    def __new__(cls, name: str, bases: tuple, dct: dict):
        dct['actions'] = actions = {}
        dct['stimuli'] = stims = {}
        for k, v in dct.items():
            if (n := getattr(v, "_action_name", None)):
                actions[n] = k
                del v._action_name
            if (n := getattr(v, "_stim_name", None)):
                stims[n] = k
                del v._stim_name
        
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

class Concept[L, S=State](metaclass=ConceptMeta):
    '''Stateful locus of behavior.'''

    # Provided by metaclass

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
    stimuli: ClassVar[dict[str, str]]
    '''Specifications of each stimulus available on the object.'''

    # Overloadable (by default calculated from __name__ and __doc__)

    name: ClassVar[str]
    '''The concept's name.'''
    purpose: ClassVar[str]
    '''What value does this concept add?'''

    # Proper instance members
    
    state: State
    '''
    The concept's public, queryable state. This must be in-depth enough to
    recover any private ephemeral state. {UUID: {attr: [value]}}
    '''

    @property
    def static(self):
        # Concept static state should always be initialized with a UUID
        return cast(S, self.state[str(UUID(int=0))])
    
    @property
    def local(self):
        return cast(dict[str, L], self.state)
    
    # By default do nothing
    
    async def __aenter__(self) -> Self:
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        pass

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
    
    flow: FlowId
    '''Flow UUID the trigger occurs within.'''
    sync: CID
    '''Sync initiated by the trigger.'''
    completions: dict[CID, 'Completion']
    '''Completions which caused the trigger.'''

    def ipld_model(self):
        return {
            "flow": self.flow,
            "sync": self.sync,
            "completions": list(sorted(self.completions))
        }

class Completion(IPLDRoot):
    '''A record of an action being completed.'''

    action: ActionId
    '''The name of the action being completed.'''
    params: Bindings
    '''Params passed to the action.'''
    result: Bindings
    '''Result of the completion.'''
    state: Mapping[str, Bindings]
    '''State of the concept after the completion.'''
    flow: FlowId
    '''Flow UUID the completion occurred within.'''
    trigger: CID | None
    '''The trigger for the completion. None for stimuli.'''

    def ipld_model(self):
        return {
            "action": self.action,
            "params": self.params,
            "result": self.result,
            "state": self.state,
            "flow": self.flow,
            "trigger": self.trigger
        }

class Flow:
    '''State for tracking the processing of a flow.'''

    uuid: FlowId
    '''Flow's UUID.'''
    completions: dict[CID, Completion]
    '''All completions in the flow.'''
    triggers: dict[CID, Trigger]
    '''All triggers in the flow.'''
    matches: defaultdict[CID, defaultdict[int, set[CID]]]
    '''Mapping between syncs and the completions available for matching.'''

    def __init__(self):
        super().__init__()
        self.uuid = str(UUID7().uuid7)
        self.completions = {}
        self.triggers = {}
        self.matches = defaultdict(lambda: defaultdict(set))

@dataclass
class ConceptEntry:
    concept: Concept
    '''The concept itself.'''
    bootstrap: asyncio.Task | None = None
    '''The task for processing stimuli.'''

@overload
def freeze_value(value: Mapping[str, value_t] | MutableMapping[str, mutable_t]) -> Mapping[str, value_t]: ...
@overload
def freeze_value(value: Sequence[value_t] | MutableSequence[mutable_t]) -> Sequence[value_t]: ...
@overload
def freeze_value[T: simple_t](value: T) -> T: ...

def freeze_value(value: value_t | mutable_t) -> value_t:
    match value:
        case dict():
            return {k: freeze_value(v) for k, v in value.items()}
        case list():
            return [freeze_value(v) for v in value]
        case _:
            return value

def freeze_state(value: State) -> Mapping[str, Bindings]:
    return {str(k): freeze_value(v) for k, v in value.items()}

def format_error(e: Exception):
    return {
        "type": type(e).__name__,
        "message": str(e.args[0]),
        "traceback": traceback.format_tb(e.__traceback__)
    }

class Engine:
    concepts: dict[ConceptId, ConceptEntry]
    '''All loaded concepts.'''
    syncs: dict[CID, Sync]
    '''All loaded syncs.'''
    syncdeps: defaultdict[ActionId, list[Sync]]
    '''An index of syncs by their action dependencies.'''
    funcs: dict[str, Callable]
    '''Functions loaded by the engine.'''
    flows: dict[FlowId, Flow]
    '''Flows being processed.'''
    queue: asyncio.Queue[Completion]
    '''Queue of completions which have not yet been processed.'''
    tg: asyncio.TaskGroup
    '''Group of all running bootstrap tasks.'''
    state: dict[str, State]
    '''App state loaded by the engine.'''

    def __init__(self,
            state: dict[str, State],
            concepts: list[Concept[Any, Any]],
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
    
    # Methods intended to be overridden to report meta-events
    # These effectively serve as hypersyncs or hyperconcept pivots

    async def bootstrap(self,
            concept: Concept
        ) -> AsyncIterable[tuple[str, Bindings, Bindings|None]]:
        '''
        Bootstrap a concept. Yields tuples of a stimulus on the concept,
        params to invoke it with, and the result of the stimulus. The stimulus
        is called and then the results are shallow merged.
        '''
        async for b in concept.bootstrap():
            yield b

    async def uncaught(self,
            action: ActionId,
            params: Bindings,
            error: Exception,
            flow: FlowId,
            trigger: CID | None = None
        ) -> Bindings | None:
        '''
        An unknown exception was thrown. This is always a programming error.
        Returns the result to use in the action's completion or None to leave
        it incomplete.
        '''
        return {"error": format_error(error)}
    
    async def ignored(self, cmp: Completion):
        '''
        An action was completed but it wasn't matched by any sync [when]
        clauses. There are always terminal actions, but it can indicate an
        overlooked edge case. For instance, matching an error completion.
        '''
    
    async def stim_invoked(self,
            stim: ActionId,
            params: Bindings,
            flow: FlowId,
            trigger: CID | None
        ) -> Bindings | None:
        '''
        A stimulus was invoked by a sync which is invalid. Return either the
        result to give the action completion or None to leave it incomplete.
        '''
    
    async def no_action(self,
            action: ActionId,
            params: Bindings,
            flow: FlowId,
            trigger: CID | None
        ) -> Bindings | None:
        '''
        An action was invoked which does not exist on an existant concept.
        Return either the result to give the action completion or None to
        leave it incomplete.
        '''
        return None

    async def no_concept(self,
            concept: str,
            params: Bindings,
            flow: FlowId,
            trigger: CID | None
        ) -> Bindings | None:
        '''
        A nonexistant concept was referenced. Return either the result to give
        the action completion or None to leave it incomplete.
        '''
        return None

    # Public methods

    async def complete(self,
            action: ActionId,
            params: Bindings,
            result: Bindings,
            state: Mapping[str, Bindings] | None = None,
            flow: FlowId|None = None,
            trigger: CID|None = None
        ):
        '''Submit an action completion.'''
        if state is None:
            c, _ = action.rsplit("/", 1)
            state = self.concepts[c].concept.state

        await self.queue.put(Completion(
            trigger=trigger,
            flow=flow or self._new_flow().uuid,
            action=action,
            params=params,
            result=result,
            state=state
        ))

    async def invoke(self,
            action: ActionId,
            params: Bindings,
            flow: FlowId |  None = None,
            trigger: CID | None = None
        ) -> Bindings | None:
        '''
        Invoke an action and return its result. If None is returned, this
        indicates an error where it could not be completed.
        '''
        if flow is None:
            flow = self._new_flow().uuid
        try:
            con, act = self._resolve_action(action)
        except StimInvoked as e:
            return await self.stim_invoked(action, params, flow, trigger)
        except NoSuchAction as e:
            return await self.no_action(action, params, flow, trigger)
        except NoSuchConcept as e:
            return await self.no_concept(action, params, flow, trigger)
        
        try:
            result: Bindings = await getattr(con, act)(flow=flow, **params)
        except Exception as e:
            return await self.uncaught(action, params, e, flow, trigger)
        
        await self.complete(
            action, params, result,
            freeze_state(con.state),
            flow=flow,
            trigger=trigger
        )
        return result

    def load_concept(self, concept: Concept):
        '''
        Load a new concept. This must be 
        '''

        self.concepts[concept.name] = ent = ConceptEntry(concept)
        ent.bootstrap = self.tg.create_task(self._concept_bootstrap(ent))
    
    def load_sync(self, sync: Sync):
        '''Load a new sync.'''

        self.syncs[sync.cid] = sync
        for act in set(when.action for when in sync.when):
            self.syncdeps[act].append(sync)

    def _resolve_action(self, action: ActionId) -> tuple[Concept, str]:
        '''Resolve an ActionId to a concept and the name of the method.'''
        c, a = action.rsplit('/', 1)
        if ce := self.concepts.get(c):
            con = ce.concept
            if act := con.actions.get(a):
                return con, act
            elif a in con.stimuli:
                raise StimInvoked(action)
            else:
                raise NoSuchAction(action)
        else:
            raise NoSuchConcept(c)

    async def _concept_bootstrap(self, ent: ConceptEntry):
        '''Run a concept's bootstrap task which yields completions.'''

        async with ent.concept as c:
            cn = c.name
            if (state := self.state.get(cn)) is None:
                # Bootstrap with a UUID
                state: State|None = {
                    str(UUID(int=0)): {"uuid": str(UUID7().uuid7)}
                }
            c.state = state
            async for event, result, out in self.bootstrap(c):
                flow = self._new_flow().uuid
                try:
                    res: Bindings | None = await getattr(c, event)(
                        flow=flow, **result
                    )
                    if res is None:
                        result = {} if out is None else out
                    elif out is None:
                        result = {} if res is None else res
                    else:
                        result = {**out, **res}
                except Exception as e:
                    result = await self.uncaught(event, result, e, flow)
                    if result is None:
                        continue
                
                await self.complete(
                    f"{cn}/{event}", result, result,
                    freeze_state(c.state),
                    flow=flow
                )
        ent.bootstrap = None

    def _new_flow(self):
        flow = Flow()
        self.flows[flow.uuid] = flow
        return flow

    async def _trigger(self,
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

        completions: dict[CID, Completion] = {}

        for candi, when in zip(candidate, sync.when):
            comp = flow.completions[candi]
            if (bs := when.match(comp, bindings)) is None:
                # Match failed
                return False
            
            bindings.update(bs)
            completions[candi] = comp
        
        # [where]

        bss = {ImmutablePromise(bindings)}
        for where in sync.where:
            # For each binding we have thus far, we want to expand them
            # into the sets of bindings
            nbss: list[set[ImmutablePromise[Bindings]]] = []
            for base in bss:
                match where:
                    # Query concept state
                    case Query(concept=concept, pattern=pattern):
                        bs = {**base.value}
                        if isinstance(concept, Var):
                            if not isinstance(concept := bs[concept.name], str):
                                continue
                        c = self.concepts[concept].concept
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

        tcid = trigger.cid
        flow.triggers[tcid] = trigger

        for bs in bss:
            for then in sync.then:
                if isinstance(action := then.action, Var):
                    if not isinstance(action := bindings[action.name], str):
                        # Todo: Report invalid?
                        continue
                
                # Completion already registered, we don't need the result
                await self.invoke(
                    action, then.resolve(bs.value),
                    flow=flow.uuid,
                    trigger=tcid
                )
        
        return True

    async def run(self):
        '''Start the engine to process flows.'''
        
        async with self.tg:
            for ent in self.concepts.values():
                ent.bootstrap = self.tg.create_task(self._concept_bootstrap(ent))
            
            while True:
                cmp = await self.queue.get()
                flow = self.flows[cmp.flow]
                flow.completions[cmp.cid] = cmp

                if not (deps := self.syncdeps.get(cmp.action)):
                    await self.ignored(cmp)
                    continue
                
                # Check all syncs dependent on this action
                for dep in deps:
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

                        if await self._trigger(flow, dep, candidate):
                            # Successfully processed, don't try any more
                            break