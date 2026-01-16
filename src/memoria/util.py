from functools import cached_property
from typing import NoReturn, Protocol, Self, override
from collections.abc import Iterator, Mapping, Sequence
import sys

from pydantic import BaseModel

from ipld import dagcbor, IPLData
from cid import CID, CIDv1

type json_t = Mapping[str, json_t]|Sequence[json_t]|str|int|float|bool|None

type nonempty_tuple[T] = tuple[T, *tuple[T, ...]]

class IPLDModel(BaseModel):
    '''Base model for IPLD objects.'''

    def ipld_model(self) -> IPLData:
        '''Return the object as an IPLD model.'''
        return self.model_dump()

class IPLDRoot(IPLDModel):
    '''Base model for IPLD objects which can get a CID.'''

    def ipld_block(self) -> bytes:
        '''Return the object as an IPLD block.'''
        return dagcbor.marshal(self.ipld_model())

    @cached_property
    def cid(self):
        return CIDv1.hash(self.ipld_block())

class Link[T: IPLDRoot](CID):
    def __init__(self, cid: CID, model: type[T]):
        self.cid = cid
        self.model = model

class Lexicographic(Protocol):
    '''Protocol for lexicographical order.'''
    def __lt__(self, other: Self, /) -> bool: ...
    def __gt__(self, other: Self, /) -> bool: ...

class LeastT:
    def __init__(self):
        raise NotImplementedError("LeastT cannot be instantiated directly")
    
    def __lt__(self, other: object, /): return True
    def __gt__(self, other: object, /): return False
    @override
    def __eq__(self, other: object, /): return isinstance(other, LeastT)
    @override
    def __repr__(self): return "Least"

Least = LeastT.__new__(LeastT)
'''A singleton representing the least element in a lexicographical order.'''

class ReverseCmp[T: Lexicographic]:
    def __init__(self, wrap: T):
        self.wrap: T = wrap
    
    def __lt__(self, other: Self, /) -> bool:
        return self.wrap > other
    def __gt__(self, other: Self, /) -> bool:
        return self.wrap < other

def expected(name: str) -> NoReturn:
    raise ValueError(f"Expected a {name}.")

def warn(msg: str):
    print(f"Warning: {msg}", file=sys.stderr)

def check_overflow[T](rest: Sequence[T]):
    if rest: warn("Too many arguments.")

def parse_opts(arg: str) -> Iterator[tuple[str, str | None]]:
    """Parse command line options from a string."""
    if arg.startswith("--"):
        try:
            eq = arg.index('=')
            yield arg[2:eq], arg[eq+1:]
        except ValueError:
            yield arg[2:], None
    else:
        # short options
        for i in range(1, len(arg)):
            yield arg[i], None

def named_value(arg: str, it: Iterator[str]) -> str:
    try: return next(it)
    except StopIteration:
        expected(f"value after {arg}")

type ap_t = bool|int|type[int]|str|type[str]
def argparse(argv: tuple[str, ...], config: dict[str, ap_t]):
    which = dict[str, tuple[ap_t, str]]()
    for aliases, v in config.items():
        als = aliases.split(',')
        match [a for a in als if a.startswith("--")]:
            case []: name = None
            case [name]: name = name.removeprefix('--')
            case long: raise ValueError(f"Multiple long options found: {long}")

        for k in als:
            k = k.lstrip('-')
            which[k] = v, name or k
    
    pos = list[str]()
    opts: dict[str, ap_t] = {
        name: t
        for t, name in which.values()
            if isinstance(t, (bool, int, str))
    }

    it = iter(argv)
    try:
        while True:
            arg = next(it)
            if not arg.startswith("-"):
                pos.append(arg)
                continue
            
            if arg == "--":
                pos.extend(it)
                break
            
            for opt, val in parse_opts(arg):
                if (c := which.get(opt)) is None:
                    raise ValueError(f"Unknown option {opt!r}")
                
                t, name = c

                if isinstance(t, bool) or t is bool:
                    if val is not None:
                        raise ValueError(f"Option {opt!r} does not take a value")
                    opts[name] = not t
                elif isinstance(t, int) or t is int:
                    if val is None:
                        val = named_value(arg, it)
                    try: opts[name] = int(val)
                    except ValueError:
                        raise ValueError(f"Expected an integer after {arg!r}") from None
                elif isinstance(t, str) or t is str:
                    if val is None:
                        val = named_value(arg, it)
                    opts[name] = val
                else:
                    raise TypeError(f"Unsupported type {t} for option {arg!r}")
    except StopIteration:
        pass
    
    return pos, opts
