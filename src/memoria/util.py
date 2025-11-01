from _typeshed import SupportsRichComparison
from typing import Callable, NoReturn, Protocol
from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import wraps
from heapq import heappop
import sys
from typing_extensions import overload

class JSONStructure(Protocol):
    def __json__(self) -> "json_t": ...

type json_t = JSONStructure|Mapping[str, json_t]|Sequence[json_t]|str|int|float|bool|None

type nonempty_tuple[T] = tuple[T, *tuple[T]]

def todo_iter[C, T](fn: Callable[[C], T]):
    '''
    Iterate over a stack, removing items as they are yielded. This can be
    appended to during iteration.
    '''
    @wraps(fn)
    def wrapper(todo: C) -> Iterable[T]:
        while todo: yield fn(todo)
    return wrapper

@todo_iter
def todo_set[T](todo: set[T]):
    return todo.pop()

@todo_iter
def todo_list[T](todo: list[T]):
    return todo.pop(0)

class RichComparison[T](Protocol):
    def __lt__(self, other: T, /) -> bool: ...
    def __gt__(self, other: T, /) -> bool: ...

@todo_iter
def todo_heap[T: SupportsRichComparison](todo: list[T]):
    return heappop(todo)

def set_pop[T](s: set[T], item: T) -> bool:
    '''
    Remove an item from a set if it exists, returning True if it was present.
    '''
    if item in s:
        s.remove(item)
        return True
    return False

@overload
def finite(f: None, /) -> None: pass
@overload
def finite(f: int|float|str, /) -> float: pass

def finite(f: int|float|str|None, /) -> float|None:
    '''Return a finite float, or 0.0 if the input is NaN or infinite.'''
    if f is None:
        return None
    f = float(f)
    if f != f or f == float('inf') or f == float('-inf'):
        return 0.0
    return f

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
