import json
from typing import Callable, Iterable, Mapping, Optional, Protocol, Self, Sequence, overload
import re
from functools import wraps
from heapq import heappop

from pydantic import BaseModel

from src.ipld import CID

class JSONStructure(Protocol):
    def __json__(self) -> "json_t": ...

type json_t = JSONStructure|Mapping[str, json_t]|Sequence[json_t]|str|int|float|bool|None

@overload
def model_dump(obj: BaseModel) -> json_t: ...
@overload
def model_dump[T](obj: T) -> T: ...

def model_dump(obj):
    try: return obj.model_dump()
    except AttributeError:
        return obj

def _fattr(x: json_t) -> str:
    if isinstance(x, str) and re.match(r"""[\s'"=</>&;]""", x):
        return json.dumps(x)
    else:
        return str(x)

def _fattrs(k: str, v: json_t) -> Iterable[str]:
    match v:
        case None|False: pass
        case True: yield k
        case list():
            for x in v:
                yield from _fattrs(k, x)
        case _:
            yield f'{k}={_fattr(v)}'

_xents = str.maketrans({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&apos;",
    '"': "&quot;"
})

def X(tag: str, **props: json_t):
    def X_Content(content: str):
        pv = []
        for k, v in props.items():
            pv.extend(_fattrs(k, v))
        
        p = f' {" ".join(pv)}' if pv else ""
        return f"<{tag}{p}>{content.translate(_xents)}</{tag}>"
    
    return X_Content

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

@todo_iter
def todo_heap[T](todo: list[T]):
    return heappop(todo)

def set_pop[T](s: set[T], item: T) -> bool:
    '''
    Remove an item from a set if it exists, returning True if it was present.
    '''
    if item in s:
        s.remove(item)
        return True
    return False

class Lexicographic(Protocol):
    def __lt__(self, other: Self, /) -> bool: ...

class LeastT:
    def __init__(self):
        raise NotImplementedError("LeastT cannot be instantiated directly")
    
    def __lt__(self, other: object, /):
        return True
    
    def __gt__(self, other: object, /):
        return False
    
    def __eq__(self, other: object, /):
        return isinstance(other, LeastT)

Least = LeastT.__new__(LeastT)

@overload
def ifnone[A](arg0: A, /) -> A: ...
@overload
def ifnone[A, B](arg0: Optional[A], arg1: B) -> A|B: ...
@overload
def ifnone[A, B, C](arg0: Optional[A], arg1: Optional[B], arg2: C) -> A|B|C: ...
@overload
def ifnone[A, B, C, D](arg0: Optional[A], arg1: Optional[B], arg2: Optional[C], arg3: D) -> A|B|C|D: ...

def ifnone[*Ts](*args: *Ts): # type: ignore
    '''Return the first non-None argument, or None if all are None.'''
    for arg in args:
        if arg is not None:
            return arg
    return None

def finite(f) -> float:
    '''Return a finite float, or 0.0 if the input is NaN or infinite.'''
    f = float(f)
    if f != f or f == float('inf') or f == float('-inf'):
        return 0.0
    return f

def error(e: BaseException):
    raise e

def cidstr(c: bytes) -> str:
    """
    Convert a CID byte string to a base32-encoded string.
    
    Args:
        c: The CID byte string.
    
    Returns:
        A base32-encoded string representation of the CID.
    """
    return CID(c).encode()

class classproperty[T, R]:
    def __init__(self, fn: Callable[[type[T]], R]):
        self.fn = fn
    
    def __get__(self, instance, owner: type[T]) -> R:
        return self.fn(owner)