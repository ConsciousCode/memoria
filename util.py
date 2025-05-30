import json
from typing import Callable, Iterable
import re
from functools import wraps

type json_t = dict[str, json_t]|list[json_t]|str|int|float|bool|None

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

def set_pop[T](s: set[T], item: T) -> bool:
    '''
    Remove an item from a set if it exists, returning True if it was present.
    '''
    if item in s:
        s.remove(item)
        return True
    return False