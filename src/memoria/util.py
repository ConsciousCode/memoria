from typing import Protocol, Self, override
from collections.abc import Mapping, Sequence

type json_t = None|bool|int|float|str|Sequence[json_t]|Mapping[str, json_t]

type nonempty_tuple[T] = tuple[T, *tuple[T, ...]]

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
    '''Wrap a value to make it compare the opposite it normally does.'''
    def __init__(self, wrap: T):
        self.wrap: T = wrap
    
    def __lt__(self, other: Self, /) -> bool:
        return self.wrap > other
    def __gt__(self, other: Self, /) -> bool:
        return self.wrap < other
