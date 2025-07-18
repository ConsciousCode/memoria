'''
Utility module for creating immutable objects.
'''

from typing import Self

class Immutable:
    '''A base class for immutable objects. Raises on any attempt to modify.'''
    def __setattr__(self, name, value, /) -> None:
        raise TypeError(type(self).__name__ + " objects are immutable.")
    
    def __delattr__(self, name, /) -> None:
        raise TypeError(type(self).__name__ + " objects are immutable.")
    
    def __copy__(self) -> Self:
        return self
    
    def __deepcopy__(self, memo) -> Self:
        return self