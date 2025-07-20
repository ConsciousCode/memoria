from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from typing import Mapping, Iterable
    from cid import CID, CIDv0, CIDv1

type IPLData = Mapping[str, IPLData]|Iterable[IPLData]|CID|CIDv0|CIDv1|bytes|str|int|float|bool|None
'''A type alias for IPLD-compatible data structures.'''

def encodec(name: str):
    '''Wrap common logic for dag-* encoding functions.'''
    def staged(pre_encode: Callable[[IPLData], Any]):
        @wraps(pre_encode)
        def transform(data: Any) -> IPLData:
            INF = float('inf')
            if (d := pre_encode(data)) is not None:
                return d
            
            match data:
                case float() if data != data:
                    raise ValueError('IPLD does not support NaN')
                case float() if data == INF:
                    raise ValueError('IPLD does not support Infinity')
                case float() if data == -INF:
                    raise ValueError('IPLD does not support -Infinity')
                
                case None | bool() | int() | float() | str():
                    return data
                
                case list(): return list(map(transform, data))
                case dict():
                    return {str(k): transform(v) for k, v in data.items()}
                
                case _: raise TypeError(
                    f'{name} Unsupported type in IPLData: {type(data)}'
                )
        return transform
    return staged

def decodec(name: str):
    '''Wrap common logic for dag-* decoding functions.'''
    def staged(pre_decode: Callable[[Any], IPLData]):
        @wraps(pre_decode)
        def transform(data: Any) -> IPLData:
            if (d := pre_decode(data)) is not None:
                return d

            match data:
                case None | bool() | int() | float() | str():
                    return data
                
                case list(): return list(map(transform, data))
                case dict():
                    return {str(k): transform(v) for k, v in data.items()}
                
                case _: raise TypeError(
                    f'{name} Unsupported type in IPLData: {type(data)}'
                 )
        return transform
    return staged

class FauxMapping:
    """A faux mapping class to act as a dictionary for __match_args__."""
    __match_args__: tuple[str, ...]

    def __getitem__(self, key: str):
        if key in self.__match_args__:
            return getattr(self, key)
        raise KeyError(key)
    
    def keys(self):
        yield from self.__match_args__
    
    def values(self):
        for key in self.__match_args__:
            yield getattr(self, key)
    
    def items(self):
        for key in self.__match_args__:
            yield key, getattr(self, key)