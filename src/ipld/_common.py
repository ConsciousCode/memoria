from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Self

if TYPE_CHECKING:
    from .cid import CID, CIDv0, CIDv1

__all__ = (
    'IPLData', '_encodec', '_decodec', 'Immutable'
)

type IPLData = Mapping[str, IPLData]|Iterable[IPLData]|CID|CIDv0|CIDv1|bytes|str|int|float|bool|None

def _encodec(name: str):
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

def _decodec(name: str):
    def staged(pre_decode: Callable[[Any], IPLData]):
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

class Immutable:
    def __setattr__(self, name, value, /) -> None:
        raise TypeError(type(self).__name__ + " objects are immutable.")
    
    def __delattr__(self, name, /) -> None:
        raise TypeError(type(self).__name__ + " objects are immutable.")
    
    def __copy__(self) -> Self:
        return self
    
    def __deepcopy__(self, memo) -> Self:
        return self