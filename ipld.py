from typing import Any, Iterable, Literal
import hashlib
import json
import base64

import base58
import cbor
import cid
import multibase
import multicodec

__all__ = (
    'HashName', 'IPLDModel',
    'dagjson_marshal', 'dagjson_unmarshal',
    'dagcbor_marshal', 'dagcbor_unmarshal',
    'multihash'
)

LINK_TAG = 42
'''DAG-CBOR tag for links.'''

INF = float('inf')

type HashName = Literal[
    'sha2_256', 'sha2_512', 'sha3_512', 'sha3', 'sha3_384',
    'sha3_256', 'sha3_224', 'shake_128', 'shake_256', 'blake2b',
    'blake2s'
]

type IPLDModel = dict[str, IPLDModel]|list[IPLDModel]|cid.CIDv0|cid.CIDv1|bytes|str|int|float|bool|None

def dagjson_marshal(data: IPLDModel) -> str:
    """
    Marshal a dictionary into DAG-JSON format.
    
    Args:
        data: The dictionary to marshal.
    
    Returns:
        A JSON string representing the DAG-JSON format.
    """

    def transform(data: IPLDModel) -> IPLDModel:
        match data:
            case float() if data != data:
                raise ValueError('DAG-JSON does not support NaN')
            case float() if data == INF:
                raise ValueError('DAG-JSON does not support Infinity')
            case float() if data == -INF:
                raise ValueError('DAG-JSON does not support -Infinity')
            
            case None | bool() | int() | float() | str():
                return data
            
            case bytes():
                return {"/": {"bytes": base64.b64encode(data)}}

            case list():
                return list(map(transform, data))
            
            case {"/": _}:
                raise ValueError('DAG-JSON does not support "/" key in dictionaries')
            
            case dict(di):
                data = {}
                for k, v in di.items():
                    if not isinstance(k, str):
                        raise TypeError(f'DAG-JSON forbids non-str keys, got {type(k)}')
                    
                    data[k] = transform(v)
                
                return data

            case cid.CIDv0():
                return {"/": base58.b58encode(data.buffer)}
            
            case cid.CIDv1():
                return {"/": multibase.encode('base32', data.buffer)}

            case _:
                raise TypeError(f'Unsupported type in DAG-JSON: {type(data)}')
    
    return json.dumps(transform(data), sort_keys=True)

def dagjson_unmarshal(data: str) -> IPLDModel:
    """
    Unmarshal a JSON string into a dictionary in DAG-JSON format.
    
    Args:
        data: The JSON string to unmarshal.
    
    Returns:
        A dictionary representing the DAG-JSON format.
    """
    
    def transform(data: Any) -> IPLDModel:
        match data:
            case None | bool() | int() | float() | str():
                return data
            
            case {"/": {"bytes": bs}}:
                return base64.b64decode(bs)
            
            case {"/": link}:
                if not isinstance(link, str):
                    raise TypeError(f'Expected string for CID, got {type(link)}')
                
                if link.startswith('Qm'):
                    return cid.CIDv0(base58.b58decode(link))
                elif link.startswith('b'):
                    return cid.CIDv1(
                        multicodec.get_codec(link),
                        multicodec.remove_prefix(link)
                    )
                else:
                    raise ValueError(f'Invalid CID format: {link}')

            case dict():
                return {k: transform(v) for k, v in data.items()}
            
            case list():
                return list(map(transform, data))
            
            case _:
                raise TypeError(f'Unsupported type in DAG-JSON: {type(data)}')
    
    return transform(json.loads(data))

def dagcbor_marshal(data: dict[str, Any]) -> str:
    def transform(data: dict[str, Any]) -> Iterable[tuple[str, Any]]:
        for k, v in data.items():
            if not isinstance(k, str):
                raise TypeError(f'DAG-CBOR forbids non-str keys, got {type(k)}')

            match v:
                case float() if v != v:
                    raise ValueError('DAG-CBOR does not support NaN')
                case float() if v == INF:
                    raise ValueError('DAG-CBOR does not support Infinity')
                case float() if v == -INF:
                    raise ValueError('DAG-CBOR does not support -Infinity')
                
                case cid.CIDv0() | cid.CIDv1():
                    v = cbor.Tag(LINK_TAG, v.buffer)
                case dict():
                    v = dict(transform(v))
            yield k, v

    return cbor.dumps(dict(transform(data)), sort_keys=True)


def dagcbor_unmarshal(data: bytes):
    def transform(data):
        match data:
            case float() if data != data:
                raise ValueError('DAG-CBOR does not support NaN')
            case float() if data == INF:
                raise ValueError('DAG-CBOR does not support Infinity')
            case float() if data == -INF:
                raise ValueError('DAG-CBOR does not support -Infinity')
            
            case None | bool() | int() | float() | str() | bytes():
                return data
            
            case list():
                return list(map(transform, data))
            
            case cbor.Tag():
                if data.tag != LINK_TAG:
                    raise ValueError(f'DAG-CBOR forbids all tags except 42 (CID). Got {data.tag}')
                return cid.from_bytes(data.value)

            case dict():
                return {
                    k: transform(v)
                        for k, v in data.items()
                }
            
            case _:
                raise TypeError(f'Unsupported type in DAG-CBOR: {type(data)}')
    
    return transform(cbor.loads(data))


def multihash(data: str|bytes, fn_name: HashName='sha2_256') -> str:
    """
    A utility function to make multihashing more convenient

    Returns:
        A base58 encoded digest of a hash (encoded in ascii)
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.new(fn_name, data).digest().decode('base58')