from typing import Iterable, Literal, Mapping, cast
import hashlib
import json
import base64

import base58
import cbor2
import multibase
import multicodec

from .cid import CID, CIDv0, CIDv1, Codec

__all__ = (
    'HashName', 'IPLData',
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

type IPLData = Mapping[str, IPLData]|Iterable[IPLData]|CIDv1|bytes|str|int|float|bool|None

def dagjson_marshal(data: IPLData) -> str:
    """
    Marshal a dictionary into DAG-JSON format.
    
    Args:
        data: The dictionary to marshal.
    
    Returns:
        A JSON string representing the DAG-JSON format.
    """

    def transform(data: IPLData) -> IPLData:
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

            case CIDv0():
                return {"/": base58.b58encode(data.buffer)}
            
            case CIDv1():
                return {"/": multibase.encode('base32', data.buffer)}

            case _:
                raise TypeError(f'Unsupported type in DAG-JSON: {type(data)}')
    
    return json.dumps(transform(data), sort_keys=True)

def dagjson_unmarshal(data: str) -> IPLData:
    """
    Unmarshal a JSON string into a dictionary in DAG-JSON format.
    
    Args:
        data: The JSON string to unmarshal.
    
    Returns:
        A dictionary representing the DAG-JSON format.
    """
    
    def transform(data: IPLData):
        match data:
            case None | bool() | int() | float() | str():
                return data
            
            case {"/": {"bytes": bs}}:
                if isinstance(bs, bytes):
                    return base64.b64decode(bs)
                raise TypeError('{"/": {"bytes": ...}} Expected bytes, got ' + str(type(bs)))
            
            case {"/": link}:
                if not isinstance(link, str):
                    raise TypeError('{"/": ...} Expected string for CID, got ' + str(type(link)))
                
                if link.startswith('Qm'):
                    raise TypeError('DAG-JSON does not support CIDv0, use CIDv1 instead')
                elif link.startswith('b'):
                    return CIDv1(
                        cast(Codec, multicodec.get_codec(link)),
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

def dagcbor_marshal(data: IPLData) -> bytes:
    def transform(data: IPLData):
        match data:
            case float() if data != data:
                raise ValueError('DAG-CBOR does not support NaN')
            case float() if data == INF:
                raise ValueError('DAG-CBOR does not support Infinity')
            case float() if data == -INF:
                raise ValueError('DAG-CBOR does not support -Infinity')
            
            case None | bool() | int() | float() | str() | bytes():
                return data

            case CIDv1():
                if data.codec != 'dag-cbor':
                    raise TypeError(f'DAG-CBOR only supports CIDv1 with codec "dag-cbor", got {data.codec}')
                return cbor2.CBORTag(LINK_TAG, data.buffer)
            
            case dict():
                return transform(data)
            
            case _:
                raise TypeError(f'Unsupported type in DAG-CBOR: {type(data)}')

    return cbor2.dumps(transform(data), canonical=True)

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
            
            case cbor2.CBORTag():
                if data.tag != LINK_TAG:
                    raise ValueError(f'DAG-CBOR forbids all tags except 42 (CID). Got {data.tag}')
                return CID(data.value)

            case dict():
                return {
                    k: transform(v)
                        for k, v in data.items()
                }
            
            case _:
                raise TypeError(f'Unsupported type in DAG-CBOR: {type(data)}')
    
    return transform(cbor2.loads(data))


def multihash(data: str|bytes, fn_name: HashName='sha2_256') -> str:
    """
    A utility function to make multihashing more convenient

    Returns:
        A base58 encoded digest of a hash (encoded in ascii)
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.new(fn_name, data).digest().decode('base58')