import json

from . import multibase
from .cid import CID, CIDv0, CIDv1
from ._common_ipld import _encodec, _decodec, IPLData

__all__ = ("marshal", "unmarshal")

@_encodec("DAG-JSON")
def _dagjson_encode(data):
    match data:
        case bytes():
            return {"/": {"bytes": multibase.base64.encode(data)}}
        case CIDv0():
            return {"/": multibase.base58.encode(data.buffer)}
        case CIDv1():
            return {"/": multibase.encode('base32', data.buffer)}
        case {"/": _}:
            raise TypeError("DAG-JSON doesn't support '/' keys")

@_decodec("DAG-JSON")
def _dagjson_decode(data: IPLData) -> IPLData:
    match data:
        case {"/": {"bytes": bs}}:
            if isinstance(bs, str):
                return multibase.base64.decode(bs)
            raise TypeError(
                '{"/": {"bytes": ...}} Expected bytes, got ' + type(bs).__name__
            )
        
        case {"/": link}:
            if isinstance(link, str):
                return CID(link)
            raise TypeError(
                '{"/": ...} Expected string for CID, got ' + type(link).__name__
            )

def marshal(data: IPLData) -> str:
    """
    Convert data to DAG-JSON format.
    
    Args:
        data: The data to convert.
    
    Returns:
        The data in DAG-JSON format.
    """
    return json.dumps(_dagjson_encode(data), sort_keys=True)

def unmarshal(data: str|bytes) -> IPLData:
    """
    Convert DAG-JSON format data back to its original form.
    
    Args:
        data: The DAG-JSON formatted string.
    
    Returns:
        The original data structure.
    """
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    return _dagjson_decode(json.loads(data))