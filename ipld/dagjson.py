import base64
import base58
import json

from . import multibase
from .cid import CID, CIDv0, CIDv1
from .ipld import _encodec, _decodec, IPLData

@_encodec("DAG-JSON")
def _dagjson_transform(data):
    match data:
        case bytes():
            return {"/": {"bytes": base64.b64encode(data).decode('utf-8')}}
        case CIDv0():
            return {"/": base58.b58encode(data.buffer).decode('utf-8')}
        case CIDv1():
            return {"/": multibase.encode('base32', data.buffer)}
        case {"/": {"bytes": bs}}:
            if isinstance(bs, str):
                return base64.b64decode(bs.encode('utf-8'))
            raise TypeError(
                '{"/": {"bytes": ...}} expected str, got ' + type(bs).__name__
            )
        
        case {"/": link}:
            if isinstance(link, str):
                return CID(link)
            raise TypeError("DAG-JSON doesn't support '/' keys")

@_decodec("DAG-JSON")
def _dagjson_decode(data: IPLData) -> IPLData:
    match data:
        case {"/": {"bytes": bs}}:
            if isinstance(bs, bytes):
                return base64.b64decode(bs)
            raise TypeError(
                '{"/": {"bytes": ...}} Expected bytes, got ' + str(type(bs))
            )
        
        case {"/": link}:
            if isinstance(link, str):
                return CID(link)
            raise TypeError(
                '{"/": ...} Expected string for CID, got ' + str(type(link))
            )

def marshal(data: IPLData) -> str:
    """
    Convert data to DAG-JSON format.
    
    Args:
        data: The data to convert.
    
    Returns:
        The data in DAG-JSON format.
    """
    return json.dumps(_dagjson_transform(data), sort_keys=True)

def unmarshal(data: str) -> IPLData:
    """
    Convert DAG-JSON format data back to its original form.
    
    Args:
        data: The DAG-JSON formatted string.
    
    Returns:
        The original data structure.
    """
    return _dagjson_decode(json.loads(data))