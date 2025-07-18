'''
DAG-CBOR encoding and decoding.
'''

import cbor2

import struct
import math
from typing import Any, Iterable

from ._common import decodec, IPLData
from .cid import CID

LINK_TAG = 42
'''DAG-CBOR tag for links.'''

# We can't use cbor2 for encoding because its canonical mode emits the
# smallest representation for floats instead of the required 64-bit
# representation. This also lets us skip pre-validation and transformation.

def _encode_item(obj: Any) -> Iterable[bytes]:
    match obj:
        case None: yield b'\xf6'
        case True: yield b'\xf5'
        case False: yield b'\xf4'
        case int(): yield from _encode_int(obj)
        case float(): yield from _encode_float(obj)
        case str(): yield from _encode_string(obj)
        case bytes(): yield from _encode_bytes(obj)
        case list(): yield from _encode_array(obj)
        case dict(): yield from _encode_map(obj)
        case CID(): yield from _encode_link(obj)

        case _:
            raise ValueError(f"Unsupported type: {type(obj)}")

def _encode_length(major: int, value: int) -> Iterable[bytes]:
    if value < 24:
        yield bytes([major << 5 | value])
    else:
        idx = ((value.bit_length() - 1) // 8)
        if idx > 3: idx = 3
        yield bytes([major << 5 | (24 + idx)])
        yield struct.pack(f'>{"BHIQ"[idx]}', value)

def _encode_int(value: int) -> Iterable[bytes]:
    sign = value < 0
    yield from _encode_length(sign, abs(value) - sign)

def _encode_bytes(value: bytes) -> Iterable[bytes]:
    yield from _encode_length(2, len(value))
    yield value

def _encode_string(value: str) -> Iterable[bytes]:
    utf8 = value.encode('utf-8')
    yield from _encode_length(3, len(utf8))
    yield utf8

def _encode_array(value: list) -> Iterable[bytes]:
    yield from _encode_length(4, len(value))
    for item in value:
        yield from _encode_item(item)

def _encode_map(value: dict) -> Iterable[bytes]:
    if not all(isinstance(k, str) for k in value.keys()):
        raise ValueError("All map keys must be strings in DAG-CBOR")
    
    yield from _encode_length(5, len(value))
    # Sort for deterministic encoding
    for key in sorted(value.keys()):
        if not isinstance(key, str):
            raise ValueError("All map keys must be strings in DAG-CBOR")
        yield from _encode_string(key)
        yield from _encode_item(value[key])

def _encode_link(link: CID) -> Iterable[bytes]:
    yield b'\xd8\x2a'  # major 6, minor 42 (tag for CID)
    yield from _encode_bytes(b'\0' + link.buffer) # identity multibase codec

def _encode_float(value: float) -> Iterable[bytes]:
    if not math.isfinite(value):
        raise ValueError("NaN, Infinity, and -Infinity are not allowed in DAG-CBOR")
    
    yield b'\xfb' # major 7, minor 27 (64-bit float always)
    yield struct.pack('>d', value)

@decodec("DAG-CBOR")
def _dagcbor_decode(data) -> IPLData:
    match data:
        case cbor2.CBORTag():
            if data.tag == LINK_TAG:
                return CID(data.value)
            raise ValueError(f'DAG-CBOR forbids all tags except {LINK_TAG} (CID). Got {data.tag}')

def marshal(data: IPLData) -> bytes:
    """Marshal data to DAG-CBOR format."""
    return b''.join(_encode_item(data))

def unmarshal(data: bytes) -> IPLData:
    """Unmarshal data from DAG-CBOR format."""
    return _dagcbor_decode(cbor2.loads(data))