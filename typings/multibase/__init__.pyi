from typing import NamedTuple
import baseconv

class Encoding(NamedTuple):
    encoding: str
    code: bytes
    converter: baseconv.BaseConverter

ENCODINGS: list[Encoding]
ENCODINGS_LOOKUP: dict[str, bytes]

def encode(encoding: str, data: bytes) -> bytes: ...
def get_codec(data: str|bytes) -> Encoding: ...
def is_encoded(data: str|bytes) -> bool: ...
def decode(data: str|bytes) -> bytes: ...