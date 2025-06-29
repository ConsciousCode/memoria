import importlib
from typing import Callable, Optional, Protocol, cast
import re
from dataclasses import dataclass

from .exceptions import ProtocolNotFoundError

class Codec(Protocol):
    SIZE: Optional[int]
    IS_PATH: bool

    to_bytes: Callable[[str], bytes]
    to_string: Callable[[bytes], str]

@dataclass
class MAProtocol:
    code: int
    name: str
    codec_name: Optional[str]

    @property
    def codec(self):
        name = self.codec_name
        if name is None:
            name = 'none'
        importlib.import_module
        return cast(Codec,
            importlib.import_module(f".codecs.{name}", __package__)
        )

    @property
    def size(self):
        return self.codec.SIZE

    @property
    def path(self):
        return self.codec.IS_PATH

# https://github.com/multiformats/multicodec/blob/master/table.csv#L382
PROTOCOLS = [
    (0x04, 'ip4', 'ip4'),
    (0x06, 'tcp', 'uint16be'),
    (0x0111, 'udp', 'uint16be'),
    (0x21, 'dccp', 'uint16be'),
    (0x29, 'ip6', 'ip6'),
    (0x2A, 'ip6zone', 'utf8'),
    (0x35, 'dns', 'idna'),
    (0x36, 'dns4', 'idna'),
    (0x37, 'dns6', 'idna'),
    (0x38, 'dnsaddr', 'idna'),
    (0x84, 'sctp', 'uint16be'),
    (0x012D, 'udt', None),
    (0x012E, 'utp', None),
    (0x01A5, 'p2p', 'p2p'),
    (0x01BC, 'onion', 'onion'),
    (0x01BD, 'onion3', 'onion3'),
    (0x01BE, 'garlic64', 'garlic64'),
    (0x01BF, 'garlic32', 'garlic32'),
    (0x01CC, 'quic', None),
    (0x01E0, 'http', None),
    (0x01BB, 'https', None),
    (0x01DD, 'ws', None),
    (0x01DE, 'wss', None),
    (0x01DF, 'p2p-websocket-star', None),
    (0x0113, 'p2p-webrtc-star', None),
    (0x0114, 'p2p-webrtc-direct', None),
    (0x0115, 'p2p-stardust', None),  # deprecated
    (0x0122, 'p2p-circuit', None),
    (0x0190, 'unix', 'fspath')
]
DEPRECATED = {'p2p-stardust'}

PROTONAMES = {}
PROTOCODES = {}
for code, name, codec_name in PROTOCOLS:
    proto = MAProtocol(code, name, codec_name)
    PROTONAMES[name] = proto
    PROTOCODES[code] = proto

def protocol_with_name(name: str):
    if name in DEPRECATED:
        raise ProtocolNotFoundError(name, "name (deprecated)")
    if proto := PROTONAMES.get(name):
        return proto
    raise ProtocolNotFoundError(name, "name")

def protocol_with_code(code: int):
    if proto := PROTOCODES.get(code):
        if proto.name in DEPRECATED:
            raise ProtocolNotFoundError(code, "code (deprecated)")
        return proto
    raise ProtocolNotFoundError(code, "code")

def protocol_with_any(proto: MAProtocol|int|str):
    match proto:
        case MAProtocol(): return proto
        case int(): return protocol_with_code(proto)
        case str(): return protocol_with_name(proto)
        
        case _: raise TypeError(
            f"Protocol object, name or code expected, got {proto!r}"
        )
