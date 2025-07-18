'''
Multiaddr is a cross-protocol, cross-platform format for representing
internet addresses. It emphasizes explicitness and self-description.
Learn more here: https://multiformats.io/multiaddr/
'''

from typing import Iterable, Literal, Optional
import io
import os
import base64

import netaddr
import idna as m_idna

from . import varint, multibase
from ._common import Immutable

__all__ = (
    'ParseError', 'StringParseError', 'BinaryParseError',
    'ProtocolNotFoundError', 'Multiaddr', 'MultiaddrCodec'
)

################
## Exceptions ##
################

class ParseError(ValueError):
    pass

class StringParseError(ParseError):
    """MultiAddr string representation could not be parsed."""

    def __init__(self, message: str, string: str, protocol: Optional[str]=None, original: Optional[str]=None):
        if protocol:
            msg = f"Invalid MultiAddr {string!r} protocol {protocol}: {message}"
        else:
            msg = f"Invalid MultiAddr {string!r}: {message}"

        super().__init__(msg)
        self.message = message
        self.string = string
        self.protocol = protocol
        self.original = original

class BinaryParseError(ParseError):
    """MultiAddr binary representation could not be parsed."""

    def __init__(self, message: str, binary: bytes, protocol: str, original: Optional[str]=None):
        super().__init__(
            f"Invalid binary MultiAddr protocol {protocol}: {message}"
        )
        self.message = message
        self.binary = binary
        self.protocol = protocol
        self.original = original

class ProtocolNotFoundError(RuntimeError):
    """No protocol with the given name or code found."""
    def __init__(self, value: int|str, kind: str="name"):
        super().__init__(
            f"No protocol with {kind} {value!r} found"
        )
        self.value = value
        self.kind = kind

####################
## Codec handling ##
####################

type MultiaddrCodec = Literal[
    'ip4', 'tcp', 'dccp', 'ip6', 'ip6zone', 'dns', 'dns4', 'dns6',
    'dnsaddr', 'sctp', 'udp', 'p2p-webrtc-star', 'p2p-webrtc-direct',
    'p2p-stardust', 'p2p-circuit', 'udt', 'utp', 'unix', 'p2p',
    'https', 'onion', 'onion3', 'garlic64', 'garlic32',
    'quic', 'ws', 'wss', 'p2p-websocket-star', 'http'
]

class AddrCodec(type):
    '''Metaclass for multiaddr codecs.'''
    size: Optional[int] = None
    path: bool = False

    def to_bytes(cls, string: str):
        raise NotImplementedError(f"{cls.__name__}.to_bytes")

    def to_string(cls, buf: bytes):
        raise NotImplementedError(f"{cls.__name__}.to_string")
    
    def __repr__(cls):
        return f"{cls.__name__}({cls.size, cls.path})"

class MACodec(metaclass=AddrCodec):
    pass

class fspath(MACodec):
    '''Filesystem path codec.'''
    path = True

    @staticmethod
    def to_bytes(string: str):
        return os.fsencode(string)

    @staticmethod
    def to_string(buf: bytes):
        return os.fsdecode(buf)

class garlic32(MACodec):
    '''I2P garlic32 codec.'''
    @staticmethod
    def validate(buf: bytes) -> bytes:
        bc = len(buf)
        if bc < 35 and bc != 32:
            raise ValueError(f"Failed to validate garlic_addr: {buf} is not an i2p base32 address. Length: {len(buf)}")
        return buf

    @staticmethod
    def to_bytes(string: str) -> bytes:
        string += '=' * (8 - len(string) % 8)  # Pad with '=' to multiple of 8
        
        try: garlic_host = base64.b32decode(string, casefold=True)
        except Exception as e:
            raise ValueError(f"Cannot decode {string!r} as base32: {e}") from e
        
        return garlic32.validate(garlic_host)

    @staticmethod
    def to_string(buf: bytes) -> str:
        return base64.b32encode(garlic32.validate(buf)).decode('ascii').rstrip('=')

class garlic64(MACodec):
    '''I2P garlic64 codec.'''
    @staticmethod
    def validate(buf: bytes) -> bytes:
        if len(buf) < 386:
            raise ValueError(f"Failed to validate garlic_addr: {buf} is not an i2p base64 address. Length: {len(buf)}")
        return buf

    @staticmethod
    def to_bytes(string: str) -> bytes:
        try: garlic_host = base64.b64decode(string)
        except Exception as e:
            raise ValueError(f"Cannot decode {string!r} as base64: {e}") from e

        return garlic64.validate(garlic_host)

    @staticmethod
    def to_string(buf: bytes) -> str:
        return base64.b64encode(garlic64.validate(buf)).decode('ascii').rstrip('=')

class idna(MACodec):
    '''Internationalized Domain Names in Applications codec for DNS.'''

    @staticmethod
    def to_bytes(string: str):
        return m_idna.encode(string, uts46=True)

    @staticmethod
    def to_string(buf: bytes):
        return m_idna.decode(buf)

class ip4(MACodec):
    '''IPv4 address codec.'''
    size = 32
    
    @staticmethod
    def to_bytes(string: str):
        return netaddr.IPAddress(string, version=4).packed

    @staticmethod
    def to_string(buf: bytes):
        return str(netaddr.IPAddress(
            int.from_bytes(buf, byteorder='big'), version=4
        ))

class ip6(MACodec):
    '''IPv6 address codec.'''
    size = 128
    path = False

    @staticmethod
    def to_bytes(string: str):
        return netaddr.IPAddress(string, version=6).packed

    @staticmethod
    def to_string(buf: bytes):
        return str(netaddr.IPAddress(
            int.from_bytes(buf, byteorder='big'), version=6
        ))

class none(MACodec):
    '''No-op codec for protocols without specific encoding.'''
    size = 0

    @staticmethod
    def to_bytes(string: str) -> bytes:
        return bytes(string, 'utf-8')

    @staticmethod
    def to_string(buf: bytes) -> str:
        return buf.decode('utf-8')

class onion(MACodec):
    size = 96
    path = False

    @staticmethod
    def to_bytes(string: str):
        addr = string.split(":")
        if len(addr) != 2:
            raise ValueError("Does not contain a port number")

        # onion address without the ".onion" substring
        if len(addr[0]) != 16:
            raise ValueError("Invalid onion host address length (must be 16 characters)")
        
        try: onion_host = base64.b32decode(addr[0].upper())
        except Exception as e:
            raise ValueError(f"Cannot decode {addr[0]!r} as base32: {1}") from e

        # onion port number
        try: port = int(addr[1], 10)
        except ValueError as e:
            raise ValueError("Port number is not a base 10 integer") from e
        
        if port not in range(1, 65536):
            raise ValueError("Port number is not in range(1, 65536)")

        return onion_host + port.to_bytes(byteorder='big')

    @staticmethod
    def to_string(buf: bytes):
        addr = base64.b32encode(buf[:-2]).decode('ascii').lower()
        port = int.from_bytes(buf[-2:], byteorder='big')
        return f"{addr}:{port}"

class onion3(MACodec):
    size = 296
    path = False

    @staticmethod
    def to_bytes(string: str):
        addr = string.split(":")
        if len(addr) != 2:
            raise ValueError("Does not contain a port number")

        # onion3 address without the ".onion" substring
        if len(addr[0]) != 56:
            raise ValueError("Invalid onion3 host address length (must be 56 characters)")
        
        try: onion3_host = base64.b32decode(addr[0].upper())
        except Exception as exc:
            raise ValueError(f"Cannot decode {addr[0]!r} as base32") from exc

        # onion3 port number
        try: port = int(addr[1], 10)
        except ValueError as exc:
            raise ValueError("Port number is not a base 10 integer") from exc
        
        if port not in range(1, 65536):
            raise ValueError("Port number is not in range(1, 65536)")

        return onion3_host + port.to_bytes(byteorder='big')

    @staticmethod
    def to_string(buf: bytes):
        addr = base64.b32encode(buf[:-2]).decode('ascii').lower()
        port = int.from_bytes(buf[-2:], byteorder='big')
        return f"{addr}:{port}"

class p2p(MACodec):
    '''P2P MultiHash codec.'''

    @staticmethod
    def to_bytes(string: str):
        # the address is a base58-encoded string
        mm = multibase.base58.decode(string)
        if len(mm) < 5:
            raise ValueError("P2P MultiHash too short: len() < 5")
        return mm

    @staticmethod
    def to_string(buf: bytes):
        return multibase.base58.encode(buf)

class uint16be(MACodec):
    '''16-bit unsigned integer codec in big-endian format.'''
    size = 16

    @staticmethod
    def to_bytes(string: str):
        return int(string, 10).to_bytes(2, byteorder='big')

    @staticmethod
    def to_string(buf: bytes):
        if len(buf) != 2:
            raise ValueError("Invalid integer length (must be 2 bytes)")
        return str(int.from_bytes(buf))

class utf8(MACodec):
    @staticmethod
    def to_bytes(string: str):
        if len(string) == 0:
            raise ValueError("value must not be empty")
        return string.encode('utf-8')

    @staticmethod
    def to_string(buf: bytes):
        if len(buf) == 0:
            raise ValueError("invalid length (should be > 0)")
        return buf.decode('utf-8')

class MultiaddrProtocol:
    '''A protocol specification for Multiaddr.'''
    code: int
    name: str
    codec: type[MACodec] = none

    @property
    def size(self):
        return self.codec.size

    @property
    def path(self):
        return self.codec.path

# https://github.com/multiformats/multicodec/blob/master/table.csv#L382
PROTOCOLS = [
    (0x04, 'ip4', ip4),
    (0x06, 'tcp', uint16be),
    (0x0111, 'udp', uint16be),
    (0x21, 'dccp', uint16be),
    (0x29, 'ip6', ip6),
    (0x2A, 'ip6zone', utf8),
    (0x35, 'dns', idna),
    (0x36, 'dns4', idna),
    (0x37, 'dns6', idna),
    (0x38, 'dnsaddr', idna),
    (0x84, 'sctp', uint16be),
    (0x012D, 'udt'),
    (0x012E, 'utp'),
    (0x01A5, 'p2p', p2p),
    (0x01BC, 'onion', onion),
    (0x01BD, 'onion3', onion3),
    (0x01BE, 'garlic64', garlic64),
    (0x01BF, 'garlic32', garlic32),
    (0x01CC, 'quic'),
    (0x01E0, 'http'),
    (0x01BB, 'https'),
    (0x01DD, 'ws'),
    (0x01DE, 'wss'),
    (0x01DF, 'p2p-websocket-star'),
    (0x0113, 'p2p-webrtc-star'),
    (0x0114, 'p2p-webrtc-direct'),
    (0x0115, 'p2p-stardust'),  # deprecated
    (0x0122, 'p2p-circuit'),
    (0x0190, 'unix', fspath)
]
DEPRECATED = {'p2p-stardust'}

PROTONAMES: dict[str, MultiaddrProtocol] = {}
PROTOCODES: dict[int, MultiaddrProtocol] = {}
for spec in PROTOCOLS:
    proto = MultiaddrProtocol(*spec)
    PROTONAMES[proto.name] = proto
    PROTOCODES[proto.code] = proto

def _proto_name(name: str):
    if name in DEPRECATED:
        raise ProtocolNotFoundError(name, "name (deprecated)")
    if proto := PROTONAMES.get(name):
        return proto
    raise ProtocolNotFoundError(name, "name")

def _proto_code(code: int):
    if proto := PROTOCODES.get(code):
        if proto.name in DEPRECATED:
            raise ProtocolNotFoundError(code, "code (deprecated)")
        return proto
    raise ProtocolNotFoundError(code, "code")

#############
## Parsing ##
#############

def _string_iter(string: str) -> Iterable[tuple[MultiaddrProtocol, Optional[str]]]:
    if not string.startswith('/'):
        raise StringParseError("Must begin with /", string)
    
    sp = string.strip('/').split('/')
    while sp:
        proto = _proto_name(sp.pop(0))
        value = None
        if proto.size != 0: # 0 is the "none" codec
            if not sp:
                raise StringParseError(
                    "Protocol requires address", string, proto.name
                )
            
            if proto.path:
                # The rest is a path
                value = "/" + "/".join(sp)
                sp.clear()
            else:
                value = sp.pop(0)
        yield proto, value

def _bytes_iter(buf: bytes) -> Iterable[tuple[int, MultiaddrProtocol, bytes]]:
    """
    Iterate over the binary Multiaddr buffer.
    Yields tuples of (offset, protocol, data).
    """
    stream = io.BytesIO(buf)
    while (offset := stream.tell()) < len(buf):
        proto = _proto_code(varint.decode_stream(stream))

        if (ps := proto.size) is not None:
            size = ps // 8
        else:
            size = varint.decode_stream(stream)

        yield offset, proto, stream.read(size)

###############
## Multiaddr ##
###############

class Multiaddr(Immutable):
    """
    Multiaddr is a representation of multiple nested internet addresses.

    Multiaddr is a cross-protocol, cross-platform format for representing
    internet addresses. It emphasizes explicitness and self-description.

    Learn more here: https://multiformats.io/multiaddr/

    Multiaddrs have both a binary and string representation.

        >>> from multiaddr import Multiaddr
        >>> addr = Multiaddr("/ip4/1.2.3.4/tcp/80")

    Multiaddr objects are immutable, so `encapsulate` and `decapsulate`
    return new objects rather than modify internal state.
    """

    __slots__ = ("_buffer", "_parts")
    _buffer: bytes
    _parts: list[tuple[MultiaddrProtocol, Optional[str]]]

    def __new__(cls, addr: 'str|bytes|Multiaddr'):
        """Create a new Multiaddr instance."""
        if isinstance(addr, cls):
            return addr
        return super().__new__(cls)

    def __init__(self, addr: 'str|bytes|Multiaddr'):
        """
        Instantiate a new Multiaddr.

        Args:
            addr : A string-encoded or a byte-encoded Multiaddr
        """
        if isinstance(addr, Multiaddr):
            return # Copy constructor
        
        super().__init__()
        if isinstance(addr, str):
            object.__setattr__(self, "_parts", list(_string_iter(addr)))
        elif isinstance(addr, bytes):
            if not addr:
                raise ValueError("Multiaddr cannot be empty")
            object.__setattr__(self, "_buffer", addr)
        else:
            raise TypeError(
                "MultiAddr must be bytes, str or another MultiAddr instance"
            )

    @classmethod
    def join(cls, *addrs: 'str|bytes|Multiaddr'):
        """
        Concatenate the values of the given MultiAddr strings or objects,
        encapsulating each successive MultiAddr value with the previous ones.
        """
        return cls(b"".join(cls(a).buffer for a in addrs))

    def __eq__(self, other):
        """Checks if two Multiaddr objects are exactly equal."""
        return self.buffer == other.buffer

    def __str__(self):
        """
        Return the string representation of this Multiaddr.

        May raise a :class:`~multiaddr.exceptions.BinaryParseError` if the
        stored MultiAddr binary representation is invalid.
        """
        def from_parts(parts: Iterable[tuple[MultiaddrProtocol, Optional[str]]]) -> Iterable[str]:
            for proto, value in parts:
                yield proto.name
                if value is not None:
                    yield value
        
        def from_bytes(buf: bytes) -> Iterable[str]:
            for _, proto, part in _bytes_iter(buf):
                yield proto.name
                if proto.size != 0:
                    value = proto.codec.to_string(part)
                    if proto.path and value[0] == '/':
                        yield value[1:]
                    else:
                        yield value
        
        if parts := getattr(self, "_parts", None):
            items = from_parts(parts)
        else:
            items = from_bytes(self.buffer)
        
        return '/' + '/'.join(items)

    def __repr__(self):
        return f"Multiaddr({str(self)!r})"

    def __hash__(self):
        return hash(self.buffer)

    def split(self, maxsplit=-1) -> Iterable['Multiaddr']:
        """
        Returns the list of individual path components this MultiAddr is made
        up of.
        """
        final_split_offset = -1
        for idx, (offset, proto, part_value) in enumerate(_bytes_iter(self.buffer)):
            # Split at most `maxplit` times
            if idx == maxsplit:
                final_split_offset = offset
                break

            # Re-assemble binary MultiAddr representation
            if proto.size is None:
                part_size = varint.encode(len(part_value))
            else:
                part_size = b''

            # Add MultiAddr with the given value
            yield self.__class__(
                varint.encode(proto.code) + part_size + part_value
            )
        # Add final item with remainder of MultiAddr if there is anything left
        if final_split_offset >= 0:
            yield self.__class__(self.buffer[final_split_offset:])
    
    @property
    def buffer(self) -> bytes:
        if hasattr(self, '_buffer'):
            return self._buffer
        
        def build_bytes():
            for proto, value in self._parts:
                yield varint.encode(proto.code)
                if value is not None:
                    buf = proto.codec.to_bytes(value)
                    if proto.size is None:
                        yield varint.encode(len(buf))
                    yield buf
        
        buf = b''.join(build_bytes())
        object.__setattr__(self, "_buffer", buf)
        return buf

    def parts(self) -> Iterable[tuple[MultiaddrProtocol, str]]:
        if parts := getattr(self, "_parts", None):
            yield from parts
        else:
            parts = []
            for _, proto, part in _bytes_iter(self.buffer):
                value = proto.codec.to_string(part)
                item = proto, value
                yield item
                parts.append(item)
            
            # Set at the end to avoid edge cases
            object.__setattr__(self, "_parts", parts)

    def encapsulate(self, other: 'str|bytes|Multiaddr'):
        """
        Wrap this Multiaddr around another.

        For example:
            /ip4/1.2.3.4 encapsulate /tcp/80 = /ip4/1.2.3.4/tcp/80
        """
        return self.join(self, other)
    
    def decapsulate(self, other: 'str|bytes|Multiaddr'):
        """
        Remove a Multiaddr wrapping.

        For example:
            /ip4/1.2.3.4/tcp/80 decapsulate /ip4/1.2.3.4 = /tcp/80
        """
        s1 = self.buffer
        s2 = Multiaddr(other).buffer
        try: idx = s1.rindex(s2)
        except ValueError:
            return self
        return Multiaddr(s1[:idx])

def join(*addrs: 'str|bytes|Multiaddr') -> Multiaddr:
    """
    Join multiple Multiaddr strings or objects into a single Multiaddr.

    Args:
        addrs: Multiaddr strings or objects to join.
    
    Returns:
        A new Multiaddr object that is the concatenation of all input addrs.
    """
    return Multiaddr.join(*addrs)
