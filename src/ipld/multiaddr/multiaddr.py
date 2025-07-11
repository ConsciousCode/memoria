from typing import Iterable, Literal, Optional
import io

from .. import varint
from .._common import Immutable

from .exceptions import StringParseError
from .protocols import MAProtocol, protocol_with_code, protocol_with_name

__all__ = (
    'MultiaddrCodec',
    "Multiaddr", 'join'
)

type MultiaddrCodec = Literal[
    'ip4', 'tcp', 'dccp', 'ip6', 'ip6zone', 'dns', 'dns4', 'dns6',
    'dnsaddr', 'sctp', 'udp', 'p2p-webrtc-star', 'p2p-webrtc-direct',
    'p2p-stardust', 'p2p-circuit', 'udt', 'utp', 'unix', 'p2p',
    'https', 'onion', 'onion3', 'garlic64', 'garlic32',
    'quic', 'ws', 'wss', 'p2p-websocket-star', 'http'
]

def string_iter(string: str) -> Iterable[tuple[MAProtocol, Optional[str]]]:
    if not string.startswith(u'/'):
        raise StringParseError("Must begin with /", string)
    
    string = string.strip('/')
    sp = string.split('/')

    while sp:
        el = sp.pop(0)
        proto = protocol_with_name(el)
        value = None
        if proto.size != 0: # 0 is the "none" codec
            if len(sp) < 1:
                raise StringParseError("Protocol requires address", string, proto.name)
            
            if proto.path:
                value = "/" + "/".join(sp)
                sp.clear()
            else:
                value = sp.pop(0)
        yield proto, value

def bytes_iter(buf: bytes) -> Iterable[tuple[int, MAProtocol, bytes]]:
    stream = io.BytesIO(buf)
    while (offset := stream.tell()) < len(buf):
        code = varint.decode_stream(stream)
        proto = protocol_with_code(code)

        if proto.size is None:
            size = varint.decode_stream(stream)
        else:
            size = proto.size // 8

        data = stream.read(size)
        yield offset, proto, data

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
    _parts: list[tuple[MAProtocol, Optional[str]]]

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
            object.__setattr__(self, "_parts", list(string_iter(addr)))
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
        def from_parts(parts: Iterable[tuple[MAProtocol, Optional[str]]]) -> Iterable[str]:
            for proto, value in parts:
                yield proto.name
                if value is not None:
                    yield value
        
        def from_bytes(buf: bytes) -> Iterable[str]:
            for _, proto, part in bytes_iter(buf):
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
        for idx, (offset, proto, part_value) in enumerate(bytes_iter(self.buffer)):
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

    def parts(self) -> Iterable[tuple[MAProtocol, str]]:
        if parts := getattr(self, "_parts", None):
            yield from parts
        else:
            parts = []
            for _, proto, part in bytes_iter(self.buffer):
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
