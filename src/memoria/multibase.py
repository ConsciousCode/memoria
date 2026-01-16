'''
Multibase codecs for encoding and decoding data in various formats.
'''

from abc import ABC, abstractmethod
from typing import Literal, Optional, overload
import math

__all__ = (
    'Base', 'IdBase', 'BitpackBase', 'SimpleBase', 'ReservedBase',
    'Encoding', 'codec', 'is_encoded', 'encode', 'decode',
    'encode_identity', 'decode_identity', 'codec_of',
    'identity', 'base2', 'base8', 'base10', 'base16', 'base16upper',
    'base32hex', 'base32hexupper', 'base32hexpad', 'base32hexpadupper',
    'base32', 'base32upper', 'base32pad', 'base32padupper', 'base32z',
    'base36', 'base36upper', 'base45', 'base58', 'base58btc', 'base58flickr',
    'base64', 'base64pad', 'base64url', 'base64urlpad'
)

class Base(ABC):
    """Abstract base class for multibase codecs."""
    __name__: str
    code: str

    def __call__(self, x: bytes, /) -> str:
        """Encodes the given bytes into a string representation."""
        return self.encode(x)

    @abstractmethod
    def encode(self, x: bytes, /) -> str:
        """Encodes the given bytes into a string representation."""

    @abstractmethod
    def decode(self, x: str, /) -> bytes:
        """Decodes the given string representation back into bytes."""
    
    def __repr__(self):
        return f"multibase.{self.__name__}"

class IdBase:
    """
    Identity codec, must be handled specially because it's the only codec
    which doesn't encode to a string but rather returns the bytes as-is.
    """
    __name__ = 'identity'
    code = '\0'

    @overload
    def encode(self, x: str) -> str: ...
    @overload
    def encode(self, x: bytes) -> bytes: ...

    def encode(self, x): return x

    @overload
    def decode(self, x: str) -> str: ...
    @overload
    def decode(self, x: bytes) -> bytes: ...

    def decode(self, x): return x

    def __repr__(self):
        return "multibase.identity"

class DigitBase(Base):
    digits: str
    padding: str

class BitpackBase(DigitBase):
    """Codec for encoding bytes into a bit-packed string."""
    
    def __init__(self, code: str, bits: int, digits: str, padding: str = ''):
        assert len(code) == 1, 'code must be a single byte'
        self.code = code
        self.bits = bits
        self.digits = digits
        self.group = math.lcm(8, bits) // bits
        self.padding = padding
    
    def encode(self, bs: bytes) -> str:
        """Encodes bytes into a bit-packed string."""
        mask = (1 << self.bits) - 1
        bits = 0
        value = 0
        res = ''
        for byte in bs:
            value = (value << 8) | byte
            bits += 8
            while bits >= self.bits:
                bits -= self.bits
                res += self.digits[(value >> bits) & mask]
        
        # Get the last bits
        if bits > 0:
            res += self.digits[(value << (self.bits - bits)) & mask]
        
        if self.padding:
            return res + self.padding[len(res) % len(self.padding):]
        return res
    
    def decode(self, s: str) -> bytes:
        """Decodes a bit-packed string back into bytes."""
        out = bytearray()
        value = 0
        bits = 0
        for digit in s.rstrip(self.padding[:1]):
            try:
                value = (value << self.bits) | self.digits.index(digit)
                bits += self.bits
                if bits >= 8:
                    bits -= 8
                    out.append(value >> bits)
                    value &= (1 << bits) - 1
            except ValueError:
                raise ValueError(f'invalid digit "{digit}"') from None
        
        return bytes(out)

class SimpleBase(DigitBase):
    """Straightforward base codec for encoding and decoding bytes."""

    def __init__(self, code: str, digits: str, padding: str = ""):
        assert len(code) == 1, 'code must be a single byte'
        self.code = code
        self.digits = digits
        self.padding = padding
    
    def encode(self, bs: bytes) -> str:
        x = int.from_bytes(bs, byteorder='big', signed=False)
        res = ''
        while x > 0:
            x, d = divmod(x, len(self.digits))
            res = self.digits[d] + res
        if self.padding:
            return res + self.padding[len(res) % len(self.padding):]
        return res
    
    def decode(self, s: str) -> bytes:
        x = 0
        for digit in s.rstrip(self.padding[:1]):
            try: x = x * len(self.digits) + self.digits.index(digit)
            except ValueError:
                raise ValueError(f'invalid digit "{digit}"') from None
        
        # Convert the integer back to bytes
        return x.to_bytes((x.bit_length() + 7) // 8, byteorder='big')

class ReservedBase(Base):
    """Base class for reserved multibase codes."""
    def __init__(self, code: str, name: str):
        assert len(code) == 1, 'code must be a single byte'
        self.code = code
        self.name = name
    
    def encode(self, bs: bytes) -> str:
        raise NotImplementedError(f"{self.name} does not support encoding")
    
    def decode(self, s: str) -> bytes:
        raise NotImplementedError(f"{self.name} does not support decoding")

def _upper[T: DigitBase](codec: T) -> T:
    """Returns a new codec with uppercase digits."""
    nc = codec.__new__(type(codec))
    nc.__dict__.update(codec.__dict__)
    nc.code = codec.code.upper()
    nc.digits = codec.digits.upper()
    return nc

def _pad[T: DigitBase](code: str, codec: T, padding: str) -> T:
    """Returns a new codec with specified padding."""
    nc = codec.__new__(type(codec))
    nc.__dict__.update(codec.__dict__)
    nc.code = code
    nc.padding = padding
    return nc

_b16 = '0123456789abcdef'
_b10 = _b16[:10]
_abc = 'abcdefghijklmnopqrstuvwxyz'
_ABC = _abc.upper()
_b58 = 'abcdefghijkmnopqrstuvwxyz'
_B58 = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
_b64 = _ABC + _abc + _b10

identity = IdBase()
base2 = SimpleBase('0', '01')
base8 = SimpleBase('7', _b10[:8])
base10 = SimpleBase('9', _b10)
base16 = BitpackBase('f', 4, _b16)
base16upper = _upper(base16)
base32hex = BitpackBase('v', 4, _b10 + _abc[:22])
base32hexupper = _upper(base32hex)
base32hexpad = _pad('t', base32hex, '='*8)
base32hexpadupper = _upper(base32hexpad)
base32 = BitpackBase('b', 5, _abc + _b10[2:8])
base32upper = _upper(base32)
base32pad = _pad('c', base32, '=' * 8)
base32padupper = _upper(base32pad)
base32z = BitpackBase('h', 5, 'ybndrfg8ejkmcpqxot1uwisza345h769')
base36 = SimpleBase('k', _b10 + _abc)
base36upper = _upper(base36)
base45 = SimpleBase('R', f"{_b10}{_ABC} $%*+-./:")
base58btc = SimpleBase('z', _b10[1:] + _B58 + _b58)
base58flickr = SimpleBase('Z', _b10[1:] + _b58 + _B58)
base64 = BitpackBase('m', 6, f'{_b64}+/')
base64pad = _pad('M', base64, '='*2)
base64url = BitpackBase('u', 6, f'{_b64}-_')
base64urlpad = _pad('U', base64url, '='*2)

# Aliases
base58 = base58btc

ENCODINGS: dict[str, Base] = {
    # Identity is the only multibase that must be bytes, so it's not
    #  compatible with the Codec protocol.
    #'identity': identity,

    # Reserved for libp2p peer ids which are base58btc encoded
    '<reserved-1>': ReservedBase('1', '<reserved-1>'),

    'base2': base2,
    'base8': base8,
    'base10': base10,
    
    'base16': base16,
    'base16upper': base16upper,

    'base32hex': base32hex,
    'base32hexupper': base32hexupper,
    'base32hexpad': base32hexpad,
    'base32hexpadupper': base32hexpadupper,
    'base32': base32,
    'base32upper': base32upper,
    'base32pad': base32pad,
    'base32padupper': base32padupper,
    'base32z': base32z,
    
    'base36': base36,
    'base36upper': base36upper,
    
    'base45': base45,
    
    'base58btc': base58btc,
    'base58flickr': base58flickr,
    
    'base64': base64,
    'base64pad': base64pad,
    'base64url': base64url,
    'base64urlpad': base64urlpad,

    # Todo: Requires special logic
    #'proquint': ReservedBase('q', 'proquint'),

    # Reserved for CIDv0 which begins with Qm, are encoded in base58btc,
    # and doesn't have a specific multibase code.
    '<reserved-Q>': ReservedBase('Q', '<reserved-Q>'),
    # Reserved to avoid conflict with URIs
    '<reserved-/>': ReservedBase('/', '<reserved-/>'),

    # base256emoji is experimental and has no definitive mapping
}
CODES: dict[str, Base] = {}
for _n, _c in ENCODINGS.items():
    CODES[_c.code] = _c
    _c.__name__ = _n
    del _n, _c

type Encoding = Literal[
    'base2', 'base8', 'base10', 'base16', 'base16upper',
    'base32hex', 'base32hexupper', 'base32hexpad', 'base32hexpadupper',
    'base32', 'base32upper', 'base32pad', 'base32padupper', 'base32z',
    'base36', 'base36upper', 'base45', 'base58btc', 'base58flickr',
    'base64', 'base64pad', 'base64url', 'base64urlpad'
]
'''Valid multibase encodings.'''

@overload
def codec(name: Literal['identity']) -> IdBase: ...
@overload
def codec(name: Encoding) -> Base: ...

def codec(name: str) -> Base|IdBase:
    """Returns the codec used to encode the given data"""
    if name == 'identity':
        return identity
    if enc := ENCODINGS.get(name):
        return enc
    raise ValueError(f'Cannot determine encoding for {name}')

def codec_of(data: str) -> Base|IdBase:
    """Returns the codec used to encode the given data"""
    try:
        c = data[0]
        return identity if c == '\0' else CODES[c]
    except (IndexError, KeyError):
        raise ValueError(f'Cannot determine encoding for {data}') from None

def is_encoded(data: str) -> bool:
    """Checks if the given data is encoded or not."""
    if not data: return False
    if base := CODES.get(data[0]):
        return not isinstance(base, ReservedBase)
    return False

def encode_identity(data: bytes) -> bytes:
    """Encodes data using the identity encoding."""
    return b'\0' + data

def decode_identity(data: bytes) -> Optional[bytes]:
    """Decodes data that was encoded using the identity encoding."""
    if data[0] != 0:
        raise ValueError('Data is not encoded with identity encoding.')
    return data[1:] if data else None

def encode(encoding: Encoding|Base, data: bytes) -> str:
    """Encodes the given data using the encoding that is specified."""
    if isinstance(encoding, str):
        enc = ENCODINGS.get(encoding)
    else:
        enc = encoding
    
    if enc: return enc.code + enc.encode(data)
    raise ValueError(f'Encoding {encoding} not supported.')

def decode(data: str) -> bytes:
    """Decode the multibase-encoded data."""
    codec = codec_of(data)
    if isinstance(codec, IdBase):
        return identity.decode(data[1:]).encode('utf-8')
    return codec.decode(data[1:])
