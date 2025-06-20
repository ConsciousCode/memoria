from typing import Literal, Optional, Protocol

__all__ = (
    'Codec', 'IdCodec', 'BaseCodec', 'ReservedBase',
    'Encoding', 'codec', 'is_encoded', 'encode', 'decode',
    'encode_identity', 'decode_identity', 'codec_of',
    'identity', 'base2', 'base8', 'base10', 'base16',
    'base16upper', 'base32hex', 'base32hexupper', 'base32hexpad',
    'base32hexpadupper', 'base32', 'base32upper', 'base32pad',
    'base32padupper', 'base32z', 'base36', 'base36upper', 'base45',
    'base58', 'base58btc', 'base58flickr', 'base64', 'base64pad',
    'base64url', 'base64urlpad'
)

class Codec(Protocol):
    code: str

    def encode(self, x: bytes, /) -> str:
        """
        Encodes the given bytes into a string representation.
        """
        ...

    def decode(self, x: str, /) -> bytes:
        """
        Decodes the given string representation back into bytes.
        """
        ...

class IdCodec:
    code = '\0'

    def encode(self, x: bytes) -> bytes: return x
    def decode(self, x: bytes) -> bytes: return x

class BaseCodec(Codec):
    def __init__(self, code: str, digits: str, padding: str = ""):
        assert len(code) == 1, 'code must be a single byte'
        self.code = code
        self.digits = digits
        self.padding = padding
    
    def _upper(self) -> 'BaseCodec':
        """Returns a new codec with uppercase digits."""
        return BaseCodec(self.code, self.digits.upper(), self.padding)
    
    def _pad(self, padding: str) -> 'BaseCodec':
        """Returns a new codec with specified padding."""
        return BaseCodec(self.code, self.digits, padding)
    
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
        for digit in s:
            try: x = x * len(self.digits) + self.digits.index(digit)
            except ValueError:
                raise ValueError(f'invalid digit "{digit}"')
        
        # Convert the integer back to bytes
        return x.to_bytes((x.bit_length() + 7) // 8, byteorder='big')

class ReservedBase:
    def __init__(self, code: str, name: str):
        assert len(code) == 1, 'code must be a single byte'
        self.code = code
        self.name = name
    
    def encode(self, bs: bytes) -> str:
        raise NotImplementedError(f"{self.name} does not support encoding")
    
    def decode(self, s: str) -> bytes:
        raise NotImplementedError(f"{self.name} does not support decoding")

def _upper(codec: BaseCodec) -> BaseCodec:
    """Returns a new codec with uppercase digits."""
    return BaseCodec(codec.code.upper(), codec.digits.upper(), codec.padding)

def _pad(code: str, codec: BaseCodec, padding: str) -> BaseCodec:
    """Returns a new codec with specified padding."""
    return BaseCodec(code, codec.digits, padding)

_b16 = '0123456789abcdef'
_b10 = _b16[:10]
_abc = 'abcdefghijklmnopqrstuvwxyz'
_ABC = _abc.upper()
_b58 = 'abcdefghijkmnopqrstuvwxyz'
_B58 = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
_x32 = _b10 + _abc[:22]
_b32 = _abc + _b10[2:8]
_b64 = _ABC + _abc + _b10

identity = IdCodec()
base2 = BaseCodec('0', '01')
base8 = BaseCodec('7', _b10[:8])
base10 = BaseCodec('9', _b10)
base16 = BaseCodec('f', _b16)
base16upper = _upper(base16)
base32hex = BaseCodec('v', _x32)
base32hexupper = _upper(base32hex)
base32hexpad = _pad('t', base32hex, '='*8)
base32hexpadupper = _upper(base32hexpad)
base32 = BaseCodec('b', _b32)
base32upper = _upper(base32)
base32pad = BaseCodec('c', _b32, '=' * 8)
base32padupper = _upper(base32pad)
base32z = BaseCodec('h', 'ybndrfg8ejkmcpqxot1uwisza345h769')
base36 = BaseCodec('k', _b10 + _abc)
base36upper = _upper(base36)
base45 = BaseCodec('R', f"{_b10}{_ABC} $%*+-./:")
base58btc = BaseCodec('z', _b10[1:] + _B58 + _b58)
base58flickr = BaseCodec('Z', _b10[1:] + _b58 + _B58)
base64 = BaseCodec('m', f'{_b64}+/')
base64pad = _pad('M', base64, '='*2)
base64url = BaseCodec('u', f'{_b64}-_')
base64urlpad = _pad('U', base64url, '='*2)

# Aliases
base58 = base58btc

ENCODINGS: dict[str, Codec] = {
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
CODES = {codec.code: codec for codec in ENCODINGS.values()}

type Encoding = Literal[
    'base2', 'base8', 'base10', 'base16',
    'base16upper', 'base32hex', 'base32hexupper', 'base32hexpad',
    'base32hexpadupper', 'base32', 'base32upper', 'base32pad',
    'base32padupper', 'base32z', 'base36', 'base36upper', 'base45',
    'base58btc', 'base58flickr', 'base64', 'base64pad', 'base64url',
    'base64urlpad'
]

def codec(name: Encoding) -> Codec:
    """Returns the codec used to encode the given data"""
    if enc := ENCODINGS.get(name):
        return enc
    raise ValueError(f'Cannot determine encoding for {name}')

def codec_of(data: str) -> Codec:
    """Returns the codec used to encode the given data"""
    try: return CODES[data[0]]
    except (IndexError, KeyError):
        raise ValueError(f'Cannot determine encoding for {data}') from None

def is_encoded(data: str) -> bool:
    """Checks if the given data is encoded or not."""
    return bool(data and data[0] in CODES)

def encode_identity(data: bytes) -> bytes:
    """Encodes data using the identity encoding."""
    return b'\x00' + data

def decode_identity(data: bytes) -> Optional[bytes]:
    """Decodes data that was encoded using the identity encoding."""
    return data[1:] if data else None

def encode(encoding: Encoding|Codec, data: bytes) -> str:
    """Encodes the given data using the encoding that is specified."""
    if isinstance(encoding, str):
        enc = ENCODINGS.get(encoding)
    else:
        enc = encoding
    
    if enc: return enc.code + enc.encode(data)
    raise ValueError(f'Encoding {encoding} not supported.')

def decode(data: str) -> bytes:
    """Decode the multibase-encoded data."""
    return codec_of(data).decode(data[1:])
