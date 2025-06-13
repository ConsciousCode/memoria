from typing import Literal, Protocol

__all__ = (
    'Encoding', 'get_codec', 'is_encoded', 'encode', 'decode'
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

class IdCodec(Codec):
    code = '\0'

    def encode(self, x: bytes) -> str:
        return x.decode('utf-8')

    def decode(self, x: str) -> bytes:
        return x.encode('utf-8')

class BaseCodec(Codec):
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

_b16 = '0123456789abcdef'
_b10 = _b16[:10]
_abc = 'abcdefghijklmnopqrstuvwxyz'
_ABC = _abc.upper()
_b58 = 'abcdefghijkmnopqrstuvwxyz'
_B58 = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
_x32 = _b10 + _abc[:22]
_X32 = _x32.upper()
_b32 = _abc + _b10[2:8]
_B32 = _b32.upper()
_b64 = _ABC + _abc + _b10

ENCODINGS: dict[str, Codec] = {
    'identity': IdCodec(),
    # Reserved for libp2p peer ids which are base58btc encoded
    '<reserved-1>': ReservedBase('1', '<reserved-1>'),

    'base2': BaseCodec('0', '01'),
    'base8': BaseCodec('7', _b10[:8]),
    'base10': BaseCodec('9', _b10),
    'base16': BaseCodec('f', _b16),
    'base16upper': BaseCodec('F', _b16.upper()),
    'base32hex': BaseCodec('v', _x32),
    'base32hexupper': BaseCodec('V', _X32),
    'base32hexpad': BaseCodec('t', _x32, '=' * 8),
    'base32hexpadupper': BaseCodec('T', _X32, '=' * 8),
    'base32': BaseCodec('b', _b32),
    'base32upper': BaseCodec('B', _B32),
    'base32pad': BaseCodec('c', _b32, '=' * 8),
    'base32padupper': BaseCodec('C', _B32, '=' * 8),
    'base32z': BaseCodec('h', 'ybndrfg8ejkmcpqxot1uwisza345h769'),
    'base36': BaseCodec('k', _b10 + _abc),
    'base36upper': BaseCodec('K', _b10 + _ABC),
    'base45': BaseCodec('R', f"{_b10}{_ABC} $%*+-./:"),
    'base58btc': BaseCodec('z', _b10[1:] + _B58 + _b58),
    'base58flickr': BaseCodec('Z', _b10[1:] + _b58 + _B58),
    'base64': BaseCodec('m', f'{_b64}+/'),
    'base64pad': BaseCodec('M', f'{_b64}+/', '=' * 2),
    'base64url': BaseCodec('u', f'{_b64}-_'),
    'base64urlpad': BaseCodec('U', f'{_b64}-_', '=' * 2),

    # Todo: Requires special logic
    'proquint': ReservedBase('q', 'proquint'),

    # Reserved for CIDv0 which begins with Qm, are encoded in base58btc,
    # and doesn't have a specific multibase code.
    '<reserved-Q>': ReservedBase('Q', '<reserved-Q>'),
    # Reserved to avoid conflict with URIs
    '<reserved-/>': ReservedBase('/', '<reserved-/>'),

    # base256emoji is experimental and has no definitive mapping
}
CODES = {codec.code: codec for codec in ENCODINGS.values()}

type Encoding = Literal[
    'identity', 'base2', 'base8', 'base10', 'base16',
    'base16upper', 'base32hex', 'base32hexupper', 'base32hexpad',
    'base32hexpadupper', 'base32', 'base32upper', 'base32pad',
    'base32padupper', 'base32z', 'base36', 'base36upper', 'base45',
    'base58btc', 'base58flickr', 'base64', 'base64pad', 'base64url',
    'base64urlpad', 'proquint'
]

def get_codec(data: str) -> Codec:
    """Returns the codec used to encode the given data"""
    try: return ENCODINGS[data[0]]
    except (IndexError, KeyError):
        raise ValueError(f'Can not determine encoding for {data}') from None

def is_encoded(data: str) -> bool:
    """Checks if the given data is encoded or not."""
    return bool(data and data[0] in CODES)

def encode(encoding: Encoding, data: bytes) -> str:
    """Encodes the given data using the encoding that is specified."""
    try:
        enc = ENCODINGS[encoding]
        return enc.code + enc.encode(data)
    except KeyError:
        raise ValueError(f'Encoding {encoding} not supported.') from None

def decode(data: str) -> bytes:
    """Decode the multibase-encoded data."""
    return get_codec(data).decode(data[1:])
