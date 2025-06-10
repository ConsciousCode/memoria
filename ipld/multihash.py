from io import BytesIO
from typing import Literal, overload

import base58
import varint

HASH_CODES: dict[str, int] = {
    'id':   0,
    'sha1': 0x11,
    'sha2-256': 0x12,
    'sha2-512': 0x13,
    'sha3-512': 0x14,
    'sha3-384': 0x15,
    'sha3-256': 0x16,
    'sha3-224': 0x17,
    'shake-128': 0x18,
    'shake-256': 0x19,
    'keccak-224': 0x1a,
    'keccak-256': 0x1b,
    'keccak-384': 0x1c,
    'keccak-512': 0x1d,
    'murmur3-128':  0x22,
    'murmur3-32':   0x23,
    'dbl-sha2-256': 0x56,
    **{f"blake2b-{i*8}"  : 0xb200 + i for i in range(1, 0x41)},
    **{f"blake2s-{i*8}"  : 0xb240 + i for i in range(1, 0x21)},
    **{f"skein256-{i*8}" : 0xb300 + i for i in range(1, 0x21)},
    **{f"skein512-{i*8}" : 0xb320 + i for i in range(1, 0x41)},
    **{f"skein1024-{i*8}": 0xb360 + i for i in range(1, 0x81)},
}
CODE_HASHES: dict[int, str] = {
    v: k for k, v in HASH_CODES.items()
}

def _pack_mh(function: int, digest: bytes) -> bytes:
    return varint.encode(function) + varint.encode(len(digest)) + digest

class Multihash:
    __slots__ = ("function", "digest")
    __match_args__ = ("function", "digest")

    @overload
    def __init__(self, function: int|str, digest: bytes): ...
    @overload
    def __init__(self, buffer: bytes): ...
    
    # I fucking despise how Python resolves overloads. *args, **kwargs is
    #  apparently the only way to make this work on the implementation's signature.
    def __init__(self, *args, **kwargs):
        function = digest = buffer = None
        match args:
            case (function, digest): pass
                
            case (buffer_or_function,):
                if digest := kwargs.pop("digest", None):
                    function = buffer_or_function
                else:
                    buffer = buffer_or_function
            case () if (buffer := kwargs.pop("buffer", None)): pass
            case ():
                function = kwargs.pop("function", None)
                digest = kwargs.pop("digest", None)
                if function is None or digest is None:
                    raise TypeError('Multihash() missing function or digest')
            
            case _:
                raise TypeError('Multihash() got too many arguments')        

        if kwargs:
            raise TypeError('Multihash() got too many arguments')
        
        if buffer is not None:
            bio = BytesIO(buffer)
            try:
                function = varint.decode_stream(bio)
            except TypeError:
                raise ValueError('Invalid varint provided')

            if function not in CODE_HASHES:
                raise ValueError(f'Unsupported hash code {function:02x}')

            try:
                length = varint.decode_stream(bio)
            except TypeError:
                raise ValueError('Invalid length provided')

            digest = bio.read()

            if len(digest) != length:
                raise ValueError(f'Inconsistent multihash length {len(digest)} != {length}')
        
        if isinstance(function, str):
            if (function := HASH_CODES.get(function)) is None:
                raise ValueError(f"Unknown hash function: {function}")

        super().__setattr__('function', function)
        super().__setattr__('digest', digest)
    
    @classmethod
    def from_hex(cls, hex_string: str) -> 'Multihash':
        """
        Create a Multihash from a hex encoded string

        :param hex_string: Hex encoded multihash string
        :return: Multihash object
        """
        return cls(bytes.fromhex(hex_string))
    
    @classmethod
    def from_b58(cls, b58_string: str) -> 'Multihash':
        """
        Create a Multihash from a base58 encoded string

        :param b58_string: Base58 encoded multihash string
        :return: Multihash object
        """
        return cls(base58.b58decode(b58_string))

    @staticmethod
    def validate(multihash: bytes) -> bool:
        bio = BytesIO(multihash)
        try:
            if varint.decode_stream(bio) not in CODE_HASHES:
                return False

            if len(bio.read()) != varint.decode_stream(bio):
                return False
        except TypeError:
            return False

        return True

    def __setattr__(self, name, value):
        raise AttributeError("Multihash is immutable")

    def __len__(self):
        return len(self.buffer)
    
    def __hash__(self):
        return hash(self.buffer)
    
    def __iter__(self):
        yield self.function
        yield self.digest
    
    def __str__(self):
        return self.encode()
    
    def __repr__(self):
        return f"Multihash(function={CODE_HASHES[self.function]!r}, digest={self.digest.hex()!r})"

    @property
    def function_name(self) -> str:
        return CODE_HASHES.get(self.function, "<unknown>")

    @property
    def buffer(self):
        """
        Returns the multihash buffer as bytes
        """
        return _pack_mh(self.function, self.digest)

    @property
    def length(self):
        return len(self.digest)
    
    def encode(self, codec: Literal['hex', 'b58'] = 'b58') -> str:
        """
        Encode the multihash to a string using the specified codec

        :param codec: The codec to use for encoding, either 'hex' or 'b58'
        :return: Encoded multihash string
        :rtype: str
        """
        match codec:
            case 'hex': return self.buffer.hex()
            case 'b58': return base58.b58encode(self.buffer).decode()
            case _:
                raise ValueError(f"Unsupported codec: {codec}")
