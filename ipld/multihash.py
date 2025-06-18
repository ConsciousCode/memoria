from io import BytesIO
from typing import Literal, Self, cast, overload
import hashlib

from . import varint, multibase

__all__ = ('Multihash', 'multihash')

HASH_CODES: dict[str, int] = {
    'id': 0,
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
    'blake3': 0x1e,
    'sha2-384': 0x20,
    'dbl-sha2-256': 0x56, # draft
    'md4': 0xd4, # draft but Python supports it
    'md5': 0xd5, # draft
    #'fr32-sha256-trunc254-padbintree': 0x1011, # draft
    'sha2-256-trunc254-padded': 0x1012, # zeros 2 most significant bits
    'sha2-224': 0x1013,
    'sha2-512-224': 0x1014,
    'sha2-512-256': 0x1015,
    #'murmur3-x64-128': 0x1022, # draft, also "hash" instead of "multihash"?
    #'ripemd-128': 0x1052, # draft
    #'ripemd-160': 0x1053, # draft
    #'ripemd-256': 0x1054, # draft
    #'ripemd-320': 0x1055, # draft
    #'x11': 0x1100, # draft
    #'kangarootwelve': 0x1d01, # draft
    #'sm3-256': 0x534d, # draft
    'murmur3-128': 0x22,
    'murmur3-32': 0x23,
    **{f"blake2b-{i*8}"  : 0xb200 + i for i in range(1, 0x41)},
    **{f"blake2s-{i*8}"  : 0xb240 + i for i in range(1, 0x21)},
    **{f"skein256-{i*8}" : 0xb300 + i for i in range(1, 0x21)},
    **{f"skein512-{i*8}" : 0xb320 + i for i in range(1, 0x41)},
    **{f"skein1024-{i*8}": 0xb360 + i for i in range(1, 0x81)},
    'poseidon-bls12_381-a2-fc1': 0xb401,
    #'poseidon-bls12_381-a2-fc1-sc': 0xb402, # draft
    #'ssz-sha2-256-bmt': 0xb502, # draft
    #'sha2-256-chunked': 0xb510, # draft
    #'bcrypt-pbkdf': 0xd00d, # draft
}
CODE_HASHES: dict[int, str] = {
    v: k for k, v in HASH_CODES.items()
}

def _pack_mh(function: int, digest: bytes) -> bytes:
    return varint.encode(function) + varint.encode(len(digest)) + digest

class Multihash:
    '''A hash with a function code and digest.'''
    __slots__ = ("function", "digest")
    __match_args__ = ("function", "digest")

    function: int
    digest: bytes

    @overload
    def __init__(self, function: int|str, digest: str|bytes): ...
    @overload
    def __init__(self, buffer: bytes): ...
    
    def __init__(self, *args, **kwargs):
        # Types of function, digest, and buffer are interlinked, we need to
        # combine them into a single tuple to match against (parts).
        match args:
            case (int(function)|str(function), bytes(digest)):
                parts = (function, digest, None)

            case (buffer,):
                if digest := kwargs.pop("digest", None):
                    parts = (cast(int|str, buffer), cast(bytes, digest), None)
                else:
                    parts = (None, None, cast(bytes, buffer))
            case ():
                if (buffer := kwargs.pop("buffer", None)):
                    parts = (None, None, cast(bytes, buffer))
                else:
                    function = kwargs.pop("function", None)
                    digest = kwargs.pop("digest", None)
                    if function is None or digest is None:
                        raise TypeError('Multihash() missing function or digest')
                    parts = (cast(int|str, function), cast(bytes, digest), None)
            
            case _:
                raise TypeError('Multihash() got too many arguments')        

        if kwargs:
            raise TypeError('Multihash() got too many arguments')

        if parts[2] is not None:
            bio = BytesIO(parts[2])
            try: function = varint.decode_stream(bio)
            except TypeError:
                raise ValueError('Invalid varint provided') from None

            if function not in CODE_HASHES:
                raise ValueError(f'Unsupported hash code {function:02x}')

            try: length = varint.decode_stream(bio)
            except TypeError:
                raise ValueError('Invalid length provided') from None

            if len(digest := bio.read()) != length:
                raise ValueError(f'Inconsistent multihash length {len(digest)} != {length}')
        else:
            function, digest, _ = parts
            if isinstance(function, str):
                if (function := HASH_CODES.get(function)) is None:
                    raise ValueError(f"Unknown hash function: {function}")
        
        if isinstance(digest, str):
            digest = bytes.fromhex(digest)

        if not isinstance(function, int):
            raise TypeError(f"Expected function to be int, got {type(function).__name__}")
        
        if not isinstance(digest, bytes):
            raise TypeError(f"Expected digest to be bytes, got {type(digest).__name__}")

        super().__setattr__('function', function)
        super().__setattr__('digest', digest)
    
    @classmethod
    def from_hex(cls, hex_string: str) -> Self:
        """
        Create a Multihash from a hex encoded string

        :param hex_string: Hex encoded multihash string
        :return: Multihash object
        """
        return cls(bytes.fromhex(hex_string))
    
    @classmethod
    def from_b58(cls, b58_string: str) -> Self:
        """
        Create a Multihash from a base58 encoded string

        :param b58_string: Base58 encoded multihash string
        :return: Multihash object
        """
        return cls(multibase.base58.decode(b58_string))

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
        return f"Multihash({CODE_HASHES[self.function]!r}, digest={self.digest.hex()!r})"

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
            case 'b58': return multibase.base58.encode(self.buffer)
            case _:
                raise ValueError(f"Unsupported codec: {codec}")

class multihash:
    """A class to handle multihashing with a specific hash function."""

    def __init__(self, name: str):
        if name not in HASH_CODES:
            raise ValueError(f"Unknown hash function: {name}")
        self.name = name
        self.hash = hashlib.new(name)

    def update(self, data: bytes):
        """Update the hash with the given data."""
        self.hash.update(data)
        return self
    
    def digest(self) -> Multihash:
        """Return the digest as a Multihash."""
        return Multihash(self.name, self.hash.digest())
