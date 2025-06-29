from typing import Any, Literal, Optional, Self, cast, overload, override

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from ._common import Immutable

from .multihash import BaseMultihash, Multihash, multihash
from . import multibase, multicodec

__all__ = (
    'Version', 'Codec',
    'CID', 'CIDv0', 'CIDv1', 'AnyCID'
)

type Version = Literal[0, 1]
type Codec = Literal[
    'raw', 'dag-pb', 'dag-cbor', 'dag-json',
    'libp2p-key', 'git-raw',
    'torrent-info', 'torrent-file',
    'leofcoin-block', 'leofcoin-tx', 'leofcoin-pr',
    'eth-block', 'eth-block-list',
    'eth-tx-trie', 'eth-tx', 'eth-tx-receipt-trie', 'eth-tx-receipt',
    'eth-state-trie', 'eth-account-snapshot', 'eth-storage-trie',
    'bitcoin-block', 'bitcoin-tx',
    'zcash-block', 'zcash-tx',
    'stellar-block', 'stellar-tx',
    'decred-block', 'decred-tx',
    'dash-block', 'dash-tx',
    'swarm-manifest', 'swarm-feed'
]
type AnyCID = 'CID|CIDv0|CIDv1'

class CID(Immutable):
    buffer: bytes

    __slots__ = ("buffer",)
    __match_args__ = ("version", "codec", "multihash")

    @overload
    def __new__[T: CID](cls, cid: T, /) -> T: ...
    @overload
    def __new__(cls, data: str|bytes, /) -> AnyCID: ...
    @overload
    def __new__(cls, /, version: Literal[0], codec: Literal['dag-pb'], multihash: str|bytes|BaseMultihash) -> 'CIDv0': ...
    @overload
    def __new__(cls, /, version: Literal[0], multihash: str|bytes|BaseMultihash) -> 'CIDv0': ...
    @overload
    def __new__(cls, /, version: Literal[1], codec: Codec, multihash: str|bytes|BaseMultihash) -> 'CIDv1': ...
    @overload
    def __new__(cls, /, version: int, multihash: str|bytes|BaseMultihash) -> AnyCID: ...
    @overload
    def __new__(cls, /, version: int, codec: Codec, multihash: str|bytes|BaseMultihash) -> AnyCID: ...
    @overload
    def __new__(cls, /, codec: Codec, multihash: str|bytes|BaseMultihash) -> 'CIDv1': ...

    def __new__(cls, *args, **kwargs) -> AnyCID:
        # Subclass fast path
        if cls is not CID:
            return super().__new__(cls)
        
        # Copy constructor
        if args and isinstance(cid := args[0], CID):
            return cid

        # Factory method for CID construction
        args = list(args)
        if (multihash := kwargs.get('multihash')) is None:
            match args:
                case [CIDv0()|CIDv1() as cid]: return cid
                case [str(data)|bytes(data)]:
                    v, c, mh = cls.parse(data)
                    if v == 0:
                        if c != CIDv0.CODEC:
                            raise ValueError(f"CIDv0 requires codec {CIDv0.CODEC}, got {c}")
                        return CIDv0(mh)
                    return CIDv1(c, mh)
                case []: raise ValueError("Multihash required for CID construction.")
                case _: multihash = args.pop()
        
        if (codec := kwargs.get('codec')) is None:
            match args:
                case [0]|[0, "dag-pb"]: return CIDv0(multihash)
                case []: raise ValueError("Codec required for CID construction.")
                case _: codec = args.pop()
        
        match args:
            case []|[1]: return CIDv1(codec, multihash)
            case _: raise ValueError("invalid arguments")
    
    def __init__(self, data: bytes, /):
        # Sanity check the data type because otherwise they're tiny bug bombs
        if isinstance(data, bytes):
            super().__init__()
            object.__setattr__(self, 'buffer', data)
        else:
            raise TypeError(f"CID.__init__ received non-bytes {data}")

    @property
    def version(self) -> int:
        """CID version"""
        raise NotImplementedError("version")

    @property
    def codec(self) -> Codec:
        """CID codec"""
        raise NotImplementedError("codec")

    @property
    def multihash(self) -> Multihash:
        """CID multihash"""
        raise NotImplementedError("multihash")
    
    def encode(self, encoding: multibase.Encoding="base32") -> str:
        """Encoded representation of the CID."""
        raise NotImplementedError("encode")
    
    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        yield self.version
        yield self.codec
        yield self.multihash

    def __repr__(self):
        return f"{type(self).__name__}(version={self.version}, codec={self.codec}, multihash={self.multihash!r})"

    def __eq__(self, other):
        if isinstance(other, (str, bytes)):
            try:
                other = CID(other)
            except ValueError:
                return NotImplemented
        elif not isinstance(other, CID):
            return NotImplemented
        return self.buffer == other.buffer
    
    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, CID):
            return NotImplemented
        return self.buffer < other.buffer
    
    def __hash__(self):
        return hash(self.buffer)
    
    def __str__(self):
        return self.encode()
    
    def __bytes__(self):
        """
        Returns the raw byte representation of the CID.
        This is useful for serialization and storage.
        """
        return self.buffer
    
    def hex(self) -> str:
        """
        Returns the hexadecimal representation of the CID.
        This is useful for debugging and logging.
        """
        return self.buffer.hex()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Integrates the CID class with Pydantic's validation and serialization,
        correctly handling the factory pattern where __new__ returns subclasses.
        """
        def validate_cid(value: Any) -> 'CID':
            """Validate and convert input to a CID object."""
            if isinstance(value, CID):
                return value
            try:
                return cls(value)
            except Exception as e:
                raise ValueError(f"Invalid CID: {value!r}") from e

        return core_schema.no_info_plain_validator_function(
            validate_cid,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: str(instance),
                return_schema=core_schema.str_schema(),
            ),
        )

    @classmethod
    def is_cid(cls, cidstr: str|bytes) -> bool:
        """
        Checks if a given input string is valid encoded CID or not.
        It takes same input as `cid.make_cid` method with a single argument

        :param cidstr: input string which can be a
            - base58-encoded multihash
            - multihash
            - multibase-encoded multihash
        :return: if the value is a valid CID or not
        """
        try: return bool(CID(cidstr))
        except ValueError:
            return False

    @staticmethod
    def parse(raw: 'str|bytes|BaseMultihash|CID') -> tuple[Version, Codec, bytes]:
        """
        Parses a CID string and returns a CID object.

        :param cidstr: input string which can be a
            - base58-encoded multihash
            - multibase-encoded CID
        :return: a CID object
        """
        match raw:
            case CIDv0():
                return 0, CIDv0.CODEC, raw.buffer
            case CIDv1():
                return 1, raw.codec, raw.multihash.buffer
            case CID():
                raise NotImplementedError(f"Unknown CID type {type(raw)}")
            case BaseMultihash():
                return 0, CIDv0.CODEC, raw.buffer
            
            case str():
                if multibase.is_encoded(raw):
                    raw = multibase.decode(raw)
                else:
                    raw = multibase.base58.decode(raw)
        
        if len(raw) <= 2:
            raise ValueError(
                f"Invalid CID length {len(raw)}. Expected at least 2 bytes for version and codec."
            )

        # if the bytestream is a CID
        version, data = raw[0], raw[1:]
        codec = cast(Codec, multicodec.get_codec(data))
        multihash = multicodec.remove_prefix(data)

        try: Multihash(multihash) # validate multihash with exception
        except ValueError:
            raise
        
        if version != 0 and version != 1:
            raise ValueError(f"Unsupported CID version {version}, expected 0 or 1")
        return version, codec, multihash
    
    @classmethod
    def build(cls, version: Version, codec: Codec, multihash: str|bytes|BaseMultihash) -> bytes:
        """
        Constructs a CID byte string from its components.

        :param version: CID version
        :param codec: CID codec
        :param multihash: Multihash for the CID
        :return: CID byte string
        """
        if isinstance(multihash, BaseMultihash):
            multihash = multihash.buffer

        if version == 0 and codec == CIDv0.CODEC:
            if isinstance(multihash, str):
                multihash = multibase.base58.decode(multihash)
            return multihash
        
        if isinstance(multihash, str):
            multihash = multibase.decode(multihash)
        return b'\1' + multicodec.add_prefix(codec, multihash)
    
    @classmethod
    def normalize(cls, cid: "str|bytes|BaseMultihash|CID") -> bytes:
        """
        Normalizes a CID string or bytes to its canonical byte representation.

        :param cid: CID string or bytes
        :return: normalized CID byte representation
        """
        return cls.build(*CID.parse(cid))

class CIDv0(CID):
    """ CID version 0 object """

    CODEC = 'dag-pb'

    def __new__(cls, data: 'str|bytes|BaseMultihash|CID', /) -> 'CIDv0':
        if isinstance(data, BaseMultihash):
            return super().__new__(cls, 0, "dag-pb", data.buffer)
        else:
            self = super().__new__(cls, data)
            if isinstance(self, CIDv0):
                return self
        raise TypeError(f"Expected CIDv0, got {type(self).__name__}")
    
    def __init__(self, data: 'str|bytes|BaseMultihash|CIDv0|CIDv1', /):
        if isinstance(data, CIDv0):
            # CIDv0(c := CIDv0()) is c
            if not hasattr(self, 'buffer'):
                super().__init__(data.buffer)
            return
        super().__init__(self.normalize(data))

    def __repr__(self):
        return f"CIDv0({self.multihash!r})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Integrates the CID class with Pydantic's validation and serialization,
        correctly handling the factory pattern where __new__ returns subclasses.
        """
        def validate_cidv0(value) -> 'CIDv0':
            """Validate and convert input to a CIDv1 object."""
            try:
                match cid := CID(value):
                    case CIDv1(): return CIDv0(cid)
                    case CIDv0(): return cid
                    case _: raise ValueError(f"Invalid CIDv0 input: {value}")
            except Exception as e:
                raise ValueError(f"Invalid CIDv0: {e}") from e

        return core_schema.no_info_plain_validator_function(
            validate_cidv0,
            serialization=core_schema.plain_serializer_function_ser_schema(
                str, return_schema=core_schema.str_schema(),
            )
        )
    
    @classmethod
    def build(cls, version: Version, codec: Codec, multihash: str|bytes|BaseMultihash) -> bytes:
        """
        Constructs a CIDv0 byte string from its components.

        :param multihash: Multihash for the CID
        :return: CIDv0 byte string
        """
        if version != 0:
            raise ValueError(f"CIDv0 requires version 0, got {version}")
        if codec != cls.CODEC:
            raise ValueError(f"CIDv0 requires codec {cls.CODEC}, got {codec}")
        return super().build(0, cls.CODEC, multihash)

    @property
    @override
    def version(self):
        return 0
    
    @property
    @override
    def codec(self):
        return self.CODEC
    
    @property
    @override
    def multihash(self):
        return Multihash("base58btc", self.buffer)

    @override
    def encode(self, encoding: multibase.Encoding="base58btc") -> str:
        """Encode with base58."""
        if encoding != "base58btc":
            raise ValueError('CIDv0 does not support encoding, use CIDv1 instead')
        return multibase.base58.encode(self.buffer)

class CIDv1(CID):
    """ CID version 1 object """

    @overload
    def __new__(cls, data: str|bytes|CID, /) -> Self: ...
    @overload
    def __new__(cls, /, codec: Codec, multihash: str|bytes|BaseMultihash) -> Self: ...

    def __new__(cls, *args, **kwargs) -> Self:
        self = super().__new__(cls, *args, **kwargs)
        if isinstance(self, CIDv1):
            return self # type: ignore
        raise TypeError(f"Expected CIDv1, got CIDv0")

    @overload
    def __init__(self, data: str|bytes|CID, /): ...
    @overload
    def __init__(self, codec: Codec, multihash: str|bytes|BaseMultihash): ...
    
    def __init__(self, codec: str|bytes|CID, multihash: Optional[str|bytes|BaseMultihash] = None):
        """
        :param codec: codec for the CID
        :param multihash: multihash for the CID, if not provided, it is cidexpected that `codec` is a multibase encoded string
        """
        if multihash is None:
            # CIDv1(c := CIDv1()) is c
            if isinstance(codec, CIDv1) and hasattr(self, 'buffer'):
                return
            return super().__init__(self.normalize(codec))
        
        if isinstance(codec, str):
            return super().__init__(
                self.build(1, cast(Codec, codec), multihash)
            )
        
        raise TypeError(
            f'`codec` should be a str, got {type(codec).__name__}. If you want to use a multibase-encoded CID, pass it as a single argument.'
        )

    def __repr__(self):
        return f"CIDv1({self.codec!r}, {self.multihash!r})"

    @classmethod    
    def __get_pydantic_core_schema__(
        cls, source_type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Integrates the CIDv1 class with Pydantic's validation and serialization.
        """
        def validate_cidv1(value) -> 'CIDv1':
            """Validate and convert input to a CIDv1 object."""
            match cid := CID(value):
                case CIDv1(): return cid
                case CIDv0(): return CIDv1(cid)
                case _: raise ValueError(f"Invalid CIDv1 input: {value}")
        
        return core_schema.union_schema([
            # Accept existing CIDv1 objects
            core_schema.is_instance_schema(CIDv1),
            # Accept strings that can be converted to CIDv1
            core_schema.no_info_after_validator_function(
                validate_cidv1,
                core_schema.str_schema(
                    pattern=r'^[a-zA-Z0-9+/=-]+$',
                    min_length=10,
                    max_length=200,
                )
            )
        ], serialization=core_schema.plain_serializer_function_ser_schema(
            str, return_schema=core_schema.str_schema(),
        ))
    
    @classmethod
    def __get_pydantic_json_schema__(
            cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
        ) -> JsonSchemaValue:
        """
        Modifies the JSON schema for MCP compatibility.
        This method is called by Pydantic when generating JSON schemas.
        """
        json_schema = handler(core_schema)
        json_schema.update({
            'type': 'string',
            'format': 'cid',
            'pattern': r'^[a-zA-Z0-9+/=-]+$',
            'minLength': 10,
            'maxLength': 200,
            'description': 'Content Identifier (CID) version 1 - a self-describing content-addressed identifier',
            'examples': [
                'zb2rhe5P4gXftAwvA4eXQ5HJwsER2owDyS9sKaQRRVQPn93bA',
                'bagaaiera5fltyykmoa6jfwfzgq54z42hs7vygbbvgkdkdimwvg7zzx5s7h5bq'
            ],
            'title': 'CIDv1'
        })
        return json_schema

    @classmethod
    def build(cls, version: Version, codec: Codec, multihash: str|bytes|BaseMultihash) -> bytes:
        """
        Constructs a CIDv1 byte string from its components.

        :param codec: Codec for the CID
        :param multihash: Multihash for the CID
        :return: CIDv1 byte string
        """
        if version != 1:
            raise ValueError(f"CIDv1 requires version 1, got {version}")
        return super().build(1, codec, multihash)

    @property
    @override
    def version(self):
        return 1
    
    @property
    @override
    def codec(self):
        """Codec for the CID, without multibase prefix."""
        return cast(Codec, multicodec.get_codec(self.buffer[1:]))
    
    @property
    @override
    def multihash(self):
        """Multihash for the CID, without multibase prefix."""
        return Multihash(multicodec.remove_prefix(self.buffer[1:]))

    @override
    def encode(self, encoding: multibase.Encoding='base32') -> str:
        """
        Encoded version of the raw representation

        :param str encoding: the encoding to use
        :return: encoded raw representation with the given encoding
        """
        return multibase.encode(encoding, self.buffer)

def cidhash(data: bytes, *, function: str = 'sha2-256', codec: Codec='dag-cbor') -> CIDv1:
    """
    Create a Multihash from the given data using the specified hash function.

    :param data: The data to hash.
    :param name: The name of the hash function to use.
    :return: A Multihash object.
    """
    return CIDv1(codec, multihash(function, data).digest)