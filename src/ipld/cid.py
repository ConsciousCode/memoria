from typing import Any, ClassVar, Literal, Self, cast, overload, override

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from ._common import Immutable

from .multihash import BaseMultihash, Multihash, multihash
from . import multibase, multicodec

__all__ = (
    'CIDVersion', 'BlockCodec',
    'CID', 'CIDv0', 'CIDv1', 'AnyCID'
)

type CIDVersion = Literal[0, 1]
type BlockCodec = Literal[
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

    'raw', 'dag-pb', 'dag-cbor', 'dag-json',
    'cbor', 'libp2p-key', 'git-raw', 'torrent-info', 'torrent-file',
    'blake3-hashseq', 'leofcoin-block', 'leofcoin-tx', 'leofcoin-pr',
    'dag-jose', 'dag-cose', 'eth-block', 'eth-block-list',
    'eth-tx-trie', 'eth-tx', 'eth-tx-receipt-trie',
    'eth-tx-receipt', 'eth-state-trie', 'eth-account-snapshot',
    'eth-storage-trie', 'eth-receipt-log-trie', 'eth-receipt-log',
    'bitcoin-block', 'bitcoin-tx', 'bitcoin-witness-commitment',
    'zcash-block', 'zcash-tx', 'stellar-block', 'stellar-tx',
    'decred-block', 'decred-tx', 'dash-block', 'dash-tx',
    'swarm-manifest', 'swarm-feed', 'beeson', 'swhid-1-snp', 'json'
]
'''Multicodec type for IPLD block codecs.'''

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
    def __new__(cls, /, version: Literal[1], codec: BlockCodec, multihash: str|bytes|BaseMultihash) -> 'CIDv1': ...
    @overload
    def __new__(cls, /, version: int, multihash: str|bytes|BaseMultihash) -> AnyCID: ...
    @overload
    def __new__(cls, /, version: int, codec: BlockCodec, multihash: str|bytes|BaseMultihash) -> AnyCID: ...
    @overload
    def __new__(cls, /, codec: BlockCodec, multihash: str|bytes|BaseMultihash) -> 'CIDv1': ...

    def __new__(cls, *args, **kwargs) -> AnyCID:
        # Subclass fast path
        if cls is not CID:
            return super().__new__(cls)
        
        # Copy constructor
        if args and isinstance(arg := args[0], CID):
            return arg
        
        # Direct data constructor because it's the only one that can be
        # called without multihash
        if len(args) == 1 and not kwargs:
            if isinstance(arg := args[0], (str, bytes, BaseMultihash)):
                return cls.build(*cls.parse(arg))
            raise TypeError(
                f"CID.__new__ received unexpected argument {arg!r} of type {type(arg)}"
            )

        # We basically want to "peel off" args from the right
        args = list(args)
        if (multihash := kwargs.pop('multihash', None)) is None:
            if not args:
                raise ValueError("CID construction requires multihash.")
            multihash = args.pop()
        
        # Codec and version are interdependent so we have to untangle them
        version = None
        if (codec := kwargs.pop('codec', None)) is None:
            if args:
                codec = args.pop()
            else:
                match version := kwargs.get('version'):
                    case 0: codec = CIDv0.CODEC
                    case None: raise ValueError(
                        "CID construction requires codec or version."
                    )
                    case 1: raise ValueError(
                        "CIDv1 requires codec, got version 1 without codec."
                    )
                    case _: raise ValueError(
                        f"Unsupported CID version {kwargs['version']},"
                        " expected 0 or 1"
                    )
        
        if version is None:
            if (version := kwargs.get('version')) is None:
                version = args.pop() if args else 1
        
        if args or kwargs:
            raise TypeError(
                f"CID.__new__ received unexpected arguments: {args}, {kwargs}"
            )
        
        if version != 0 and version != 1:
            raise ValueError(f"Unsupported CID version {version}, expected 0 or 1")
        
        if not isinstance(codec, str):
            raise TypeError(
                f"CID requires codec as str, got {type(codec)}"
            )

        if not isinstance(multihash, (str, bytes, BaseMultihash)):
            raise TypeError(
                f"CID requires multihash as str, bytes or BaseMultihash, got {type(multihash)}"
            )
        
        return cls.build(version, cast(BlockCodec, codec), multihash)
    
    # We repeat the overloads of __new__ on __init__ too, but CID.__init__
    # is only ever called by subclasses to set the buffer.
    @overload
    def __init__(self, cid: 'CID', /): ...
    @overload
    def __init__(self, data: str|bytes, /): ...
    @overload
    def __init__(self, /, version: Literal[0], codec: Literal['dag-pb'], multihash: str|bytes|BaseMultihash): ...
    @overload
    def __init__(self, /, version: Literal[0], multihash: str|bytes|BaseMultihash): ...
    @overload
    def __init__(self, /, version: Literal[1], codec: BlockCodec, multihash: str|bytes|BaseMultihash): ...
    @overload
    def __init__(self, /, version: int, multihash: str|bytes|BaseMultihash): ...
    @overload
    def __init__(self, /, version: int, codec: BlockCodec, multihash: str|bytes|BaseMultihash): ...
    @overload
    def __init__(self, /, codec: BlockCodec, multihash: str|bytes|BaseMultihash): ...

    def __init__(self, data: bytes, /): # type: ignore
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
    def codec(self) -> BlockCodec:
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
        if isinstance(other, str|bytes):
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
    def parse(raw: 'str|bytes|BaseMultihash|CID') -> tuple[CIDVersion, BlockCodec, Multihash]:
        """
        Parses a CID string and returns a CID object.

        :param cidstr: input string which can be a
            - base58-encoded multihash
            - multibase-encoded CID
        :return: a CID object
        """
        match raw:
            case CIDv0():
                return 0, CIDv0.CODEC, Multihash(raw.buffer)
            case CIDv1():
                return 1, raw.codec, Multihash(raw.multihash.buffer)
            case CID():
                raise NotImplementedError(f"Unknown CID type {type(raw)}")
            case BaseMultihash():
                return 0, CIDv0.CODEC, Multihash(raw.buffer)
            
            case str():
                if multibase.is_encoded(raw):
                    raw = multibase.decode(raw)
                    version, raw = raw[0], raw[1:]
                    codec, multihash = multicodec.split_codec(raw)
                else:
                    version = 0
                    codec = CIDv0.CODEC
                    multihash = multibase.base58.decode(raw)
            
            case bytes():
                version, raw = raw[0], raw[1:]
                codec, multihash = multicodec.split_codec(raw)

            case _:
                raise NotImplementedError(f"CID({type(raw)})")
        
        if len(multihash) <= 2:
            raise ValueError(
                f"Invalid CID length {len(multihash)}."
                " Expected at least 2 bytes for version and codec."
            )

        try: Multihash(multihash) # validate multihash with exception
        except ValueError:
            raise
        
        if version != 0 and version != 1:
            raise ValueError(f"Unsupported CID version {version}, expected 0 or 1")
        return version, cast(BlockCodec, codec), Multihash(multihash)
    
    @classmethod
    def combine(cls, version: CIDVersion, codec: BlockCodec, multihash: str|bytes|BaseMultihash) -> bytes:
        """
        Constructs a CID byte string from its components.

        :param version: CID version
        :param codec: CID codec
        :param multihash: Multihash for the CID
        :return: CID byte string
        """
        if isinstance(multihash, BaseMultihash):
            multihash = multihash.buffer

        if version == 0:
            if codec == CIDv0.CODEC:
                if isinstance(multihash, str):
                    multihash = multibase.base58.decode(multihash)
                if not isinstance(multihash, bytes):
                    raise TypeError(
                        f"CIDv0 requires multihash as bytes, got {type(multihash)}"
                    )
                return multihash
            raise ValueError(f"CIDv0 requires codec {CIDv0.CODEC}, got {codec}")
        elif version != 1:
            raise ValueError(
                f"Unsupported CID version {version!r}, expected 0 or 1"
            )
        
        if isinstance(multihash, str):
            multihash = multibase.decode(multihash)
        return b'\1' + multicodec.add_prefix(codec, multihash)
    
    @classmethod
    def build(cls, version: CIDVersion, codec: BlockCodec, multihash: str|bytes|BaseMultihash) -> 'CIDv0|CIDv1':
        if version == 0:
            if codec != CIDv0.CODEC:
                raise ValueError(
                    f"CIDv0 requires codec {CIDv0.CODEC}, got {codec}"
                )
            return CIDv0(multihash)
        elif version != 1:
            raise ValueError(
                f"Unsupported CID version {version!r}, expected 0 or 1"
            )
        return CIDv1(codec, multihash)

    @classmethod
    def normalize(cls, cid: "str|bytes|BaseMultihash|CID") -> bytes:
        """
        Normalizes a CID string, multihash, or bytes to its canonical byte
        representation.

        :param cid: CID string or bytes
        :return: normalized CID byte representation
        """
        return cls.combine(*CID.parse(cid))

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
            try: return cls(value)
            except Exception as e:
                raise ValueError(f"Invalid CID: {value!r}") from e

        return core_schema.union_schema([
            # Accept existing CID objects
            core_schema.is_instance_schema(CID),
            # Accept strings that can be converted to CID
            core_schema.no_info_after_validator_function(
                validate_cid, core_schema.str_schema()
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
        try:
            json_schema = handler(core_schema)
            json_schema.update({
                'type': 'string',
                'format': 'cid',
                'pattern': r'^(Qm[1-9A-HJ-NP-Za-km-z]+|b[a-z2-7]+)$',
                'minLength': 46,
                'maxLength': 46,
                'description': 'Content Identifier (CID) of any version - a self-describing content-addressed identifier',
                'examples': [
                    'QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG',
                    'bagaaiera5fltyykmoa6jfwfzgq54z42hs7vygbbvgkdkdimwvg7zzx5s7h5bq'
                ],
                'title': 'CID'
            })
            return json_schema
        except:
            print("Hello", core_schema)
            import traceback
            traceback.print_exc()
            print("Here")
            raise
    
    @overload
    @staticmethod
    def hash(data: bytes, *,
            version: Literal[0],
            codec: Literal['dag-pb'],
            function: str
        ) -> 'CIDv0': ...
    
    @overload
    @staticmethod
    def hash(data: bytes, *,
            version: Literal[1],
            codec: BlockCodec,
            function: str
        ) -> 'CIDv1': ...
    
    @overload
    @staticmethod
    def hash(data: bytes, *,
            version: Literal[0, 1],
            codec: BlockCodec,
            function: str
        ) -> 'CIDv0 | CIDv1': ...
    
    @staticmethod
    def hash(data: bytes, *,
            version: Literal[0, 1]=1,
            codec: BlockCodec='dag-cbor',
            function: str = 'sha2-256'
        ) -> 'CIDv0 | CIDv1':
        """
        Create a CID from the given data using the specified hash function.
        :param data: The data to hash.
        :param version: The CID version (0 or 1).
        :param codec: The codec for the CID.
        :param function: The hash function to use (default is 'sha2-256').
        :return: A CID object.
        """
        if version == 0:
            if codec != CIDv0.CODEC:
                raise TypeError(
                    f"CIDv0 requires codec {CIDv0.CODEC}, got {codec}"
                )
            return CIDv0(multihash(function, data).digest)
        elif version != 1:
            raise TypeError(
                f"Unsupported CID version {version!r}, expected 0 or 1"
            )
        else:
            return CIDv1(codec, multihash(function, data).digest)

class CIDv0(CID):
    """ CID version 0 object """

    CODEC: ClassVar[BlockCodec] = 'dag-pb'

    def __new__(cls, data: 'str|bytes|BaseMultihash|CID', /) -> 'CIDv0':
        if isinstance(data, BaseMultihash):
            return super().__new__(cls, 0, "dag-pb", data.buffer)
        self = super().__new__(cls, data)
        if isinstance(self, CIDv0):
            return self
        raise TypeError(f"Expected CIDv0, got {type(self).__name__}")
    
    def __init__(self, data: 'str|bytes|BaseMultihash|CIDv0|CIDv1', /):
        if isinstance(data, CIDv0):
            # CIDv0(c := CIDv0()) is c
            if not hasattr(self, 'buffer'):
                super().__init__(data.buffer)
        else:
            super().__init__(self.normalize(data))

    def __repr__(self):
        return f"CIDv0({self.multihash!r})"
    
    @classmethod
    def combine(cls, version: CIDVersion, codec: BlockCodec, multihash: str|bytes|BaseMultihash) -> bytes:
        """
        Constructs a CIDv0 byte string from its components.

        :param multihash
                cid_version=cid_version,
                codec=codec,: Multihash for the CID
        :return: CIDv0 byte string
        """
        if version != 0:
            raise ValueError(f"CIDv0 requires version 0, got {version}")
        if codec != cls.CODEC:
            raise ValueError(f"CIDv0 requires codec {cls.CODEC}, got {codec}")
        return super().combine(0, cls.CODEC, multihash)

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
        return Multihash("sha2-256", self.buffer)

    @override
    def encode(self, encoding: multibase.Encoding="base58btc") -> str:
        """Encode with base58."""
        if encoding != "base58btc":
            raise ValueError('CIDv0 does not support encoding, use CIDv1 instead')
        return multibase.base58.encode(self.buffer)

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

        return core_schema.union_schema([
            # Accept existing CIDv0 objects
            core_schema.is_instance_schema(CIDv1),
            # Accept strings that can be converted to CIDv0
            core_schema.no_info_after_validator_function(
                validate_cidv0,
                core_schema.str_schema(
                    pattern=r'^Qm[1-9A-HJ-NP-Za-km-z]+$',
                    min_length=46,
                    max_length=46,
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
            'format': 'cidv0',
            'pattern': r'^Qm[1-9A-HJ-NP-Za-km-z]+$',
            'minLength': 46,
            'maxLength': 46,
            'description': 'Content Identifier (CID) version 0 - a self-describing content-addressed identifier',
            'examples': [
                'QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG',
                'QmRgR7Bpa9xDMUNGiKaARvFL9MmnoFyd86rF817EZyfdGE'
            ],
            'title': 'CIDv0'
        })
        return json_schema
    
    @overload
    @staticmethod
    def hash(data: bytes, *,
            version: Literal[0],
            codec: Literal['dag-pb'],
            function: str
        ) -> 'CIDv0': ...
    
    @overload
    @staticmethod
    def hash(data: bytes, *,
            version: Literal[1],
            codec: BlockCodec,
            function: str
        ) -> 'CIDv1': ...
    
    @overload
    @staticmethod
    def hash(data: bytes, *,
            version: Literal[0, 1],
            codec: BlockCodec,
            function: str
        ) -> 'CIDv0 | CIDv1': ...
    
    @overload
    @staticmethod
    def hash(data: bytes) -> 'CIDv0': ...
    
    @staticmethod
    def hash(data: bytes, *,
            version: Literal[0, 1]=0,
            codec: BlockCodec='dag-pb',
            function: str = 'sha2-256'
        ) -> 'CIDv0 | CIDv1':
        return CID.hash(data, version=version, codec=codec, function=function)

class CIDv1(CID):
    """ CID version 1 object """

    @overload
    def __new__(cls, data: str|bytes|CID, /) -> Self: ...
    @overload
    def __new__(cls, /, version: Literal[1], codec: BlockCodec, multihash: str|bytes|BaseMultihash) -> Self: ...
    @overload
    def __new__(cls, /, codec: BlockCodec, multihash: str|bytes|BaseMultihash) -> Self: ...

    def __new__(cls, *args, **kwargs) -> Self:
        self = super().__new__(cls, *args, **kwargs)
        if isinstance(self, CIDv1):
            return self # type: ignore
        raise TypeError(f"Expected CIDv1, got CIDv0")

    @overload
    def __init__(self, data: str|bytes|CID, /): ...
    @overload
    def __init__(self, version: Literal[1], codec: BlockCodec, multihash: str|bytes|BaseMultihash): ...
    @overload
    def __init__(self, codec: BlockCodec, multihash: str|bytes|BaseMultihash): ...
    
    def __init__(self, *args, **kwargs):
        """
        :param codec: codec for the CID
        :param multihash: multihash for the CID, if not provided, it is cidexpected that `codec` is a multibase encoded string
        """

        if len(args) == 1 and not kwargs:
            # CIDv1(c := CIDv1()) is c
            if isinstance(arg := args[0], CIDv1) and hasattr(self, 'buffer'):
                return
            return super().__init__(self.normalize(arg))

        args = list(args)
        if (multihash := kwargs.pop("multihash", None)) is None:
            if not args:
                raise ValueError("CIDv1 construction requires multihash.")
            multihash = args.pop()
        
        if (codec := kwargs.pop('codec', None)) is None:
            if not args:
                raise ValueError("CIDv1 construction requires codec.")
            codec = args.pop()
        
        if (version := kwargs.pop('version', None)) is None:
            if args:
                if (version := args.pop()) != 1:
                    raise ValueError(
                        f"CIDv1 requires version 1, got {version}"
                    )
            else:
                version = 1
        
        if args or kwargs:
            raise TypeError(
                f"CIDv1.__init__ received unexpected arguments: {args}, {kwargs}"
            )

        super().__init__(self.combine(version, codec, multihash))

    def __repr__(self):
        return f"CIDv1({self.codec!r}, {self.multihash!r})"

    @classmethod
    def combine(cls, version: CIDVersion, codec: BlockCodec, multihash: str|bytes|BaseMultihash) -> bytes:
        """
        Constructs a CIDv1 byte string from its components.

        :param codec: Codec for the CID
        :param multihash: Multihash for the CID
        :return: CIDv1 byte string
        """
        if version != 1:
            raise ValueError(f"CIDv1 requires version 1, got {version}")
        return super().combine(1, codec, multihash)

    @property
    @override
    def version(self):
        return 1
    
    @property
    @override
    def codec(self):
        """Codec for the CID, without multibase prefix."""
        return cast(BlockCodec, multicodec.get_codec(self.buffer[1:]))
    
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
                validate_cidv1, core_schema.str_schema()
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
            'format': 'cidv1',
            'minLength': 62,
            'maxLength': 64,
            'description': 'Content Identifier (CID) version 1 - a self-describing content-addressed identifier',
            'examples': [
                'zb2rhe5P4gXftAwvA4eXQ5HJwsER2owDyS9sKaQRRVQPn93bA',
                'bagaaiera5fltyykmoa6jfwfzgq54z42hs7vygbbvgkdkdimwvg7zzx5s7h5bq'
            ],
            'title': 'CIDv1'
        })
        return json_schema
    
    @overload
    @staticmethod
    def hash(data: bytes, *,
            version: Literal[0],
            codec: Literal['dag-pb'],
            function: str
        ) -> 'CIDv0': ...
    
    @overload
    @staticmethod
    def hash(data: bytes, *,
            version: Literal[1],
            codec: BlockCodec,
            function: str
        ) -> 'CIDv1': ...
    
    @overload
    @staticmethod
    def hash(data: bytes, *,
            version: Literal[0, 1],
            codec: BlockCodec,
            function: str
        ) -> 'CIDv0 | CIDv1': ...

    @overload
    @staticmethod
    def hash(data: bytes) -> 'CIDv1': ...
    
    @staticmethod
    def hash(data: bytes, *,
            version: Literal[0, 1]=1,
            codec: BlockCodec='dag-cbor',
            function: str = 'sha2-256'
        ) -> 'CIDv0 | CIDv1':
        return CID.hash(data, version=version, codec=codec, function=function)