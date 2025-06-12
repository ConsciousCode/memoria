from typing import Any, Literal, Optional, Self, cast, overload, override

import base58
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from .multihash import Multihash
from . import multibase
from . import multicodec

__all__ = (
    'Version', 'Codec',
    'CID', 'CIDv0', 'CIDv1', 'AnyCID'
)

type Version = Literal[0, 1]
type Codec = Literal[
    'raw', 'dag-pb', 'dag-cbor',
    'libp2p-key',
    'git-raw',
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
    'swarm-manifest', 'swarm-feed',
    'dag-json'
]
type AnyCID = 'CID|CIDv0|CIDv1'

def _ensure_bytes(obj, encoding=None):
    match obj:
        case bytes(): return obj
        case str(): pass
        case _: obj = str(obj)
    
    return obj.encode(encoding or 'utf-8')

class CID:
    buffer: bytes

    __slots__ = ("buffer",)
    __match_args__ = ("version", "codec", "multihash")

    @overload
    def __new__(cls, cid: 'CIDv0', /) -> 'CIDv0': ...
    @overload
    def __new__(cls, cid: 'CIDv1', /) -> 'CIDv1': ...
    @overload
    def __new__(cls, data: str|bytes, /) -> AnyCID: ...
    @overload
    def __new__(cls, /, version: Literal[0], codec: Literal['dag-pb'], multihash: str|bytes) -> 'CIDv0': ...
    @overload
    def __new__(cls, /, version: Literal[0], multihash: str|bytes) -> 'CIDv0': ...
    @overload
    def __new__(cls, /, version: Literal[1], codec: Codec, multihash: str|bytes) -> 'CIDv1': ...
    @overload
    def __new__(cls, /, version: int, multihash: str|bytes) -> AnyCID: ...
    @overload
    def __new__(cls, /, version: int, codec: Codec, multihash: str|bytes) -> AnyCID: ...
    @overload
    def __new__(cls, /, codec: Codec, multihash: str|bytes) -> 'CIDv1': ...

    def __new__(cls, *args, **kwargs) -> AnyCID:
        if cls is not CID:
            return super().__new__(cls)
        
        args = list(args)
        if (multihash := kwargs.get('multihash')) is None:
            match args:
                case [CIDv0()|CIDv1() as cid]: return cid.copy()
                case [str(data)]: return CID(*cls.parse(data))
                case [bytes(data)]: return CID(*cls.parse(data))
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
    
    def __init__(self, data: str|bytes, /):
        super().__init__()
        super().__setattr__('buffer', CID.normalize(data))

    def __setattr__(self, name: str, value: Any):
        raise AttributeError("CID objects are immutable.")

    @property
    def version(self) -> int:
        """CID version"""
        raise NotImplementedError("version")

    @property
    def codec(self) -> Codec:
        """CID codec"""
        raise NotImplementedError("codec")

    @property
    def multihash(self) -> bytes:
        """CID multihash"""
        raise NotImplementedError("multihash")
    
    def copy(self) -> Self:
        """
        Returns a copy of the CID object.

        :return: a copy of the CID object
        :rtype: CID
        """
        return type(self)(self.buffer)

    def encode(self, encoding: multibase.Encoding="identity") -> str:
        """
        Encoded representation of the CID

        :param str encoding: the encoding to use to encode the raw representation, should be supported by
        :return: encoded representation of the CID
        :rtype: str
        """
        raise NotImplementedError("encode")
    
    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        yield self.version
        yield self.codec
        yield self.multihash

    def __repr__(self):
        def truncate(s, length):
            return s[:length] + b'..' if len(s) > length else s
        
        return f"{type(self).__name__}(version={self.version}, codec={self.codec}, multihash={truncate(self.multihash, 20)})"

    def __eq__(self, other):
        if not isinstance(other, CID):
            return NotImplemented
        return (
            self.version == other.version and
            self.codec == other.codec and
            self.multihash == other.multihash
        )
    
    def __hash__(self):
        return hash(self.buffer)
    
    def __str__(self):
        return self.encode()

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
                raise ValueError(f"Invalid CID: {e}") from e

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
        :type cidstr: str or bytes
        :return: if the value is a valid CID or not
        :rtype: bool
        """
        try:
            return bool(CID(cidstr))
        except ValueError:
            return False

    @staticmethod
    def parse(cidbytes: str|bytes) -> tuple[Version, Codec, bytes]:
        """
        Parses a CID string and returns a CID object.

        :param cidstr: input string which can be a

            - base58-encoded multihash
            - multihash
            - multibase-encoded multihash
        :type cidstr: str or bytes
        :return: a CID object
        :rtype: :py:class:`cid.CIDv0` or :py:class:`cid.CIDv1`
        """
        if len(cidbytes) < 2:
            raise ValueError('argument length can not be zero')
        
        cidbytes = _ensure_bytes(cidbytes, 'utf-8')

        # first byte for identity multibase and CIDv0 is 0x00
        # putting in assumption that multibase for CIDv0 can not be identity
        # refer: https://github.com/ipld/cid/issues/13#issuecomment-326490275
        if cidbytes[0] != 0 and multibase.is_encoded(cidbytes.decode('utf-8')):
            # if the bytestream is multibase encoded
            cid = multibase.decode(cidbytes.decode('utf-8'))

            if len(cid) < 2:
                raise ValueError('cid length is invalid')

            version = int(cid[0])
            data = cid[1:]
            codec = cast(Codec, multicodec.get_codec(data))
            multihash = multicodec.remove_prefix(data)
        elif cidbytes[0] in {0, 1}:
            # if the bytestream is a CID
            version = cidbytes[0]
            data = cidbytes[1:]
            codec = cast(Codec, multicodec.get_codec(data))
            multihash = multicodec.remove_prefix(data)
        else:
            try: # otherwise its just base58-encoded multihash
                version = 0
                codec = CIDv0.CODEC
                multihash = base58.b58decode(cidbytes)
            except ValueError as e:
                raise ValueError('multihash is not a valid base58 encoded multihash') from e

        try:
            Multihash(multihash) # validate multihash
        except ValueError:
            raise
        
        if version != 0 and version != 1:
            raise ValueError(f"Unsupported CID version {version}, expected 0 or 1")
        return version, codec, multihash
    
    @staticmethod
    def unparse(version: Version, codec: Codec, multihash: str|bytes) -> bytes:
        """
        Constructs a CID byte string from its components.

        :param version: CID version
        :type version: int
        :param codec: CID codec
        :type codec: Codec
        :param multihash: Multihash for the CID
        :type multihash: str or bytes
        :return: CID byte string
        :rtype: bytes
        """
        if isinstance(multihash, str):
            multihash = _ensure_bytes(multihash, 'utf-8')
        
        if version == 0 and codec == CIDv0.CODEC:
            return base58.b58encode(multihash)
        
        return b'\1' + multicodec.add_prefix(codec, multihash)
    
    @staticmethod
    def normalize(cid: str|bytes) -> bytes:
        """
        Normalizes a CID string or bytes to its canonical byte representation.

        :param cid: CID string or bytes
        :type cid: str or bytes
        :return: normalized CID byte representation
        :rtype: bytes
        """
        return CID.unparse(*CID.parse(cid))

class CIDv0(CID):
    """ CID version 0 object """

    CODEC = 'dag-pb'

    @overload
    def __new__(cls, data: str|bytes, /) -> 'CIDv0': ...
    @overload
    def __new__(cls, /, version: int, multihash: str|bytes) -> 'CIDv0': ...
    @overload
    def __new__(cls, /, version: int, codec: Codec, multihash: str|bytes) -> 'CIDv0': ...

    def __new__(cls, *args, **kwargs) -> 'CIDv0':
        self = super().__new__(cls, *args, **kwargs)
        if isinstance(self, CIDv0):
            return self
        raise TypeError(f"Expected CIDv0, got {type(self).__name__}")
    
    def __repr__(self):
        return f"CIDv0({base58.b58encode(self.buffer).decode('utf-8')!r})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Integrates the CID class with Pydantic's validation and serialization,
        correctly handling the factory pattern where __new__ returns subclasses.
        """
        def validate_cid(value) -> 'CIDv0':
            """Validate and convert input to a CIDv1 object."""
            try:
                match cid := CID(value):
                    case CIDv1(): return cid.v0()
                    case CIDv0(): return cid
                    case _: raise ValueError(f"Invalid CIDv0 input: {value}")
            except Exception as e:
                raise ValueError(f"Invalid CIDv0: {e}") from e

        return core_schema.no_info_plain_validator_function(
            validate_cid,
            serialization=core_schema.plain_serializer_function_ser_schema(
                str, return_schema=core_schema.str_schema(),
            )
        )

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
        return self.buffer

    @override
    def encode(self, encoding: multibase.Encoding="base58btc") -> str:
        """
        base58-encoded buffer

        :return: encoded representation or CID
        :rtype: bytes
        """
        if encoding != "base58btc":
            raise ValueError('CIDv0 does not support encoding, use CIDv1 instead')
        return base58.b58encode(self.buffer).decode('utf-8')

    def v1(self):
        """
        Get an equivalent :py:class:`cid.CIDv1` object.

        :return: :py:class:`cid.CIDv1` object
        :rtype: :py:class:`cid.CIDv1`
        """
        return CIDv1(self.CODEC, self.buffer)

class CIDv1(CID):
    """ CID version 1 object """

    @overload
    def __new__(cls, data: str|bytes, /) -> 'CIDv1': ...
    @overload
    def __new__(cls, /, version: int, multihash: str|bytes) -> 'CIDv1': ...
    @overload
    def __new__(cls, /, version: int, codec: Codec, multihash: str|bytes) -> 'CIDv1': ...
    @overload
    def __new__(cls, /, codec: Codec, multihash: str|bytes) -> 'CIDv1': ...

    def __new__(cls, *args, **kwargs) -> 'CIDv1':
        self = super().__new__(cls, *args, **kwargs)
        if isinstance(self, CIDv1):
            return self
        raise TypeError(f"Expected CIDv1, got CIDv0")

    def __init__(self, codec: str|bytes, multihash: Optional[str|bytes] = None):
        """
        :param codec: codec for the CID
        :type codec: str or bytes
        :param multihash: multihash for the CID, if not provided, it is cidexpected that `codec` is a multibase encoded string
        :type multihash: str or bytes, optional
        """

        if multihash is None:
            # "codec" is actually the entire CID
            super().__init__(codec)
        elif isinstance(codec, bytes):
            raise TypeError(
                '`codec` should be a string, got bytes. If you want to use a multibase encoded CID, pass it as a single argument.'
            )
        else:
            if isinstance(multihash, str):
                multihash = multibase.decode(multihash)
            super().__init__(multicodec.add_prefix(codec, multihash))

    def __repr__(self):
        return f"CIDv1({multibase.encode('base32', self.buffer)!r})"

    @classmethod    
    def __get_pydantic_core_schema__(
        cls, source_type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Integrates the CIDv1 class with Pydantic's validation and serialization.
        """
        def validate_cidv1(value) -> 'CIDv1':
            """Validate and convert input to a CIDv1 object."""
            try:
                match cid := CID(value):
                    case CIDv1(): return cid
                    case CIDv0(): return cid.v1()
                    case _: raise ValueError(f"Invalid CIDv1 input: {value}")
            except Exception as e:
                raise ValueError(f"Invalid CIDv1: {e}") from e

        return core_schema.no_info_plain_validator_function(
            validate_cidv1,
            serialization=core_schema.plain_serializer_function_ser_schema(
                str, return_schema=core_schema.str_schema(),
            ),
        )

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
        return multicodec.remove_prefix(self.buffer)

    @override
    def encode(self, encoding: multibase.Encoding='base58btc') -> str:
        """
        Encoded version of the raw representation

        :param str encoding: the encoding to use to encode the raw representation, should be supported by
            ``py-multibase``
        :return: encoded raw representation with the given encoding
        :rtype: bytes
        """
        return multibase.encode(encoding, self.buffer)

    def v0(self):
        """
        Get an equivalent :py:class:`cid.CIDv0` object.

        :return: :py:class:`cid.CIDv0` object
        :rtype: :py:class:`cid.CIDv0`
        :raise ValueError: if the codec is not 'dag-pb'
        """
        if self.codec != CIDv0.CODEC:
            raise ValueError(f'CIDv1 can only be converted for codec {CIDv0.CODEC}')

        return CIDv0(self.multihash)