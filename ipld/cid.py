from typing import Any, Literal, Optional, Self, cast, overload, override

import base58
import multibase
import multicodec
import multihash as mh

__all__ = (
    'Codec',
    'CID', 'CIDv0', 'CIDv1', 'AnyCID'
)

type Codec = Literal['raw', 'dag-pb', 'dag-cbor', 'dag-json']
type AnyCID = 'CIDv0|CIDv1'

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
        args = list(args)
        if (multihash := kwargs.get('multihash')) is None:
            match args:
                case [str(data)]: return cls.from_string(data)
                case [bytes(data)]: return cls.from_bytes(data)
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
        super().__setattr__('buffer', _ensure_bytes(data))

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
    
    def encode(self, encoding: str="") -> bytes:
        """
        Encoded representation of the CID

        :param str encoding: the encoding to use to encode the raw representation, should be supported by
        :return: encoded representation of the CID
        :rtype: str
        """
        raise NotImplementedError("encode")
    
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
        return self.encode().decode('utf-8')

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

    @classmethod
    def from_string(cls, cidstr: str) -> AnyCID:
        """
        Creates a CID object from a encoded form

        :param str cidstr: can be

            - base58-encoded multihash
            - multihash
            - multibase-encoded multihash
        :return: a CID object
        :rtype: :py:class:`cid.CIDv0` or :py:class:`cid.CIDv1`
        """
        return cls.from_bytes(_ensure_bytes(cidstr, 'utf-8'))

    @classmethod
    def from_bytes(cls, cidbytes: bytes) -> AnyCID:
        """
        Creates a CID object from a encoded form

        :param bytes cidbytes: can be

            - base58-encoded multihash
            - multihash
            - multibase-encoded multihash
        :return: a CID object
        :rtype: :py:class:`cid.CIDv0` or :py:class:`cid.CIDv1`
        :raises: `ValueError` if the base58-encoded string is not a valid string
        :raises: `ValueError` if the length of the argument is zero
        :raises: `ValueError` if the length of decoded CID is invalid
        """
        if len(cidbytes) < 2:
            raise ValueError('argument length can not be zero')

        version: int
        codec: Codec

        # first byte for identity multibase and CIDv0 is 0x00
        # putting in assumption that multibase for CIDv0 can not be identity
        # refer: https://github.com/ipld/cid/issues/13#issuecomment-326490275
        if cidbytes[0] != 0 and multibase.is_encoded(cidbytes):
            # if the bytestream is multibase encoded
            cid = multibase.decode(cidbytes)

            if len(cid) < 2:
                raise ValueError('cid length is invalid')

            version = int(cid[0])
            data = cid[1:]
            codec = cast(Codec, multicodec.get_codec(data))
            multihash = multicodec.remove_prefix(data)
        elif cidbytes[0] in (0, 1):
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
            except ValueError:
                raise ValueError('multihash is not a valid base58 encoded multihash') from None

        try:
            mh.decode(multihash)
        except ValueError:
            raise

        return CID(version, codec, multihash)

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

    def __init__(self, multihash: bytes):
        """
        :param bytes multihash: multihash for the CID
        """
        super(CIDv0, self).__init__(multihash)

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
    def encode(self, encoding=None) -> bytes:
        """
        base58-encoded buffer

        :return: encoded representation or CID
        :rtype: bytes
        """
        if encoding is not None:
            raise ValueError('CIDv0 does not support encoding, use CIDv1 instead')
        return _ensure_bytes(base58.b58encode(self.buffer))

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

    @property
    @override
    def version(self):
        return 1
    
    @property
    @override
    def codec(self):
        """Codec for the CID, without multibase prefix."""
        return cast(Codec, multicodec.get_codec(self.buffer))
    
    @property
    @override
    def multihash(self):
        """Multihash for the CID, without multibase prefix."""
        return multicodec.remove_prefix(self.buffer)

    @override
    def encode(self, encoding: str='base58btc') -> bytes:
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
            raise ValueError('CIDv1 can only be converted for codec {}'.format(CIDv0.CODEC))

        return CIDv0(self.multihash)