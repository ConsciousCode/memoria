from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Data(_message.Message):
    __slots__ = ["Data", "Type", "blocksizes", "fanout", "filesize", "hashType"]
    class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    BLOCKSIZES_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    Data: bytes
    Directory: Data.DataType
    FANOUT_FIELD_NUMBER: _ClassVar[int]
    FILESIZE_FIELD_NUMBER: _ClassVar[int]
    File: Data.DataType
    HAMTShard: Data.DataType
    HASHTYPE_FIELD_NUMBER: _ClassVar[int]
    Metadata: Data.DataType
    Raw: Data.DataType
    Symlink: Data.DataType
    TYPE_FIELD_NUMBER: _ClassVar[int]
    Type: Data.DataType
    blocksizes: _containers.RepeatedScalarFieldContainer[int]
    fanout: int
    filesize: int
    hashType: int
    def __init__(self, Type: _Optional[_Union[Data.DataType, str]] = ..., Data: _Optional[bytes] = ..., filesize: _Optional[int] = ..., blocksizes: _Optional[_Iterable[int]] = ..., hashType: _Optional[int] = ..., fanout: _Optional[int] = ...) -> None: ...

class Metadata(_message.Message):
    __slots__ = ["MimeType"]
    MIMETYPE_FIELD_NUMBER: _ClassVar[int]
    MimeType: str
    def __init__(self, MimeType: _Optional[str] = ...) -> None: ...
