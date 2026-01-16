'''
Specification is scattered across:
- https://github.com/ipfs/specs/blob/main/UNIXFS.md
- https://github.com/ipfs/go-unixfs/blob/master/unixfs.go (archived)
- https://github.com/ipfs/boxo/blob/main/ipld/unixfs/unixfs.go
'''

from typing import Optional

from pydantic import BaseModel

from . import unixfs_pb2

class UnixTime(BaseModel):
    Seconds: int
    FractionalNanoseconds: Optional[int] = None

class BaseData(BaseModel):
    '''All UnixFS data supports these properties.'''
    mode: Optional[int] = None
    mtime: Optional[UnixTime] = None

class RawData(BaseData):
    '''UnixFS backwards compatibility with raw-leaves before CIDv1.'''
    data: bytes
    filesize: int

    def marshal(self) -> bytes:
        return unixfs_pb2.Data(
            Type=unixfs_pb2.Data.Raw,
            Data=self.data,
            filesize=self.filesize
        ).SerializeToString()

class DirectoryData(BaseData):
    '''UnixFS directory data.'''
    def marshal(self) -> bytes:
        return unixfs_pb2.Data(
            Type=unixfs_pb2.Data.Directory
        ).SerializeToString()

class SmallFileData(BaseData):
    '''Root UnixFS file data. Small files embed the data directly.'''
    data: bytes

    def marshal(self) -> bytes:
        return unixfs_pb2.Data(
            Type=unixfs_pb2.Data.File,
            Data=self.data,
            filesize=len(self.data)
        ).SerializeToString()

class BigFileData(BaseData):
    '''Root UnixFS file data. Big files use a block structure.'''
    filesize: int
    blocksizes: list[int]

    def marshal(self) -> bytes:
        return unixfs_pb2.Data(
            Type=unixfs_pb2.Data.File,
            filesize=self.filesize,
            blocksizes=self.blocksizes
        ).SerializeToString()

class MetadataData(BaseData):
    '''
    UnixFS metadata data. This is used for MIME types and other metadata but
    was never actually specified anywhere so who knows if it's ever been used.
    '''
    mimeType: Optional[str] = None

    def marshal(self) -> bytes:
        md = unixfs_pb2.Metadata(MimeType=self.mimeType)
        return unixfs_pb2.Data(
            Type=unixfs_pb2.Data.Metadata,
            Data=md.SerializeToString()
        ).SerializeToString()

class SymlinkData(BaseData):
    '''UnixFS symlink data. This is used for symlinks in UnixFS.'''
    path: str

    def marshal(self) -> bytes:
        return unixfs_pb2.Data(
            Type=unixfs_pb2.Data.Symlink,
            Data=self.path.encode('utf-8')
        ).SerializeToString()

class HAMTShardData(BaseData):
    '''
    Hash Array Mapped Trie (HAMT) shard data. Allows for efficient
    representation of large directories or files with many links sharded
    across multiple blocks.
    '''
    hashType: int
    bitfield: bytes
    fanout: int

    def marshal(self) -> bytes:
        return unixfs_pb2.Data(
            Type=unixfs_pb2.Data.HAMTShard,
            Data=self.bitfield,
            hashType=self.hashType,
            fanout=self.fanout
        ).SerializeToString()

type UnixFSData = (
    RawData | DirectoryData | SmallFileData | BigFileData |
    MetadataData | SymlinkData | HAMTShardData
)

def unmarshal(buf: bytes) -> UnixFSData:
    '''
    Unmarshal bytes into a UnixFS data structure.
    '''
    data = unixfs_pb2.Data()
    data.ParseFromString(buf)

    match data.Type:
        case unixfs_pb2.Data.Raw:
            return RawData(data=data.Data, filesize=data.filesize)
        
        case unixfs_pb2.Data.Directory:
            return DirectoryData()
        
        case unixfs_pb2.Data.File:
            # Switching on blocksizes lets us detect the empty small file
            if data.blocksizes:
                return BigFileData(
                    filesize=data.filesize, blocksizes=list(data.blocksizes)
                )
            else:
                if data.filesize != len(data.Data):
                    raise ValueError("Small file data size mismatch")
                return SmallFileData(data=data.Data)
        
        case unixfs_pb2.Data.Metadata:
            md = unixfs_pb2.Metadata()
            md.ParseFromString(data.Data)
            return MetadataData(mimeType=md.MimeType)
        
        case unixfs_pb2.Data.Symlink:
            return SymlinkData(path=data.Data.decode('utf-8'))
        
        case unixfs_pb2.Data.HAMTShard:
            return HAMTShardData(
                bitfield=data.Data, hashType=data.hashType, fanout=data.fanout
            )
        
        case _:
            raise ValueError(f"Unknown DataType: {data.Type}")