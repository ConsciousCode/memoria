from abc import ABC, abstractmethod
from functools import cached_property
from typing import IO, Callable, Iterable, Literal, Optional, override
import os

from . import ipld

from .dagpb import PBLink, PBNode
from .unixfs import Data
from .multihash import multihash
from .cid import CID, CIDv0, CIDv1, Codec
from .car import CARv2Index, carv1_iter, carv2_iter

class CIDResolveError(Exception):
    """Raised when a CID cannot be resolved to a node."""
    def __init__(self, cid: CID):
        super().__init__(str(cid))

class RawBlockLink:
    def __init__(self, cid: CID, filesize: int):
        self.cid = cid
        self.size = filesize
    
    def ByteSize(self):
        return self.size

class DAGPBNode(PBNode):
    size: int

    @cached_property
    def cid(self):
        return CIDv0(multihash("sha2-256", self.SerializeToString()))

class FileLeaf(DAGPBNode):
    '''Leaf of an IPFS UnixFS file.'''
    def __init__(self, data: bytes):
        super().__init__(
            Links=[],
            Data=Data(Type=Data.File, Data=data).SerializeToString()
        )
        self.size = len(data)

class FileNode(DAGPBNode):
    '''Root of an IPFS UnixFS file, containing multiple leaves.'''

    def __init__(self, nodes: Iterable[RawBlockLink|DAGPBNode]):
        size = 0
        blocksizes: list[int] = []
        links: list[PBLink] = []
        for node in nodes:
            size += node.size
            blocksizes.append(node.size)
            links.append(PBLink(
                Name="",
                Hash=node.cid.buffer,
                Tsize=node.ByteSize()
            ))
        
        super().__init__(
            Links=links,
            Data=Data(
                Type=Data.File,
                blocksizes=blocksizes,
                filesize=size
            ).SerializeToString()
        )
        self.size = size

class Blocksource(ABC):
    @abstractmethod
    def dag_has(self, cid: CID) -> bool:
        """Check if the node has a local copy of the block."""
        pass

    @abstractmethod
    def dag_get(self, cid: CID) -> Optional[bytes]:
        """Retrieve a block from the IPFS DAG by its CID."""
        pass

    def dag_export(self,
            cid: CID,
            *,
            version: Literal[1, 2] = 1,
            index: Optional[CARv2Index] = None,
            fully_indexed: bool = True
        ) -> Iterable[bytes]:
        """Export the data of a FileNode as bytes using the indicated CAR format."""

        # Iterate over *whole blocks* rather than just the data as in dag_cat
        def iter_blocks(cid: CID):
            if (block := self.dag_get(cid)) is None:
                raise CIDResolveError(cid)
            
            yield block
            for link in ipld.iter_links(cid, block):
                yield from iter_blocks(link)
        
        it = iter_blocks(cid)

        if version == 1:
            return carv1_iter(it)
        elif version == 2:
            return carv2_iter(
                it,
                index=index,
                fully_indexed=fully_indexed
            )
        else:
            raise ValueError("Unsupported CAR version, must be 1 or 2")
    
    def ipfs_cat(self, cid: CID) -> Iterable[bytes]:
        """
        Concatenate the data of a FileNode from its CID. This yields whole
        blocks of data, not arbitrary chunks.
        """
        node_data = self.dag_get(cid)
        if node_data is None:
            raise CIDResolveError(cid)
        
        node = DAGPBNode()
        node.ParseFromString(node_data)
        
        if node.HasField('Data'):
            yield node.Data
        else:
            for link in node.Links:
                yield from self.ipfs_cat(CIDv0(link.Hash))

type ChunkingStrategy = Callable[[IO[bytes]], Iterable[bytes]]
"""Strategy for chunking a stream into smaller parts."""

def chunker_size(size: int=256*1024):
    def chunking_strategy(stream: IO[bytes]) -> Iterable[bytes]:
        """Chunk the stream into fixed-size chunks."""
        while chunk := stream.read(size):
            yield chunk
    return chunking_strategy

type ShardingStrategy = Callable[[CID], str]
"""Determine the sharded path for a given CID."""

def shard_last2(cid: CID) -> str:
    """
    Sharding strategy that uses the last two characters of the CID's hex
    representation to determine the path.
    """
    enc = str(cid)
    return f"{enc[-2:]}/{enc}"

class Blockstore(Blocksource):
    @abstractmethod
    def dag_put(self, block: bytes, *, codec: Codec='dag-cbor', function: str='sha2-256') -> CID:
        """Store a block in the IPFS DAG and return its CID."""
        pass
    
    def ipfs_put(self, data: bytes) -> RawBlockLink|DAGPBNode:
        """
        Given a raw block, store it in the IPFS DAG and return its CID.
        This allows subclasses to override the behavior of how blocks are
        stored in IPFS. Use raw-leaves by default.
        """
        return RawBlockLink(self.dag_put(data, codec='dag-pb'), len(data))

    def ipfs_add(self, stream: IO[bytes], *, chunker: ChunkingStrategy=chunker_size()) -> CID:
        """Add a file to the IPFS DAG from a stream."""
        
        return self.dag_put(FileNode(
            map(self.ipfs_put, chunker(stream))
        ).SerializeToString())

class FlatfsBlockstore(Blockstore):
    def __init__(self, root: str, sharding: ShardingStrategy = shard_last2):
        self.root = root
        self.sharding = sharding
    
    def build_path(self, cid: CID) -> str:
        """Build the full path for a CID in the flatfs."""
        return os.path.join(self.root, self.sharding(cid))
    
    @override
    def dag_has(self, cid: CID) -> bool:
        """Check if the block with the given CID exists in the flatfs."""
        return os.path.exists(self.build_path(cid))

    @override
    def dag_put(self, block: bytes, *, codec: Codec='dag-cbor', function: str='sha2-256') -> CID:
        """Store a block in the flatfs and return its CID."""
        mh = multihash(function, block)
        cid = CIDv0(mh) if codec == 'dag-pb' else CIDv1(codec, mh)
        path = self.build_path(cid)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(block)
        return cid

    @override
    def dag_get(self, cid: CID) -> Optional[bytes]:
        """Retrieve a block from the flatfs by its CID."""
        try:
            with open(self.build_path(cid), "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

class CompositeBlocksource(Blocksource):
    def __init__(self, *sources: Blocksource):
        self.sources = sources
    
    @override
    def dag_has(self, cid: CID) -> bool:
        """Check if any source has the block."""
        return any(src.dag_has(cid) for src in self.sources)

    @override
    def dag_get(self, cid: CID) -> Optional[bytes]:
        """Retrieve a block from the first source that has it."""
        for source in self.sources:
            if (block := source.dag_get(cid)) is not None:
                return block
        return None