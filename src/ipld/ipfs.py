from abc import ABC, abstractmethod
from functools import cached_property
from typing import IO, Callable, Iterable, Literal, Optional, override
import os

from . import ipld, multibase, dagpb

from ._common import IPLData
from .unixfs import Data
from .multihash import multihash
from .cid import CID, CIDv0, BlockCodec
from .car import CARv2Index, carv1_iter, carv2_iter

__all__ = (
    'CIDResolveError', 'RawBlockLink', 'DAGPBNode', 'FileLeaf', 'FileNode',
    'Blocksource', 'Blockstore',
    'FlatfsBlockstore', 'CompositeBlocksource',
    'ShardingStrategy', 'ShardLast',
    'ChunkingStrategy', 'chunker_size'
)

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

class DAGPBNode(dagpb.PBNode):
    size: int

    @cached_property
    def cid(self):
        return CIDv0(multihash("sha2-256", self.dump()))

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
        links: list[dagpb.PBLink] = []
        for node in nodes:
            size += node.size
            blocksizes.append(node.size)
            links.append(dagpb.PBLink(
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
    def block_has(self, cid: CID) -> bool:
        """Check if the node has a local copy of the block."""
        pass

    @abstractmethod
    def block_get(self, cid: CID) -> Optional[bytes]:
        """Retrieve a block from the IPFS DAG by its CID."""
        pass

    def dag_get(self, cid: CID) -> IPLData:
        """Retrieve a DAG node by its CID and return its IPLD model."""
        if (block := self.block_get(cid)) is None:
            raise CIDResolveError(cid)
        return ipld.dag_load(cid.codec, block)

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
            if (block := self.block_get(cid)) is None:
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
    
    def ipfs_cat(self,
            cid: CID,
            offset: int=0,
            length: Optional[int]=None
        ) -> Iterable[bytes]:
        """
        Concatenate the data of a FileNode from its CID. This yields whole
        blocks of data, not arbitrary chunks.
        """
        if length is not None and length <= 0:
            return

        if (node_data := self.block_get(cid)) is None:
            raise CIDResolveError(cid)
        
        node = dagpb.unmarshal(node_data)
        data = Data()
        data.ParseFromString(node.Data)
        
        if node.Links:
            for size, link in zip(data.blocksizes, node.Links):
                # Skip until we reach the offset
                if offset >= size:
                    offset -= size
                    continue
                
                yield from self.ipfs_cat(CID(link.Hash), offset, length)
                
                # No more offset from this point
                offset = 0

                # Length upkeep
                if length is not None:
                    length -= size
                    if length <= 0:
                        break # Early break when we're done
        else:
            yield node.Data[offset:][:length]

type ChunkingStrategy = Callable[[IO[bytes]], Iterable[bytes]]
"""Strategy for chunking a stream into smaller parts."""

def chunker_size(size: int=256*1024):
    def chunking_strategy(stream: IO[bytes]) -> Iterable[bytes]:
        """Chunk the stream into fixed-size chunks."""
        while chunk := stream.read(size):
            yield chunk
    return chunking_strategy

class ShardingStrategy(ABC):
    @abstractmethod
    def name(self) -> str:
        """Return the name of the sharding strategy."""

    @abstractmethod
    def shard(self, cid: CID) -> str:
        """Determine the sharded path for a given CID."""

class ShardLast(ShardingStrategy):
    """
    Sharding strategy that uses the next-to-last N characters of the CID's
    representation to determine the path.
    """
    def __init__(self, last: int = 2):
        self.last = last
    
    def name(self) -> str:
        return f"/next-to-last/{self.last}"
    
    def shard(self, cid: CID) -> str:
        # Since Kubo v0.12 only the multihash is sharded
        # Since v0.35 base32upper is used without a multibase prefix
        enc = multibase.base32upper.encode(cid.multihash.buffer)
        # [:-1] because next-to-last, not last (better entropy in base32)
        return f"{enc[:-1][-self.last:]}/{enc}.data"

class Blockstore(Blocksource):
    @abstractmethod
    def block_put(self, block: bytes, *,
            cid_version: Literal[0, 1]=1,
            codec: BlockCodec='dag-cbor',
            function: str='sha2-256'
        ) -> CID:
        """Store a block in the IPFS DAG and return its CID."""
        pass
    
    def ipfs_put(self, data: bytes, *,
            raw_leaves: bool=True,
            function: str='sha2-256'
        ) -> RawBlockLink|DAGPBNode:
        """
        Given a raw block, store it in the IPFS DAG and return its block.
        This allows subclasses to override the behavior of how blocks are
        stored in IPFS. Uses raw-leaves by default.
        """
        if raw_leaves:
            return RawBlockLink(self.block_put(
                data,
                cid_version=1, # raw codec is not possible with CIDv0
                codec='raw',
                function=function
            ), len(data))
        else:
            return FileLeaf(data)

    def ipfs_add(self, stream: IO[bytes], *,
            raw_leaves: bool=True,
            chunker: ChunkingStrategy=chunker_size(),
            cid_version: Literal[0, 1]=0,
            codec: BlockCodec='dag-pb',
            function: str='sha2-256'
        ) -> CID:
        """Add a file to the IPFS DAG from a stream."""
        
        blocks = [
            self.ipfs_put(
                chunk,
                raw_leaves=raw_leaves,
                function=function
            ) for chunk in chunker(stream)
        ]
        if raw_leaves and len(blocks) == 1:
            # If there's only one block, return it directly
            return blocks[0].cid

        return self.block_put(
            FileNode(blocks).dump(),
            cid_version=cid_version,
            codec=codec,
            function=function
        )

class FlatfsBlockstore(Blockstore):
    def __init__(self, root: str, sharding: ShardingStrategy = ShardLast(2)):
        self.root = root
        self.sharding = sharding
    
    def build_path(self, cid: CID) -> str:
        """Build the full path for a CID in the flatfs."""
        print("Path", os.path.join(self.root, self.sharding.shard(cid)))
        return os.path.join(self.root, self.sharding.shard(cid))
    
    @override
    def block_has(self, cid: CID) -> bool:
        """Check if the block with the given CID exists in the flatfs."""
        return os.path.exists(self.build_path(cid))

    @override
    def block_put(self, block: bytes, *, 
            cid_version: Literal[0, 1]=1,
            codec: BlockCodec='dag-cbor',
            function: str='sha2-256'
        ) -> CID:
        """Store a block in the flatfs and return its CID."""
        cid = CID(cid_version, codec, multihash(function, block))
        path = self.build_path(cid)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(block)
        return cid

    @override
    def block_get(self, cid: CID) -> Optional[bytes]:
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
    def block_has(self, cid: CID) -> bool:
        """Check if any source has the block."""
        return any(src.block_has(cid) for src in self.sources)

    @override
    def block_get(self, cid: CID) -> Optional[bytes]:
        """Retrieve a block from the first source that has it."""
        for source in self.sources:
            if (block := source.block_get(cid)) is not None:
                return block
        return None