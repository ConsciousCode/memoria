from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterable, Iterator, Literal, Optional, override

import varint

from .multihash import multihash
from .cid import CIDv1
from . import dagcbor

__all__ = (
    'carv1_iter', 'carv2_iter', 'carv1', 'carv2',
)

INDEX_CODECS = {
    'car-index-sorted': 0x0400,
    'car-multihash-index-sorted': 0x0401,
}

type CARv2Index = Literal['car-index-sorted', 'car-multihash-index-sorted']

def _car_block(cid: CIDv1, data: bytes) -> Iterable[bytes]:
    yield varint.encode(len(cid) + len(data))
    yield cid.buffer
    yield data

def _build_cid(data: bytes) -> CIDv1:
    """Build a CIDv1 from the given data."""
    return CIDv1(
        "dag-cbor", multihash("sha2-256").update(data).digest().buffer
    )

class CARv1Indexer(ABC):
    def __init__(self):
        self.roots: list[CIDv1] = []
        self.blocks: list[bytes] = []

    def add(self, block: bytes) -> CIDv1:
        '''Add a block to the index.'''
        cid = _build_cid(block)
        self.roots.append(cid)
        self.blocks.append(block)
        return cid

    def __iter__(self) -> Iterator[tuple[CIDv1, bytes]]:
        '''Iterate over the contents of the CAR file.'''
        return zip(self.roots, self.blocks)
    
    def carv1_header(self) -> Iterable[bytes]:
        """Return the CARv1 header as bytes."""
        header = dagcbor.marshal({"version": 1, "roots": self.roots})
        yield from _car_block(_build_cid(header), header)
    
    def carv1(self) -> Iterable[bytes]:
        """Yield bytes for the CAR file."""
        
        yield from self.carv1_header()
        for cid, data in self:
            yield from _car_block(cid, data)

class CARv2Indexer(CARv1Indexer):
    '''
    All CARv2 indexers, even the no-index one, must keep track of the total
    size of the data they index, as the CARv2 header requires it.
    '''

    has_index = False

    def __init__(self):
        super().__init__()
        self.data_size = 0
    
    @override
    def add(self, block: bytes):
        cid = super().add(block)
        self.data_size += len(cid) + len(block)
        return cid
    
    carv1_size = 0

    @override
    def carv1_header(self) -> Iterable[bytes]:
        header = b''.join(super().carv1_header())
        self.header_size = len(header)
        yield header

    def index(self) -> Iterable[bytes]:
        if False: yield b'' # No index

class CARv2WithIndexer[K, V](CARv2Indexer):
    codec: int

    has_index = True

    def __init__(self):
        self.offsets: list[int] = []
        self.buckets: dict[K, list[V]] = defaultdict(list)
    
    @abstractmethod
    def add_bucket(self, cid: CIDv1):
        """Add a block to the appropriate bucket based on its CID length."""
    
    @property
    def carv1_size(self) -> int:
        """Size of the CARv1 section."""
        return self.header_size + self.data_size
    
    @override
    def add(self, block: bytes) -> CIDv1:
        cid = super().add(block)
        self.add_bucket(cid)
        self.offsets.append(self.data_size)
        return cid
    
    @override
    def index(self) -> Iterable[bytes]:
        yield varint.encode(self.codec)
    
    def render_bucket(self, width: int, indices: list[int]) -> Iterable[bytes]:
        """Render a bucket of indices."""
        yield width.to_bytes(4, 'little')  # width
        yield len(indices).to_bytes(8, 'little')  # count
        for idx in indices:
            yield self.roots[idx].multihash.digest  # spec says "digest"
            yield (self.header_size + self.offsets[idx]).to_bytes(8, 'little')  # offset from CIDv1

class CARv2SortedIndexer(CARv2WithIndexer[int, int]):
    codec = INDEX_CODECS['car-index-sorted']

    def add_bucket(self, cid: CIDv1):
        """Add a block to the sorted bucket based on its CID length."""
        self.buckets[len(cid)].append(len(self.offsets))
    
    @override
    def index(self) -> Iterable[bytes]:
        yield from super().index()
        for width, indices in sorted(self.buckets.items(), key=lambda k: k[0]):
            yield from self.render_bucket(width, indices)

class CARv2MultihashSortedIndexer(CARv2WithIndexer[tuple[int, int], int]):
    codec = INDEX_CODECS['car-multihash-index-sorted']

    def add_bucket(self, cid: CIDv1):
        """Add a block to the multihash sorted bucket based on its CID length."""
        mh = cid.multihash
        self.buckets[mh.function, len(cid)].append(len(self.offsets))
    
    @override
    def index(self) -> Iterable[bytes]:
        last_fn = 0
        yield from super().index()
        for (fn, width), indices in sorted(self.buckets.items(), key=lambda k: k[0]):
            if last_fn != fn:
                yield (last_fn := fn).to_bytes(8, 'little')
            yield from self.render_bucket(width, indices)

def carv1_iter(biter: Iterable[bytes]) -> Iterable[bytes]:
    indexer = CARv1Indexer()
    for block in biter:
        indexer.add(block)
    yield from indexer.carv1()

def carv2_iter(
        biter: Iterable[bytes],
        *,
        index: Optional[CARv2Index] = 'car-index-sorted',
        fully_indexed=True
    ) -> Iterable[bytes]:
    # [pragma] [header] ... [carv1] ... [index]
    # [header]: [feature bitfield] [data offset] [data size] [index offset]

    if fully_indexed and not index:
        raise ValueError("Cannot be fully indexed without an index")

    indexer: CARv2Indexer = {
        'car-index-sorted': CARv2SortedIndexer,
        'car-multihash-index-sorted': CARv2MultihashSortedIndexer,
        None: CARv1Indexer
    }[index]()
    for block in biter:
        indexer.add(block)

    ##[[ Header ]]##
    # [pragma]
    yield b"\xa1\x67\x76\x65\x72\x73\x69\x6f\x6e\x02"
    
    # [feature bitfield]
    yield (fully_indexed << 127).to_bytes(16, 'big')

    # [data offset]
    header_size = 11 + 16 + 8 + 8 + 8
    leading_padding = 0
    data_offset = header_size + leading_padding
    yield data_offset.to_bytes(8, 'little') # data offset
    
    # [data size]
    yield indexer.data_size.to_bytes(8, 'little')  # data size

    # [index offset]
    trailing_padding = 0
    if indexer.has_index:
        index_offset = data_offset + indexer.carv1_size + trailing_padding
    else:
        index_offset = 0
    yield index_offset.to_bytes(8, 'little')

    ##[[ Leading padding ]]##
    if leading_padding:
        yield bytes(leading_padding)

    ##[[ CIDv1 section ]]##
    yield from indexer.carv1()

    ##[[ Trailing padding ]]##
    if trailing_padding:
        yield bytes(trailing_padding)
    
    ##[[ Index section ]]##
    yield from indexer.index()

def carv1(blocks: Iterable[bytes]) -> bytes:
    """Create a CARv1 file from the given blocks."""
    return b''.join(carv1_iter(blocks))

def carv2(
        blocks: Iterable[bytes],
        *,
        index: Optional[CARv2Index] = 'car-index-sorted',
        fully_indexed=True
    ) -> bytes:
    """Create a CARv2 file from the given blocks."""
    return b''.join(carv2_iter(blocks, index=index, fully_indexed=fully_indexed))