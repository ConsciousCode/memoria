from typing import Iterable

import varint

from .multihash import multihash
from .cid import CIDv1
from .ipld import dagcbor_marshal

def _car_block(cid: CIDv1, data: bytes) -> bytes:
    return varint.encode(len(cid) + len(data)) + cid.buffer + data

def _carv1_iter(blocks: Iterable[tuple[CIDv1, bytes]]) -> Iterable[bytes]:
    header = dagcbor_marshal({
        "version": 1,
        "roots": [cid for cid, _ in blocks]
    })
    header_cid = CIDv1("dag-cbor",
        multihash("sha2-256").update(header).digest().buffer
    )
    
    yield _car_block(header_cid, header)

    for cid, data in blocks:
        yield _car_block(cid, data)

def carv1(blocks: Iterable[tuple[CIDv1, bytes]]) -> bytes:
    """
    Create a CARv1 file from the given root CID and blocks.

    Args:
        blocks (Iterable[tuple[CIDv1, bytes]]): An iterable of tuples containing CIDs and their corresponding data.

    Returns:
        bytes: The CARv1 file as bytes.
    """
    return b''.join(_carv1_iter(blocks))