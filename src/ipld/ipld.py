from typing import Iterable

from . import dagcbor, dagjson, dagpb
from ._common import IPLData
from .cid import CID

def iter_python_links(data: IPLData) -> Iterable[CID]:
    match data:
        case CID():
            yield data
        case list():
            for x in data:
                yield from iter_python_links(x)
        case dict():
            for x in data.values():
                yield from iter_python_links(x)

def iter_links(cid: CID, block: bytes) -> Iterable[CID]:
    match cid.codec:
        case "dag-pb":
            node = dagpb.unmarshal(block)
            for link in node.Links:
                yield CID(link.Hash)
        
        case "dag-cbor":
            node = dagcbor.unmarshal(block)
            yield from iter_python_links(node)
        
        case "dag-json":
            node = dagjson.unmarshal(block.decode('utf-8'))
            yield from iter_python_links(node)
        
        case cc:
            raise NotImplementedError(f"Unknown codec {cc}")