'''
Utilities for working with IPLD data structures without knowing the specific
codec used.
'''

from typing import TYPE_CHECKING, Iterable, Literal, overload
import json
import cbor2

if TYPE_CHECKING:
    from dagpb import PBNodeModel

from ._common import IPLData
import dagcbor, dagjson, dagpb
from cid import CID, BlockCodec

def links(data: IPLData) -> Iterable[CID]:
    """Extract all CIDs from an IPLD data structure."""
    match data:
        case CID():
            yield data
        case list():
            for x in data:
                yield from links(x)
        case dict():
            for x in data.values():
                yield from links(x)

@overload
def unmarshal(codec: Literal["dag-pb"], block: bytes) -> 'PBNodeModel': ...
@overload
def unmarshal(codec: BlockCodec, block: bytes) -> IPLData: ...

def unmarshal(codec: BlockCodec, block: bytes) -> IPLData:
    """
    Parse a block of data into its IPLD model based on the codec.
    If a CID is provided, it will be used to determine the codec.
    """
    match codec:
        case 'dag-pb':
            node = dagpb.unmarshal(block)
            return {
                'Links': [{
                    'Name': link.Name,
                    'Hash': CID(link.Hash),
                    'Tsize': link.Tsize
                } for link in node.Links],
                'Data': node.Data
            }
        case 'dag-cbor': return dagcbor.unmarshal(block)
        case 'dag-json': return dagjson.unmarshal(block)

        case 'json':
            return json.loads(block.decode('utf-8'))
        case 'cbor':
            return cbor2.loads(block)

        case _:
            raise NotImplementedError(f"Unsupported codec: {codec}")

@overload
def marshal(codec: Literal["dag-json", "json"], node: IPLData) -> str: ...
@overload
def marshal(codec: Literal["dag-cbor", "dag-pb", "cbor"], node: IPLData) -> bytes: ...
@overload
def marshal(codec: BlockCodec, node: IPLData) -> str|bytes: ...

def marshal(codec: BlockCodec, node: IPLData) -> str|bytes:
    """
    Serialize an IPLD node into bytes based on the codec.
    If a CID is provided, it will be used to determine the codec.
    """
    match codec:
        case 'dag-pb': return dagpb.marshal(node)
        case 'dag-cbor': return dagcbor.marshal(node)
        case 'dag-json': return dagjson.marshal(node)

        case 'json': return json.dumps(node, indent=2)
        case 'cbor': return cbor2.dumps(node)

        case _:
            raise NotImplementedError(f"Unsupported codec: {codec}")