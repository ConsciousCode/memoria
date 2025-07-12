from typing import Iterable
import json
import cbor2

from . import dagcbor, dagjson, dagpb
from ._common import IPLData
from .cid import CID, BlockCodec

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

def dag_load(codec: BlockCodec, block: bytes) -> IPLData:
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

def dag_dump(codec: BlockCodec, node: IPLData) -> str|bytes:
    """
    Serialize an IPLD node into bytes based on the codec.
    If a CID is provided, it will be used to determine the codec.
    """
    
    match codec:
        case 'dag-pb': return dagpb.marshal(node)
        case 'dag-cbor': return dagcbor.marshal(node)
        case 'dag-json': return dagjson.marshal(node)

        case 'json':
            return json.dumps(node, indent=2).encode('utf-8')
        case 'cbor':
            return cbor2.dumps(node)

        case _:
            raise NotImplementedError(f"Unsupported codec: {codec}")