'''
IPLD DAG-PB (Protocol Buffers) module. In IPLD this is solely used for
representing unixfs data structures.
'''

from typing import TypedDict
from ipld._common import IPLData
from ipld.cid import CID

from .wrap import PBNode, PBLink

class PBLinkModel(TypedDict):
    """An IPLD model for a DAG-PB link."""
    Name: str
    Hash: CID
    Tsize: int

class PBNodeModel(TypedDict):
    """An IPLD model for a DAG-PB node."""
    Links: list[PBLinkModel]
    Data: bytes

def _encode_pblink(link) -> PBLink:
    match link:
        case dict({"Name": str(name), "Hash": hash, "Size": int(size)}):
            return PBLink(
                Name=name,
                Hash=CID(hash).buffer,
                Tsize=size
            )
    raise ValueError(f"Invalid PBLink: {link}")

def _encode_pbnode(node) -> PBNode:
    match node:
        case dict({"Links": list(links), "Data": bytes(data)}):
            if nk := node.keys() - {"Links", "Data"}:
                raise ValueError(f"PBNode contains unrecognized keys: {nk}")
            return PBNode(
                Links=map(_encode_pblink, links),
                Data=data
            )
    raise ValueError(f"Invalid PBNode: {node}")

def marshal(data: PBNode|IPLData) -> bytes:
    '''Marshal DAG-PB data to bytes.'''
    if not isinstance(data, PBNode):
        data = _encode_pbnode(data)
    return data.dump()

def unmarshal(data: bytes) -> PBNode:
    '''Unmarshal bytes to DAG-PB data.'''
    return PBNode.load(data)