from .._common_ipld import IPLData
from ..cid import CID
from .wrap import PBNode, PBLink

__all__ = (
    'PBLink',
    'PBNode',
    'marshal',
    'unmarshal'
)

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
    if not isinstance(data, PBNode):
        data = _encode_pbnode(data)
    return data.dump()

def unmarshal(data: bytes) -> PBNode:
    return PBNode.load(data)