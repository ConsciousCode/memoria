from .._common import IPLData
from .dagpb_pb2 import PBLink, PBNode

__all__ = (
    'PBLink',
    'PBNode',
    'marshal',
    'unmarshal'
)

def marshal(data: PBNode) -> bytes:
    return data.SerializeToString()

def unmarshal(data: bytes) -> PBNode:
    node = PBNode()
    node.ParseFromString(data)
    return node