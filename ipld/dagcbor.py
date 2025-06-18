import cbor2

from .cid import CID
from .ipld import _encodec, _decodec, IPLData

__all__ = (
    "marshal", "unmarshal"
)

LINK_TAG = 42
'''DAG-CBOR tag for links.'''

@_encodec("DAG-CBOR")
def _dagcbor_encode(data: IPLData):
    match data:
        case bytes(): return data
        case CID(): return cbor2.CBORTag(LINK_TAG, data.buffer)

@_decodec("DAG-CBOR")
def _dagcbor_decode(data) -> IPLData:
    match data:
        case cbor2.CBORTag():
            if data.tag == LINK_TAG:
                return CID(data.value)
            raise ValueError(f'DAG-CBOR forbids all tags except {LINK_TAG} (CID). Got {data.tag}')

def marshal(data: IPLData) -> bytes:
    """
    Convert data to DAG-CBOR format.

    Args:
        data: The data to convert.

    Returns:
        The data in DAG-CBOR format.
    """
    return cbor2.dumps(_dagcbor_encode(data), canonical=True)

def unmarshal(data: bytes) -> IPLData:
    """
    Convert DAG-CBOR format data back to its original form.

    Args:
        data: The DAG-CBOR formatted bytes.

    Returns:
        The original data.
    """
    return _dagcbor_decode(cbor2.loads(data))