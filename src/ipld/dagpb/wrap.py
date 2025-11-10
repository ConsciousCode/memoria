from collections.abc import Iterable

from .._common import FauxMapping
from .dagpb_pb2 import PBLink as RawPBLink, PBNode as RawPBNode

class PBLink(FauxMapping):
    """A dag-pb link."""
    __match_args__: tuple[str, ...] = ("Hash", "Name", "Tsize")

    def __init__(self, Hash: bytes, Name: str, Tsize: int):
        self.Hash: bytes = Hash
        self.Name: str = Name
        self.Tsize: int = Tsize

    @classmethod
    def parse(cls, data: bytes):
        link = RawPBLink()
        _ = link.ParseFromString(data)
        return cls(link.Hash, link.Name, link.Tsize)

class _PBNodeLinks:
    '''Proxy class for PBNode's Links field to avoid multiple iterations.'''
    def __init__(self, links: Iterable[RawPBLink]):
        self._links: Iterable[RawPBLink] = links
    
    def __iter__(self):
        for link in self._links:
            yield PBLink(Hash=link.Hash, Name=link.Name, Tsize=link.Tsize)

class PBNode(FauxMapping):
    """A dag-pb node."""
    __match_args__: tuple[str, ...] = ("Data", "Links")

    def __init__(self, Data: bytes, Links: Iterable[PBLink], bytesize: int | None=None):
        self.Data: bytes = Data
        self.Links: Iterable[PBLink] = Links
        self.bytesize: int | None = bytesize

    @classmethod
    def unmarshal(cls, data: bytes) -> 'PBNode':
        node = RawPBNode()
        _ = node.ParseFromString(data)
        return cls(node.Data, _PBNodeLinks(node.Links), len(data))
    
    def marshal(self):
        return _from_wrap(self).SerializeToString()

    def ByteSize(self):
        if self.bytesize is not None:
            return self.bytesize
        bs = _from_wrap(self).ByteSize()
        self.bytesize = bs
        return bs

def _from_wrap(node: PBNode) -> RawPBNode:
    """Convert a PBNode to its protobuf representation."""
    return RawPBNode(
        Data=node.Data,
        Links=(
            RawPBLink(link.Hash, link.Name, link.Tsize)
                for link in node.Links
        )
    )
