from typing import Iterable, Optional
from . import dagpb_pb2

class FauxMapping:
    """A faux mapping class to act as a dictionary for __match_args__."""
    __match_args__: tuple[str, ...]

    def __getitem__(self, key: str):
        if key in self.__match_args__:
            return getattr(self, key)
        raise KeyError(key)
    
    def keys(self):
        yield from self.__match_args__
    
    def values(self):
        for key in self.__match_args__:
            yield getattr(self, key)
    
    def items(self):
        for key in self.__match_args__:
            yield key, getattr(self, key)

class PBLink(FauxMapping):
    """A dag-pb link."""
    __match_args__ = ("Hash", "Name", "Tsize")

    def __init__(self, Hash: bytes, Name: str, Tsize: int):
        self.Hash = Hash
        self.Name = Name
        self.Tsize = Tsize

    @classmethod
    def parse(cls, data: bytes):
        link = dagpb_pb2.PBLink()
        link.ParseFromString(data)
        return cls(link.Hash, link.Name, link.Tsize)

class _PBNodeLinks:
    '''Proxy class for PBNode's Links field to avoid multiple iterations.'''
    def __init__(self, links: Iterable[dagpb_pb2.PBLink]):
        self._links = links
    
    def __iter__(self):
        for link in self._links:
            yield PBLink(link.Hash, link.Name, link.Tsize)

class PBNode(FauxMapping):
    """A dag-pb node."""
    __match_args__ = ("Data", "Links")

    def __init__(self, Data: bytes, Links: Iterable[PBLink], bytesize: Optional[int]=None):
        self.Data = Data
        self.Links = Links
        self.bytesize = bytesize

    @classmethod
    def load(cls, data: bytes) -> 'PBNode':
        node = dagpb_pb2.PBNode()
        node.ParseFromString(data)
        return cls(node.Data, _PBNodeLinks(node.Links), len(data))
    
    def dump(self):
        return _from_wrap(self).SerializeToString()

    def ByteSize(self):
        if self.bytesize is not None:
            return self.bytesize
        bs = _from_wrap(self).ByteSize()
        self.bytesize = bs
        return bs

def _from_wrap(node: PBNode) -> dagpb_pb2.PBNode:
    """Convert a PBNode to its protobuf representation."""
    return dagpb_pb2.PBNode(
        Data=node.Data,
        Links=(
            dagpb_pb2.PBLink(link.Hash, link.Name, link.Tsize)
            for link in node.Links
        )
    )
    