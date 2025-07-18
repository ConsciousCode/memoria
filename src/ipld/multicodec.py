'''
Multicodec is a protocol for identifying data formats and protocols.
This module provides utilities to work with multicodec prefixes,
including extracting, adding, and removing prefixes, as well as checking codec validity.
'''

from . import varint
from .multihash import HASH_CODES

__all__ = (
    'extract_prefix', 'get_prefix', 'add_prefix', 'remove_prefix',
    'get_codec', 'is_codec'
)

CODECS = {
    # multiformat
    'multicodec': 0x30,
    'multihash': 0x31,
    'multiaddr': 0x32,
    'multibase': 0x33,

    # multihash
    **HASH_CODES,

    # multiaddr
    'ip4': 0x04,
    'tcp': 0x06,
    'dccp': 0x21,
    'ip6': 0x29,
    'ip6zone': 0x2a,
    'dns': 0x35,
    'dns4': 0x36,
    'dns6': 0x37,
    'dnsaddr': 0x38,
    'sctp': 0x84,
    'udp': 0x0111,
    'p2p-webrtc-star': 0x0113,
    'p2p-webrtc-direct': 0x0114,
    'p2p-stardust': 0x0115,
    'p2p-circuit': 0x0122,
    'udt': 0x012d,
    'utp': 0x012e,
    'unix': 0x0190,
    'p2p': 0x01a5,
    'https': 0x01bb,
    'onion': 0x01bc,
    'onion3': 0x01bd,
    'garlic64': 0x01be,
    'garlic32': 0x01bf,
    'quic': 0x01cc,
    'ws': 0x01dd,
    'wss': 0x01de,
    'p2p-websocket-star': 0x01df,
    'http': 0x01e0,

    # ipld
    'raw': 0x55,
    'dag-pb': 0x70,
    'dag-cbor': 0x71,
    'libp2p-key': 0x72,
    'git-raw': 0x78,
    'torrent-info': 0x7b,
    'torrent-file': 0x7c,
    'leofcoin-block': 0x81,
    'leofcoin-tx': 0x82,
    'leofcoin-pr': 0x83,
    'eth-block': 0x90,
    'eth-block-list': 0x91,
    'eth-tx-trie': 0x92,
    'eth-tx': 0x93,
    'eth-tx-receipt-trie': 0x94,
    'eth-tx-receipt': 0x95,
    'eth-state-trie': 0x96,
    'eth-account-snapshot': 0x97,
    'eth-storage-trie': 0x98,
    'bitcoin-block': 0xb0,
    'bitcoin-tx': 0xb1,
    'zcash-block': 0xc0,
    'zcash-tx': 0xc1,
    'stellar-block': 0xd0,
    'stellar-tx': 0xd1,
    'decred-block': 0xe0,
    'decred-tx': 0xe1,
    'dash-block': 0xf0,
    'dash-tx': 0xf1,
    'swarm-manifest': 0xfa,
    'swarm-feed': 0xfb,
    'dag-json': 0x0129,

    # namespace
    'path': 0x2f,
    'ipld-ns': 0xe2,
    'ipfs-ns': 0xe3,
    'swarm-ns': 0xe4,
    'ipns-ns': 0xe5,
    'zeronet': 0xe6,

    # key
    'ed25519-pub': 0xed,

    # holochain
    'holochain-adr-v0': 0x807124,
    'holochain-adr-v1': 0x817124,
    'holochain-key-v0': 0x947124,
    'holochain-key-v1': 0x957124,
    'holochain-sig-v0': 0xa27124,
    'holochain-sig-v1': 0xa37124,
}
NAMES = {v: n for n, v in CODECS.items()}

def extract_prefix(bs: bytes) -> int:
    """Extracts the prefix from multicodec prefixed data."""
    try:
        return varint.decode_bytes(bs)
    except TypeError:
        raise ValueError('incorrect varint provided')

def get_prefix(multicodec: str):
    """Returns prefix for a given multicodec."""
    try:
        return varint.encode(CODECS[multicodec])
    except KeyError:
        raise ValueError('{} multicodec is not supported.'.format(multicodec))

def add_prefix(multicodec: str, bs: bytes) -> bytes:
    """Adds multicodec prefix to the given bytes input."""
    return b''.join([get_prefix(multicodec), bs])

def remove_prefix(bs: bytes) -> bytes:
    """Removes prefix from a prefixed data."""
    return bs[len(varint.encode(extract_prefix(bs))):]

def get_codec(bs: bytes) -> str:
    """Gets the codec used for prefix the multicodec prefixed data."""
    prefix = extract_prefix(bs)
    try: return NAMES[prefix]
    except KeyError:
        raise ValueError(f'Prefix {prefix} not present in the lookup table') from None

def split_codec(bs: bytes) -> tuple[str, bytes]:
    """Splits the multicodec prefixed data into codec and data."""
    prefix = extract_prefix(bs)
    try: codec = NAMES[prefix]
    except KeyError:
        raise ValueError(f'Prefix {prefix} not present in the lookup table') from None
    return codec, bs[len(varint.encode(prefix)):]

def is_codec(name: str) -> bool:
    """Check if the codec is a valid codec or not"""
    return name in CODECS
