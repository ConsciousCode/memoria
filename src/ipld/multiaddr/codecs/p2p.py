from src.ipld.multibase import base58

SIZE = None
IS_PATH = False

def to_bytes(string: str):
    # the address is a base58-encoded string
    mm = base58.decode(string)
    if len(mm) < 5:
        raise ValueError("P2P MultiHash too short: len() < 5")
    return mm


def to_string(buf: bytes):
    return base58.encode(buf)
