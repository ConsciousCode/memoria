import idna

SIZE = None
IS_PATH = False

def to_bytes(string: str):
    return idna.encode(string, uts46=True)

def to_string(buf: bytes):
    return idna.decode(buf)
