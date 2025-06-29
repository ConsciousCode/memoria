import netaddr

SIZE = 0
IS_PATH = False

def to_bytes(string: str):
    return string.encode('utf-8')

def to_string(buf: bytes):
    return buf.decode('utf-8')