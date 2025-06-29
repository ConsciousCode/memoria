SIZE = None
IS_PATH = False

def to_bytes(string: str):
    if len(string) == 0:
        raise ValueError("value must not be empty")
    return string.encode('utf-8')

def to_string(buf: bytes):
    if len(buf) == 0:
        raise ValueError("invalid length (should be > 0)")
    return buf.decode('utf-8')
