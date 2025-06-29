SIZE = 16
IS_PATH = False

def to_bytes(string: str):
    try:
        return int(string, 10).to_bytes(2, byteorder='big')
    except ValueError as e:
        raise ValueError("Not a base 10 integer") from e

def to_string(buf: bytes):
    if len(buf) != 2:
        raise ValueError("Invalid integer length (must be 2 bytes / 16 bits)")
    return str(int.from_bytes(buf))
