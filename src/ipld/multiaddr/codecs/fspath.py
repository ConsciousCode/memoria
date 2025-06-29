import os

SIZE = None
IS_PATH = True

def to_bytes(string: str):
    return os.fsencode(string)

def to_string(buf: bytes):
    return os.fsdecode(buf)
