from typing import IO, Iterable

def encode_iter(number: int) -> Iterable[int]:
    """Pack `number` into varint bytes as an iterable of integers"""
    while True:
        towrite = number & 0x7f
        number >>= 7
        if number:
            yield towrite | 0x80
        else:
            yield towrite
            break

def decode_iter(it: Iterable[int]) -> int:
    """Read a varint from `buf` bytes as an iterable of bytes"""
    shift = 0
    result = 0
    for i in it:
        result |= (i & 0x7f) << shift
        shift += 7
        if not (i & 0x80):
            return result

    raise ValueError("Incomplete varint data")

def stream_bytes(stream: IO[bytes]) -> Iterable[int]:
    """Read bytes from `stream` as an iterable of integers"""
    while b := stream.read(1):
        yield b[0]

def encode(number: int) -> bytes:
    """Pack `number` into varint bytes"""
    return bytes(encode_iter(number))

def decode_stream(stream: IO[bytes]) -> int:
    """Read a varint from `stream`"""
    return decode_iter(stream_bytes(stream))

def decode_bytes(buf: bytes) -> int:
    """Read a varint from from `buf` bytes"""
    return decode_iter(buf)