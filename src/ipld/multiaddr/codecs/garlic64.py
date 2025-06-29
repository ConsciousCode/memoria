import base64

SIZE = None
IS_PATH = False

def validate(buf: bytes) -> bytes:
    """Validate a garlic64 address."""
    if len(buf) < 386:
        raise ValueError(f"Failed to validate garlic_addr: {buf} is not an i2p base64 address. Length: {len(buf)}")
    return buf

def to_bytes(string: str) -> bytes:
    """Convert a garlic64 address to bytes."""
    try: garlic_host = base64.b64decode(string)
    except Exception as e:
        raise ValueError(f"Cannot decode {string!r} as base64: {e}") from e

    return validate(garlic_host)

def to_string(buf: bytes) -> str:
    """Convert bytes to a garlic64 address string."""
    return base64.b64encode(validate(buf)).decode('ascii').rstrip('=')