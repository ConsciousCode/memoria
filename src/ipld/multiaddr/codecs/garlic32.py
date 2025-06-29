import base64

SIZE = None
IS_PATH = False

def validate(buf: bytes) -> bytes:
    """Validate a garlic32 address."""
    bc = len(buf)
    if bc < 35 and bc != 32:
        raise ValueError(f"Failed to validate garlic_addr: {buf} is not an i2p base32 address. Length: {len(buf)}")
    return buf

def to_bytes(string: str) -> bytes:
    """Convert a garlic32 address to bytes."""
    string += '=' * (8 - len(string) % 8)  # Pad with '=' to multiple of 8
    
    try: garlic_host = base64.b32decode(string, casefold=True)
    except Exception as e:
        raise ValueError(f"Cannot decode {string!r} as base32: {e}") from e
    
    return validate(garlic_host)

def to_string(buf: bytes) -> str:
    """Convert bytes to a garlic32 address string."""
    return base64.b32encode(validate(buf)).decode('ascii').rstrip('=')