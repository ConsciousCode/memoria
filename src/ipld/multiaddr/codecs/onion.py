import base64

SIZE = 96
IS_PATH = False

def to_bytes(string: str):
    addr = string.split(":")
    if len(addr) != 2:
        raise ValueError("Does not contain a port number")

    # onion address without the ".onion" substring
    if len(addr[0]) != 16:
        raise ValueError("Invalid onion host address length (must be 16 characters)")
    
    try: onion_host = base64.b32decode(addr[0].upper())
    except Exception as e:
        raise ValueError(f"Cannot decode {addr[0]!r} as base32: {1}") from e

    # onion port number
    try: port = int(addr[1], 10)
    except ValueError as e:
        raise ValueError("Port number is not a base 10 integer") from e
    
    if port not in range(1, 65536):
        raise ValueError("Port number is not in range(1, 65536)")

    return onion_host + port.to_bytes(byteorder='big')

def to_string(buf: bytes):
    addr = base64.b32encode(buf[:-2]).decode('ascii').lower()
    port = int.from_bytes(buf[-2:], byteorder='big')
    return f"{addr}:{port}"
