from typing import Optional

class ProtocolLookupError(LookupError):
    """MultiAddr did not contain a protocol with the requested code."""

    def __init__(self, proto: str, string: str):
        super().__init__(
            f"MultiAddr {string!r} does not contain protocol {proto}"
        )
        self.proto = proto
        self.string = string

class ParseError(ValueError):
    pass

class StringParseError(ParseError):
    """MultiAddr string representation could not be parsed."""

    def __init__(self, message: str, string: str, protocol: Optional[str]=None, original: Optional[str]=None):
        if protocol:
            msg = f"Invalid MultiAddr {string!r} protocol {protocol}: {message}"
        else:
            msg = f"Invalid MultiAddr {string!r}: {message}"

        super().__init__(msg)
        self.message = message
        self.string = string
        self.protocol = protocol
        self.original = original

class BinaryParseError(ParseError):
    """MultiAddr binary representation could not be parsed."""

    def __init__(self, message: str, binary: bytes, protocol: str, original: Optional[str]=None):
        super().__init__(
            f"Invalid binary MultiAddr protocol {protocol}: {message}"
        )
        self.message = message
        self.binary = binary
        self.protocol = protocol
        self.original = original

class ProtocolManagerError(Exception):
    pass

class ProtocolNotFoundError(ProtocolManagerError):
    """No protocol with the given name or code found."""
    def __init__(self, value: int|str, kind: str="name"):
        super().__init__(
            f"No protocol with {kind} {value!r} found"
        )
        self.value = value
        self.kind = kind
