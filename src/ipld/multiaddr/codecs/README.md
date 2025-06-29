All modules in this directory are used to encode and decode multiaddr codecs. They act as objects in and of themselves implementing a protocol,

```python
class Codec(Protocol):
    SIZE: int
    IS_PATH: bool

    to_bytes: Callable[[MAProtocol, str], bytes]
    to_string: Callable[[MAProtocol, bytes], str]
```