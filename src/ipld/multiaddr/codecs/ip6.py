import netaddr

SIZE = 128
IS_PATH = False

def to_bytes(proto, string: str):
    return netaddr.IPAddress(string, version=6).packed

def to_string(proto, buf: bytes):
    ip_addr = netaddr.IPAddress(int.from_bytes(buf, byteorder='big'), version=6)
    return str(ip_addr)
