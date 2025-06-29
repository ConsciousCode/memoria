import netaddr

SIZE = 32
IS_PATH = False

def to_bytes(string: str):
    return netaddr.IPAddress(string, version=4).packed

def to_string(buf: bytes):
    ip_addr = netaddr.IPAddress(int.from_bytes(buf, byteorder='big'), version=4)
    return str(ip_addr)
