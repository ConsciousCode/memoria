from typing import Optional
from base64 import b64encode

from fastmcp.client.transports import ClientTransport

from models import UploadResponse
from server._common_server import AddParameters

from .emulator.client_emu import ClientEmulator

class MemoriaClient[TransportT: ClientTransport](ClientEmulator[TransportT]):
    async def upload(self, data: bytes, mimetype: str, filename: Optional[str], params: AddParameters) -> UploadResponse:
        """Upload a file to the Memoria MCP client."""
        return await self._call_tool(UploadResponse, "upload", {
            "file": b64encode(data),
            "filename": filename,
            "content_type": mimetype,
            "params": params
        })