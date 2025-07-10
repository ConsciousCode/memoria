#!/usr/bin/env python3
"""
MCP host for Memoria: autonomously drives ACT scheduling by calling act_advance
against a running Memoria MCP server, using the same sampling logic as the CLI.
"""
import os
import sys
import asyncio

from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.client.client import Client
from mcp.types import SamplingMessage

from cli import MemoriaApp
from src.emulator.client_emu import ClientEmulator


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Autonomous Memoria host: schedules ACT steps via act_advance" )
    parser.add_argument("--url", "-u",
        default="http://localhost:8000/http/mcp",
        help="Memoria MCP server URL (e.g. http://host:port/http/mcp)")
    parser.add_argument("--config", "-c",
        default="~/.memoria/config",
        help="Path to Memoria client configuration file")
    parser.add_argument("--sona", "-s",
        required=True,
        help="Sona UUID or alias to advance")
    parser.add_argument("--interval", "-i",
        type=float, default=1.0,
        help="Seconds between act_advance calls")
    args = parser.parse_args()

    # Instantiate MemoriaApp to use its sampling_handler and config logic
    app = MemoriaApp([], help=None, config=os.path.expanduser(args.config))
    sampling_handler = app.sampling_handler

    # Connect to the Memoria server over MCP, providing our sampling handler
    async with Client(
        StreamableHttpTransport(args.url),
        sampling_handler=sampling_handler
    ) as client:
        emu = ClientEmulator(client)
        while True:
            try:
                new_memories = await emu.act_advance(args.sona)
                for mem in new_memories:
                    print(mem)
            except Exception as e:
                print(f"Error advancing ACT: {e}", file=sys.stderr)
            await asyncio.sleep(args.interval)


if __name__ == "__main__":
    asyncio.run(main())