#!/usr/bin/env python3.13

'''
Server which hosts the Memoria Subject.

Endpoints are provided for a trustless IPFS gateway, Kubo-compatible IPFS API,
a REST API for Memoria, and an Anthropic Model Context Protocol (MCP) interface.
'''

from fastapi import FastAPI

from .ipfs import ipfs_gateway, ipfs_api
from .mcp import mcp
from .rest import rest_api

def build_app():
    mcp_http = mcp.http_app()
    app = FastAPI(lifespan=mcp_http.lifespan)

    app.mount("/ipfs", ipfs_gateway)
    app.mount("/api/v0", ipfs_api)
    app.mount("/memoria", rest_api)

    # Nothing can be mounted after this
    app.mount("", mcp_http)

    return app

def main():
    import uvicorn
    config = uvicorn.Config(
        build_app(),
        host="0.0.0.0",
        port=8000
    )
    server = uvicorn.Server(config)
    try: server.run()
    except KeyboardInterrupt:
        print("Subject server stopped by user.")

if __name__ == "__main__":
    main()