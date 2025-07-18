#!/usr/bin/env python3.13

'''
Server which hosts the Memoria Subject, a social construction which is the
locus of experience and agency. It does nothing on its own except manage ACT dynamics, push requests for state advancement to the Memoria Processor, and
equivalences the results as a coherent subjective experience.

Endpoints are provided for a trustless IPFS gateway, Kubo-compatible IPFS API,
a REST API for Memoria, and an Anthropic Model Context Protocol (MCP) interface.
'''

from fastapi import FastAPI

from src.server import ipfs_gateway, ipfs_api, rest_api, mcp

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
    try:
        server.run()
    except KeyboardInterrupt:
        print("Server stopped by user.")

if __name__ == "__main__":
    main()