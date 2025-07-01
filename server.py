from src.server import app, ipfs_gateway, ipfs_api, rest_api, mcp

mcp_app = mcp.http_app()
app.mount("/ipfs", ipfs_gateway)
app.mount("/api/v0", ipfs_api)
app.mount("/api", rest_api)

# Nothing can be mounted after this
app.mount("", mcp_app)

def main():
    import uvicorn
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    main()