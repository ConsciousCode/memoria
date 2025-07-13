from src.server import app, ipfs_gateway, ipfs_api, rest_api, mcp_http

app.mount("/ipfs", ipfs_gateway)
app.mount("/api/v0", ipfs_api)
app.mount("/memoria", rest_api)

# Nothing can be mounted after this
app.mount("", mcp_http)

def main():
    import uvicorn
    config = uvicorn.Config(
        app,
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