from fastapi import FastAPI

from src.server import ipfs_app, rest_app, mcp

mcp_app = mcp.http_app()
app = FastAPI(lifespan=mcp_app.lifespan)
app.mount("/ipfs", ipfs_app)
app.mount("/api", rest_app)

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