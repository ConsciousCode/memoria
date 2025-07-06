from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
app = FastAPI()
@app.post("/add")
async def add(request: Request):
    async def stream():
        print("Begin")
        print("parser")
        print("p")
        async for chunk in request.stream():
            print("chunk of", len(chunk))
            yield chunk
        print("Done")
        yield b''
    return StreamingResponse(stream(), media_type="application/octet-stream")
