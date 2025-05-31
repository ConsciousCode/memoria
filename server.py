import traceback as tb

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager

from memoria import Database, Memoria # Assuming memoria.py is in the same directory

# Configuration (can remain global or be moved into lifespan/create_app if preferred)
DATABASE_PATH = "memoria.db"
FILE_PATH = "files"

class EventPayload(BaseModel):
    prompt: str
    prev: Optional[int] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Starting up server...")
    with Database(db_path=DATABASE_PATH, file_path=FILE_PATH) as db:
        app.state.db = db
        app.state.memoria = Memoria(db)
        yield  # The application runs while the context manager is active
    
    # Shutdown logic
    print("Shutting down server...")
    if hasattr(app.state, 'db') and app.state.db is not None:
        try:
            app.state.db.__exit__(None, None, None)  # Close connection
            print("Database connection closed successfully.")
        except Exception as e:
            print(f"Error closing database connection: {e}")

app = FastAPI(lifespan=lifespan)

@app.post("/event")
async def create_event(payload: EventPayload, request: Request):
    memoria: Memoria = request.app.state.memoria
    
    try:
        return await memoria.prompt("user", payload.prev, "", payload.prompt)
    except Exception as e:
        tb.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process event: {str(e)}")

@app.get("/recall")
async def recall(payload: EventPayload, request: Request):
    memoria: Memoria = request.app.state.memoria
    
    try:
        ms: dict[str, dict] = {}
        for m in memoria.recall(prev=payload.prev, prompt=payload.prompt):
            d = {
                "kind": m.kind,
                "data": m.data
            }
            if m.edges: d['edges'] = m.edges
            if m.timestamp: d["timestamp"] = m.timestamp.timestamp()
            if m.role: d["role"] = m.role
            
            ms[f"{m.rowid:03x}"] = d
        
        return ms
    except Exception as e:
        tb.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to recall: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Memoria REST server is running."}

def main():
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
