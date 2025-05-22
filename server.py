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

def create_app():
    app = FastAPI(lifespan=lifespan)

    @app.post("/events/")
    async def create_event(payload: EventPayload, request: Request):
        if not hasattr(request.app.state, 'memoria_instance') or \
           request.app.state.memoria_instance is None or \
           not hasattr(request.app.state, 'db') or \
           request.app.state.db is None:
            # This check might be redundant if startup guarantees these or exits
            raise HTTPException(status_code=503, detail="Memoria service is not initialized properly.")
        
        memoria_instance = request.app.state.memoria_instance
        
        try:
            return await memoria_instance.process(prev=payload.prev, prompt=payload.prompt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process event: {str(e)}")

    @app.get("/")
    async def root():
        return {"message": "Memoria REST server is running."}
        
    return app

# Create the app instance for uvicorn discovery (e.g., uvicorn server:app)
app = create_app()

if __name__ == "__main__":
    # This allows running the server with 'python server.py'
    uvicorn.run(app, host="127.0.0.1", port=8000)
