# Memoria Project

This project implements Memoria, a system for storing and recalling information, with an AI agent interface.

## Running the REST Server

This server provides a REST API for sending events to Memoria.

### Prerequisites

- Python 3.x
- Pip

### Installation

1.  Clone the repository (if you haven't already).
2.  Navigate to the project directory.
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Server

To run the FastAPI server, you have two main options:

1.  **Using Uvicorn (recommended for development with auto-reload):**
    ```bash
    uvicorn server:app --reload
    ```
    -   `server:app` refers to the `app` instance of FastAPI in the `server.py` file.
    -   `--reload` enables auto-reloading when code changes are detected.

2.  **Directly executing the Python script:**
    ```bash
    python server.py
    ```
    -   This method will run the server using the Uvicorn settings specified within the script (e.g., host and port).

The server will typically be available at `http://127.0.0.1:8000`. You can access the root endpoint at `http://127.0.0.1:8000/` to check if it's running, and the API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

### API Endpoints

-   **POST /events/**: Send an event to Memoria.
    -   **Request Body**:
        ```json
        {
            "prompt": "Your event description or query",
            "prev": null // or an integer ID of a previous memory
        }
        ```
    -   **Response**: Returns a JSON object with the processing result, including the ID of the new memory.
```
