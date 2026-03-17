# API server

An API server for Woosh model inference is provided, with `Woosh-DFlow` only being supported at the moment.

## Usage

### Starting the server

From the `woosh` root repo folder, run

```bash
uv run uvicorn api.api_server:app --host 0.0.0.0 --port 8000
```
The server will be accessible on `http://127.0.0.1:8000`


## API client test
To test audio generation via the API, run

```python
uv run api/test_api.py
```

The audio output will be stored in the main `outputs/` folder.

### API documentation

You can visit

```
http://localhost:8000/docs
```

on the API server to access the documentation.
