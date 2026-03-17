# API server


## Usage

### Starting the server

```bash
uv run uvicorn api.api_server:app --host 0.0.0.0 --port 8000
```

## API test
To test audio generation via the API run

```python
uv run api/test_api.py
```

The audio output will be stored in the `outputs/` folder.

### API documentation

You can visit

```
http://localhost:8000/docs
```

on the API server to access the documentation.
