# syllabus-rag-qna

Small prototype for extracting content from course syllabi and running lightweight RAG/Q&A.

This repository includes:

- `main.py` — tools to load PDFs, split text into chunks, a small LLM helper (via `transformers`) and a CLI wrapper exposing `env-check`, `llm`, and `process` subcommands.
- `app/server.py` — FastAPI backend exposing `/env-check`, `/llm` and `/process` endpoints that reuse helpers from `main.py`.
- `requirements.txt` — pinned dependencies for reproducible installs.

## Quickstart (development)

1. Create & activate the project's venv (if you haven't already):

   On Windows (PowerShell):

   ```powershell
   python -m venv venv
   venv\Scripts\Activate.ps1
   ```

2. Install requirements:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Use the CLI in `main.py`:

   - Check the environment (quick, no heavy imports):

     ```powershell
     python main.py env-check
     ```

   - Run a small local LLM generation (may download model weights on first run):

     ```powershell
     python main.py llm "Write a short greeting" --model distilgpt2
     ```

   - Process PDFs in the `data/` directory (loads PDFs and splits to chunks):

     ```powershell
     python main.py process --data-dir data --sample
     ```

## Running the FastAPI backend

Start the server with Uvicorn (development):

```powershell
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload
```

Open http://127.0.0.1:8000/docs for the interactive API docs (Swagger UI).

Available endpoints:

- `GET /env-check` — returns installed package versions as JSON.
- `POST /llm` — JSON {prompt, model, max_new_tokens} returns generated text.
- `POST /process` — JSON {data_dir, sample} returns pages/chunks counts and optional sample text.

## Notes & safety

- The `llm` endpoints and `main.py llm` call into `transformers` and `torch`. Running them will likely download model weights (hundreds of MB) on first use.
- Defaults are chosen to be small (e.g., `gpt2` / `distilgpt2`) to keep resource needs reasonable. For production or faster results, use hosted inference APIs or GPU-backed instances.
- The `process` endpoint uses `langchain` document loaders; if that package or its backends are not installed, the endpoint will return an informative error.

## Tests

There is a small FastAPI test (`app/test_server.py`) that exercises the `/env-check` endpoint using FastAPI's TestClient. Run it with:

```powershell
python -m app.test_server
```

## Commit & push

All project changes have been committed locally; push to the remote with:

```powershell
git push origin main
```

If your remote requires authentication, configure your Git credentials or use an access token.

---
If you'd like, I can:

- Start the uvicorn server for you locally now.
- Add a JSON output flag to the CLI or endpoints for CI usage.
- Add an integration test for `/llm` using a tiny model or mocking.

Tell me what you'd like next.
