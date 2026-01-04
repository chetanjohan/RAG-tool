from fastapi import APIRouter, UploadFile, File, Form, HTTPException, FastAPI
import asyncio
from typing import List
from pathlib import Path
import shutil
import re

router = APIRouter()

# Expose a FastAPI app from this module so uvicorn app.main:app works
app = FastAPI(title="syllabus-rag-qna generate")


def _safe_text_from_pdf(path: Path) -> List[str]:
    """Try to extract text from PDF with several fallbacks.

    Returns a list of page texts.
    """
    # Preferred: langchain loader if available
    try:
        from langchain.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(path))
        docs = loader.load()
        pages = [d.page_content for d in docs]
        return pages
    except Exception:
        pass

    # Fallback: pypdf (PyPDF) if available
    try:
        import pypdf

        reader = pypdf.PdfReader(str(path))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return pages
    except Exception:
        pass

    raise RuntimeError("No available PDF loader (install langchain or pypdf)")


def _split_texts(pages: List[str], chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split page texts into overlapping chunks. Returns list of chunk strings."""
    chunks = []
    for page in pages:
        text = page.strip()
        if not text:
            continue
        start = 0
        L = len(text)
        while start < L:
            end = min(start + chunk_size, L)
            chunks.append(text[start:end])
            start = end - overlap
            if start < 0:
                start = 0
            if start >= L:
                break
    return chunks


def _score_chunks(question: str, chunks: List[str]) -> List[tuple]:
    """Score chunks with a simple token-overlap heuristic and return list of (score, chunk)"""
    def tokens(s: str):
        return set(re.findall(r"\w+", s.lower()))

    q_tokens = tokens(question)
    scored = []
    for c in chunks:
        c_tokens = tokens(c)
        if not c_tokens:
            score = 0
        else:
            score = len(q_tokens & c_tokens)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


@router.post("/generate")
async def generate(file: UploadFile = File(...), question: str = Form(...)):
    """Accept a PDF file upload and a question, run simple RAG retrieval and generate an answer.

    Returns JSON {answer, context_chunks, scores}
    """
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    fname = uploads_dir / file.filename
    try:
        with fname.open("wb") as out_f:
            shutil.copyfileobj(file.file, out_f)
    finally:
        file.file.close()

    # Extract page texts
    try:
        pages = _safe_text_from_pdf(fname)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract PDF text: {e}")

    # Split into chunks
    chunks = _split_texts(pages, chunk_size=800, overlap=200)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted from PDF")

    # Score and pick top-k
    scored = _score_chunks(question, chunks)
    top_k = [c for s, c in scored[:5] if s > 0]

    # Prepare context
    context = "\n\n---\n\n".join(top_k)
    prompt = (
        "You are an assistant that answers questions using the provided context."
        "\nContext:\n" + context + "\n\nQuestion:\n" + question + "\n\nAnswer:"
    )

    # Call existing LLM helper
    try:
        from main import run_llm
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM helper not available: {e}")

    try:
        # run the CPU-bound LLM in a thread to avoid blocking the event loop
        answer = await asyncio.to_thread(run_llm, prompt, "distilgpt2", 200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    return {
        "answer": answer,
        "context_chunks": top_k,
        "scores": [s for s, c in scored[:5]],
    }


# include router after route definitions so routes are registered on the app
app.include_router(router)

# include server routes (env-check, llm, process) from app.server
try:
    from app.server import router as server_router
    app.include_router(server_router)
except Exception:
    # if import fails, the app will still run with /generate only
    pass
