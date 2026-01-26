import importlib
print("DEBUG multipart import", importlib.util.find_spec("multipart"))

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, FastAPI
import asyncio
from typing import List
from pathlib import Path
import shutil
import re
from fastapi.middleware.cors import CORSMiddleware


router = APIRouter()

# Expose a FastAPI app from this module so uvicorn app.main:app works
app = FastAPI(title="syllabus-rag-qna generate")

# CORS MUST COME IMMEDIATELY AFTER app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://study-buddy-ai-5d935db1.base44.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





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


def _split_texts(pages, chunk_size=800, overlap=200, max_chunks=50):
    chunks = []
    for text in pages:
        start = 0
        # safety: ensure overlap is smaller than chunk_size to make forward progress
        if overlap >= chunk_size:
            overlap = max(1, chunk_size // 2)
        # safety: avoid producing an enormous number of chunks that could exhaust memory
        max_chunks = 20000
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            iterations = 0
            if len(chunks) >= max_chunks:
                # stop splitting further to avoid memory blowup
                break
            # advance start: ensure progress and avoid infinite loops
            new_start = end - overlap
            if new_start <= start:
                # no forward progress; move to end to avoid infinite loop
                start = end
            else:
                start = new_start
            start = end - overlap
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


try:
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
        if not top_k:
            return {
                "answer": "Answer not found in document.",
                "context_chunks": [],
                "scores": []
        }

        # Prepare context and a strict, exam-focused instruction for the LLM.
        context = "\n\n---\n\n".join(top_k)
        prompt = (
            "You are an exam assistant. Use ONLY the CONTEXT below to answer the QUESTION. "
            "Do NOT use any outside knowledge. If the information to answer the question is not "
            "present in the CONTEXT, respond exactly: 'Answer not found in document.'\n\n"
            "CONTEXT:\n" + context + "\n\nQUESTION:\n" + question + "\n\n"
            "INSTRUCTIONS:\n1) If answer is present in the context, write a short direct answer (1-3 sentences). "
            "2) Then add 2 short exam-style bullet points summarizing the key facts or how to remember them. "
            "3) If the answer is NOT present, respond exactly with 'Answer not found in document.'\n\n"
            "ANSWER:\n"
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
except RuntimeError as _route_err:
    # If FastAPI raises a RuntimeError during route registration (for example,
    # the multipart dependency is missing), register a fallback route that
    # returns a clear error message instead of letting the import-time error
    # crash the app.
    @router.post("/generate")
    async def generate_disabled():
        raise HTTPException(
            status_code=500,
            detail=(
                "/generate endpoint unavailable: import-time error during route registration. "
                f"Underlying error: {_route_err}. Install python-multipart or check server logs."
            ),
        )


# include router after route definitions so routes are registered on the app
app.include_router(router)

# include server routes (env-check, llm, process) from app.server
try:
    from app.server import router as server_router
    app.include_router(server_router)
except Exception:
    # if import fails, the app will still run with /generate only
    pass
