from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import importlib.metadata as importlib_metadata
from pathlib import Path

router = APIRouter()


def get_package_versions(pkg_list):
    out = {}
    for name in pkg_list:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = None
    return out


@router.get("/env-check")
def env_check():
    pkgs = [
        "langchain",
        "langchain-community",
        "langchain-huggingface",
        "faiss-cpu",
        "pypdf",
        "sentence-transformers",
        "transformers",
        "torch",
    ]
    return get_package_versions(pkgs)


class LLMRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gpt2"
    max_new_tokens: Optional[int] = 50


@router.post("/llm")
def llm_endpoint(req: LLMRequest):
    # import the run_llm helper from main.py (lazy heavy imports are inside)
    try:
        from main import run_llm
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM helper not available: {e}")

    try:
        out = run_llm(req.prompt, model_name=req.model, max_new_tokens=req.max_new_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")
    return {"output": out}


class ProcessRequest(BaseModel):
    data_dir: Optional[str] = "data"
    sample: Optional[bool] = False


@router.post("/process")
def process_endpoint(req: ProcessRequest):
    try:
        from main import load_pdfs, split_documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing helper not available: {e}")

    data_path = Path(req.data_dir)
    if not data_path.exists():
        raise HTTPException(status_code=400, detail=f"data_dir not found: {data_path}")

    try:
        docs = load_pdfs(data_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load PDFs: {e}")

    chunks = split_documents(docs)
    resp = {"pages": len(docs), "chunks": len(chunks)}
    if req.sample:
        resp["sample"] = chunks[0].page_content[:1000] if chunks else ""
    return resp
