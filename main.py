from pathlib import Path
import sys
import argparse
import importlib
import importlib.metadata as importlib_metadata

DATA_DIR = Path("data")

def load_pdfs(data_dir: Path):
    # import loader locally so running the env-check doesn't require langchain
    from langchain.document_loaders import PyPDFLoader

    documents = []
    for pdf_file in data_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        documents.extend(docs)
    return documents

def split_documents(documents):
    # import splitter locally so running the env-check doesn't require langchain
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def smoke_test_env():
    """Print versions for common packages useful for debugging the environment."""
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

    print("Environment check:\n")
    for dist_name in pkgs:
        try:
            ver = importlib_metadata.version(dist_name)
            print(f"{dist_name}: {ver} (distribution)")
            continue
        except importlib_metadata.PackageNotFoundError:
            # fall through to try importing module by a guessed name
            pass

        # try to import a module name derived from distribution name
        mod_name = dist_name.replace("-", "_")
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", None)
            if ver is None:
                # try importlib metadata for module name as a fallback
                try:
                    ver = importlib_metadata.version(mod_name)
                except Exception:
                    ver = "unknown"
            print(f"{dist_name} -> import {mod_name}: {ver}")
        except Exception as e:
            print(f"{dist_name}: not installed or import failed ({e.__class__.__name__})")


def run_llm(prompt: str, model_name: str = "gpt2", max_new_tokens: int = 50):
    """Run a very small LLM text-generation locally using transformers.

    This function imports heavy libraries only when called. It uses a CPU
    fallback and keeps defaults small so it works on low-resource machines.
    """
    # local imports to avoid top-level dependency on transformers/torch
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as e:
        raise RuntimeError("transformers and torch must be installed to run the LLM") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # keep generation conservative on CPU
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    return out


if __name__ == "__main__":
    def cli():
        parser = argparse.ArgumentParser(
            description="CLI wrapper for syllabus-rag-qna: env-check, llm, and PDF processing"
        )
        sub = parser.add_subparsers(dest="command", required=True)

        sub.add_parser("env-check", help="Print installed package versions for debugging the env")

        p_llm = sub.add_parser("llm", help="Run a small LLM generation (requires transformers+torch)")
        p_llm.add_argument("prompt", help="Prompt text to send to the model")
        p_llm.add_argument("--model", default="gpt2", help="Model name (default: gpt2)")
        p_llm.add_argument("--max-new-tokens", type=int, default=50, help="Max new tokens to generate")

        p_proc = sub.add_parser("process", help="Load PDFs from data dir, split into chunks, and show counts")
        p_proc.add_argument("--data-dir", default="data", help="Directory containing PDFs (default: data)")
        p_proc.add_argument("--sample", action="store_true", help="Print a sample chunk's text")

        args = parser.parse_args()

        if args.command == "env-check":
            smoke_test_env()
            return

        if args.command == "llm":
            print("Running LLM (this may download model weights on first run)...")
            out = run_llm(args.prompt, model_name=args.model, max_new_tokens=args.max_new_tokens)
            print("\n--- LLM OUTPUT ---\n")
            print(out)
            return

        if args.command == "process":
            data_path = Path(args.data_dir)
            try:
                docs = load_pdfs(data_path)
            except Exception as e:
                print(f"Failed to load PDFs: {e.__class__.__name__}: {e}")
                print("Make sure the 'langchain' loader dependencies are installed (langchain/document_loaders)")
                return

            print(f"Loaded {len(docs)} pages")
            chunks = split_documents(docs)
            print(f"Created {len(chunks)} text chunks")
            if args.sample:
                if chunks:
                    print("\n--- SAMPLE CHUNK ---\n")
                    print(chunks[0].page_content[:1000])
                else:
                    print("No chunks available to sample.")

    cli()
