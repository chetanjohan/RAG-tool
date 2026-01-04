from pathlib import Path
import sys
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


if __name__ == "__main__":
    # run a light environment check without loading langchain-heavy modules
    if len(sys.argv) > 1 and sys.argv[1] == "env-check":
        smoke_test_env()
        sys.exit(0)

    docs = load_pdfs(DATA_DIR)
    print(f"Loaded {len(docs)} pages")

    chunks = split_documents(docs)
    print(f"Created {len(chunks)} text chunks")

    # sanity check
    print("\n--- SAMPLE CHUNK ---\n")
    print(chunks[0].page_content[:500])
