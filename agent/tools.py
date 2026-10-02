from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient
from langchain_core.tools import tool
from agent.docker_tool import execute_python

CHROMA_PATH = "./chroma_db"
embedding_fn = DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path=CHROMA_PATH)
tavily_client = TavilyClient()

_indexed_paths = set()

DENIED_NAMES = {".env", "chroma_db", ".git", ".venv"}

def index_codebase(codebase_path: str) -> None:
    """Chunk every source file in codebase_path and store embeddings in Chroma.
    Only runs once per path (cached in _indexed_paths)."""
    if codebase_path in _indexed_paths:
        return

    collection = client.get_or_create_collection(
        name="codebase", embedding_function=embedding_fn
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    documents, metadatas, ids = [], [], []
    chunk_id = 0
    for file_path in Path(codebase_path).rglob("*.py"):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = splitter.split_text(text)
        for chunk in chunks:
            documents.append(chunk)
            metadatas.append({"source": str(file_path)})
            ids.append(f"chunk_{chunk_id}")
            chunk_id += 1

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
    _indexed_paths.add(codebase_path)

def resolve_safe_path(codebase_path: str, relative_path: str) -> Path:
    """Resolve relative_path against codebase_path and reject anything
    that escapes it or touches a denied name."""
    base = Path(codebase_path).resolve()
    candidate = (base / relative_path).resolve()

    if not base.is_dir():
        raise ValueError(f"Codebase root '{codebase_path}' is not an existing directory.")

    if not candidate.is_relative_to(base):
        raise ValueError(f"Path '{relative_path}' escapes the allowed directory.")

    if any(part in DENIED_NAMES for part in candidate.relative_to(base).parts):
        raise ValueError(f"Access to '{relative_path}' is denied.")

    return candidate

@tool
def search_codebase(query: str, codebase_path: str) -> str:
    """Search for relevant code in the specified codebase."""
    index_codebase(codebase_path)
    collection = client.get_or_create_collection(
        name="codebase", embedding_function=embedding_fn
    )
    results = collection.query(query_texts=[query], n_results=3)

    if not results["documents"][0]:
        return "No relevant code found."

    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output.append(f"[{meta['source']}]\n{doc}")
    return "\n---\n".join(output)

@tool
def search_web(query: str) -> str:
    """Search the internet for information. Use this when the question
    is not related to code in the current repository."""
    response = tavily_client.search(query=query, max_results=3)

    output = []
    for result in response["results"]:
        output.append(f"[{result['url']}]\n{result['content']}")
    return "\n---\n".join(output)

@tool
def read_file(path: str, codebase_path: str) -> str:
    """Read the contents of a file inside the codebase.

    Args:
        path: file path relative to the codebase root
        codebase_path: root directory of the codebase being inspected
    """
    try:
        safe_path = resolve_safe_path(codebase_path, path)
    except ValueError as e:
        return str(e)

    if not safe_path.exists():
        return f"File not found: {path}. Use list_directory to find the correct path."
    if not safe_path.is_file():
        return f"Not a file: {path}"

    return safe_path.read_text(encoding="utf-8", errors="ignore")


@tool
def list_directory(path: str, codebase_path: str) -> str:
    """List files and folders inside a directory in the codebase.

    Args:
        path: directory path relative to the codebase root
        codebase_path: root directory of the codebase being inspected
    """
    try:
        safe_path = resolve_safe_path(codebase_path, path)
    except ValueError as e:
        return str(e)

    if not safe_path.exists():
        return f"Directory not found: {path}"
    if not safe_path.is_dir():
        return f"Not a directory: {path}"

    entries = sorted(
        p.name + ("/" if p.is_dir() else "")
        for p in safe_path.iterdir()
        if p.name not in DENIED_NAMES
    )
    return "\n".join(entries) if entries else "(empty directory)"

@tool
def write_file(path: str, content: str, codebase_path: str) -> str:
    """Create a new file or overwrite an existing file with the given content.

    Args:
        path: file path relative to the codebase root
        content: full text content to write into the file
        codebase_path: root directory of the codebase being inspected
    """
    try:
        safe_path = resolve_safe_path(codebase_path, path)
    except ValueError as e:
        return str(e)

    safe_path.parent.mkdir(parents=True, exist_ok=True)
    safe_path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {path}"


@tool
def edit_file(path: str, old_str: str, new_str: str, codebase_path: str) -> str:
    """Replace an exact, unique piece of text inside an existing file.
    Fails if old_str appears zero times or more than once, to avoid
    accidentally editing the wrong spot.

    Args:
        path: file path relative to the codebase root
        old_str: exact text to find and replace, must appear exactly once
        new_str: text to replace it with
        codebase_path: root directory of the codebase being inspected
    """
    try:
        safe_path = resolve_safe_path(codebase_path, path)
    except ValueError as e:
        return str(e)

    if not safe_path.exists():
        return f"File not found: {path}"

    text = safe_path.read_text(encoding="utf-8", errors="ignore")
    count = text.count(old_str)

    if count == 0:
        return f"old_str not found in {path}. No changes made."
    if count > 1:
        return f"old_str appears {count} times in {path} — must be unique. No changes made."

    safe_path.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
    return f"Edited {path} successfully."

tools = [search_codebase, search_web, read_file, list_directory, write_file, edit_file, execute_python]
tools_by_name = {t.name: t for t in tools}