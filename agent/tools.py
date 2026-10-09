from pathlib import Path
import chromadb
import difflib
import os
import fnmatch
import hashlib
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient
from langchain_core.tools import tool
from agent.docker_tool import execute_python

CHROMA_PATH = "./chroma_db"
READ_CHAR_BUDGET = 5500
embedding_fn = DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path=CHROMA_PATH)
tavily_client = TavilyClient()

_indexed_paths = set()

DENIED_NAMES = {".env", "chroma_db", ".git", ".venv"}

SKIP_DIRS = DENIED_NAMES | {"__pycache__", "node_modules", "venv", "site-packages", "dist", "build"}

SUFFIXES = (".py", ".md", ".toml", ".json", ".yaml", ".yml", ".txt", ".ini", ".cfg")

def suggest_paths(base: Path, relative_path: str, limit: int = 5) -> list[str]:
    wanted = Path(relative_path).name.lower()
    names = {}
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            names.setdefault(name.lower(), []).append(
                str((Path(dirpath) / name).relative_to(base)))
    close = difflib.get_close_matches(wanted, names, n=limit, cutoff=0.6)
    return [p for key in close for p in names[key]][:limit]

def collection_for(codebase_path: str, fresh: bool = False):
    name = "cb_" + hashlib.sha1(codebase_path.encode()).hexdigest()[:12]
    if fresh:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(name=name, embedding_function=embedding_fn)

def index_codebase(codebase_path: str) -> None:
    if codebase_path in _indexed_paths:
        return
    collection = collection_for(codebase_path, fresh=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents, metadatas, ids = [], [], []
    root = Path(codebase_path)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            file_path = Path(dirpath) / name
            if not name.endswith(SUFFIXES) or file_path.stat().st_size > 200_000:
                continue
            rel = str(file_path.relative_to(root))
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            for i, chunk in enumerate(splitter.split_text(text)):
                documents.append(chunk)
                metadatas.append({"source": rel})
                ids.append(f"{rel}:{i}")
    print(f"Indexing {len(documents)} chunks...")
    for start in range(0, len(documents), 200):
        end = start + 200
        collection.upsert(documents=documents[start:end],
                          metadatas=metadatas[start:end], ids=ids[start:end])
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

def err(message: str) -> str:
    return f"ERROR: {message}"

@tool
def search_codebase(query: str, codebase_path: str) -> str:
    """Semantic search over the CONTENT of source files, never over file names.
    Use it to find where something is implemented. To locate a file by name, use find_files.

    Args:
        query: what to look for, described in words
        codebase_path: root directory of the codebase being inspected
    """
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
def find_files(pattern: str, codebase_path: str) -> str:
    """Find files by NAME (not content) anywhere in the codebase, recursively.
    Use when you do not know the exact path. Not for searching file contents.

    Args:
        pattern: filename or glob pattern, e.g. 'tools.py' or '*tool*.py'
        codebase_path: root directory of the codebase being inspected
    """
    base = Path(codebase_path).resolve()
    matches = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if fnmatch.fnmatch(name.lower(), pattern.lower()):
                matches.append(str((Path(dirpath) / name).relative_to(base)))
                if len(matches) >= 50:
                    return "\n".join(matches) + "\n[more results omitted]"
    if matches:
        return "\n".join(matches)
    hints = suggest_paths(base, pattern.replace("*", ""))
    extra = f" Did you mean: {', '.join(hints)}?" if hints else ""
    return err(f"No file matches '{pattern}'.{extra}")

@tool
def read_file(path: str, codebase_path: str, offset: int = 0, limit: int = 200) -> str:
    """Read a file inside the codebase, a window of lines at a time.

    Args:
        path: file path relative to the codebase root
        codebase_path: root directory of the codebase being inspected
        offset: first line to read, 0-based (use it to continue a long file)
        limit: maximum number of lines to return
    """
    try:
        safe_path = resolve_safe_path(codebase_path, path)
    except ValueError as e:
        return err(str(e))

    if not safe_path.exists():
        hints = suggest_paths(Path(codebase_path).resolve(), path)
        extra = f" Did you mean: {', '.join(hints)}?" if hints else ""
        return err(f"File not found: {path}.{extra}")
    if not safe_path.is_file():
        return err(f"Not a file: {path}")

    lines = safe_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out, size, end = [], 0, offset
    for line in lines[offset:offset + limit]:
        if out and size + len(line) + 1 > READ_CHAR_BUDGET:
            break
        out.append(line)
        size += len(line) + 1
        end += 1
    text = "\n".join(out)
    if end < len(lines):
        text += (f"\n[showing lines {offset + 1}-{end} of {len(lines)}; "
                 f"call read_file again with offset={end} to continue]")
    return text

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
        return err(str(e))

    if not safe_path.exists():
        return err(f"Directory not found: {path}")
    if not safe_path.is_dir():
        return err(f"Not a directory: {path}")

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
        return err(str(e))

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
        return err(str(e))

    if not safe_path.exists():
        return err(f"File not found: {path}")

    text = safe_path.read_text(encoding="utf-8", errors="ignore")
    count = text.count(old_str)

    if count == 0:
        return err(f"old_str not found in {path}. No changes made.")
    if count > 1:
        return err(f"old_str appears {count} times in {path} — must be unique. No changes made.")

    safe_path.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
    return f"Edited {path} successfully."

tools = [search_codebase, search_web, read_file, list_directory, write_file, edit_file, execute_python, find_files]
tools_by_name = {t.name: t for t in tools}