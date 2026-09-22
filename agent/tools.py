from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

CHROMA_PATH = "./chroma_db"
embedding_fn = DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path=CHROMA_PATH)

_indexed_paths = set()


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
    """ Search the internet for information. Use this when the question is not
    related to the code in current repository.

    Args:
        query: search query or question to look up on the web
    """

    return f"[stub] web search results for '{query}'"

tools = [search_codebase, search_web]
tools_by_name = {t.name: t for t in tools}