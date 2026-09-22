from pathlib import Path
from langchain_core.tools import tool

# Every read/write operation is confined to this directory.
# Change this to point at whichever project the agent should inspect.
BASE_DIR = Path("./workspace").resolve()
BASE_DIR.mkdir(exist_ok=True)


def resolve_safe_path(relative_path: str) -> Path:
    """Resolve relative_path against BASE_DIR and reject anything that
    escapes it. This is the direct fix for the path-traversal lesson
    (CVE-2026-34070): never trust a path string on its own, always
    check where it actually points after resolving '..' segments."""
    candidate = (BASE_DIR / relative_path).resolve()

    if not candidate.is_relative_to(BASE_DIR):
        raise ValueError(
            f"Path '{relative_path}' escapes the allowed directory."
        )
    return candidate


@tool
def read_file(path: str) -> str:
    """Read the contents of a file inside the agent's workspace.

    Args:
        path: file path relative to the workspace root
    """
    try:
        safe_path = resolve_safe_path(path)
    except ValueError as e:
        return str(e)

    if not safe_path.exists():
        return f"File not found: {path}"
    if not safe_path.is_file():
        return f"Not a file: {path}"

    return safe_path.read_text(encoding="utf-8", errors="ignore")


@tool
def list_directory(path: str = ".") -> str:
    """List files and folders inside a directory in the agent's workspace.

    Args:
        path: directory path relative to the workspace root, defaults to root
    """
    try:
        safe_path = resolve_safe_path(path)
    except ValueError as e:
        return str(e)

    if not safe_path.exists():
        return f"Directory not found: {path}"
    if not safe_path.is_dir():
        return f"Not a directory: {path}"

    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in safe_path.iterdir())
    return "\n".join(entries) if entries else "(empty directory)"