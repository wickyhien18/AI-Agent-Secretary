from pathlib import Path
from langchain_core.tools import tool

# Paths containing any of these names are always denied, even inside
# an otherwise-allowed codebase_path — prevents the agent from reading
# its own secrets (.env) or internal data (chroma_db, .git).
DENIED_NAMES = {".env", "chroma_db", ".git", ".venv"}


def resolve_safe_path(codebase_path: str, relative_path: str) -> Path:
    """Resolve relative_path against codebase_path and reject anything
    that escapes it or touches a denied name."""
    base = Path(codebase_path).resolve()
    candidate = (base / relative_path).resolve()

    if not candidate.is_relative_to(base):
        raise ValueError(f"Path '{relative_path}' escapes the allowed directory.")

    if any(part in DENIED_NAMES for part in candidate.relative_to(base).parts):
        raise ValueError(f"Access to '{relative_path}' is denied.")

    return candidate


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
        return f"File not found: {path}"
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