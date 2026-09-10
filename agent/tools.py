from langchain_core.tools import tool

@tool
def search_codebase(query: str, codebase_path: str) -> str:
    """ Search for relevant code in the specified codebase

    Args: 
        query: keyword or question about the code to search for
        codebase_path: path to the repository to search in
    """

    return f"[stub] searched '{query} in '{codebase_path}'"

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