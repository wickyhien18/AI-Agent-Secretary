import docker
from docker.errors import ImageNotFound
from langchain_core.tools import tool

_client = docker.from_env()
PYTHON_IMAGE = "python:3.12-slim"
TIMEOUT_SECONDS = 10


@tool
def execute_python(code: str) -> str:
    """Execute a short Python snippet in an isolated sandbox and return
    its stdout/stderr. The sandbox has no network access and no access
    to the host filesystem — use this only for self-contained
    computations, not for reading/writing project files (use read_file/
    write_file for that instead).

    Args:
        code: the Python source code to run
    """
    container = None
    try:
        container = _client.containers.run(
            image=PYTHON_IMAGE,
            command=["python", "-c", code],
            network_disabled=True,
            mem_limit="128m",
            nano_cpus=500_000_000,
            detach=True,
        )

        try:
            result = container.wait(timeout=TIMEOUT_SECONDS)
        except Exception:
            container.kill()
            return f"Execution timed out after {TIMEOUT_SECONDS}s."

        logs = container.logs().decode("utf-8", errors="ignore")
        exit_code = result.get("StatusCode", -1)

        if exit_code != 0:
            return f"Execution failed (exit code {exit_code}):\n{logs}"
        return logs

    except ImageNotFound:
        return f"Sandbox image '{PYTHON_IMAGE}' not found locally. Run: docker pull {PYTHON_IMAGE}"
    except Exception as e:
        return f"Sandbox execution error: {e}"
    finally:
        if container is not None:
            container.remove(force=True)