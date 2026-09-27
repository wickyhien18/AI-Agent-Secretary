import docker
from docker.errors import ContainerError, ImageNotFound
from langchain_core.tools import tool

_client = docker.from_env()
PYTHON_IMAGE = "python:3.12-slim"
TIMEOUT_SECONDS = 100


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
    try:
        output = _client.containers.run(
            image=PYTHON_IMAGE,
            command=["python", "-c", code],
            network_disabled=True,      # no network access at all
            mem_limit="128m",           # hard memory cap
            nano_cpus=500_000_000,      # 0.5 CPU core cap
            remove=True,                # auto-delete container after run
            stdout=True,
            stderr=True,
            detach=False,
            timeout=TIMEOUT_SECONDS,
        )
        return output.decode("utf-8", errors="ignore")
    except ContainerError as e:
        return f"Execution failed (exit code {e.exit_status}):\n{e.stderr.decode('utf-8', errors='ignore')}"
    except ImageNotFound:
        return f"Sandbox image '{PYTHON_IMAGE}' not found locally. Run: docker pull {PYTHON_IMAGE}"
    except Exception as e:
        return f"Sandbox execution error: {e}"