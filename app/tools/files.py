"""WRITE tools: create_file / delete_file, confined to a sandbox directory.

These have side effects, so they are registered with Permission.WRITE and the agent asks
the user for confirmation before the executor is allowed to run them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.tools.base import Permission, Tool, ToolError

MAX_FILE_BYTES = 100_000


def _safe_path(sandbox: Path, filename: str) -> Path:
    """Resolve `filename` inside the sandbox; reject anything that escapes it."""
    sandbox = sandbox.resolve()
    target = (sandbox / filename).resolve()
    if target == sandbox or sandbox not in target.parents:
        raise ToolError(f"Path {filename!r} is outside the allowed directory.", "forbidden_path")
    return target


def make_file_tools(sandbox_dir: Path) -> list[Tool]:
    def create_file(filename: str, content: str, overwrite: bool = False) -> dict[str, Any]:
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ToolError(f"Content too large (max {MAX_FILE_BYTES} bytes).", "too_large")
        path = _safe_path(sandbox_dir, filename)
        if path.exists() and not overwrite:
            raise ToolError(f"{filename!r} already exists. Set overwrite=true to replace it.", "already_exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"path": path.relative_to(sandbox_dir.resolve()).as_posix(), "bytes_written": path.stat().st_size}

    def delete_file(filename: str) -> dict[str, Any]:
        path = _safe_path(sandbox_dir, filename)
        if not path.is_file():
            raise ToolError(f"{filename!r} does not exist.", "not_found")
        path.unlink()
        return {"deleted": path.relative_to(sandbox_dir.resolve()).as_posix()}

    filename_schema = {
        "type": "string",
        "description": "Relative file path inside the workspace, e.g. 'notes/todo.txt'.",
        "minLength": 1,
        "maxLength": 200,
    }
    return [
        Tool(
            name="create_file",
            description="Create a text file in the user's workspace. Requires user confirmation.",
            parameters={
                "type": "object",
                "properties": {
                    "filename": filename_schema,
                    "content": {"type": "string", "description": "Full text content of the file."},
                    "overwrite": {"type": "boolean", "description": "Replace the file if it exists (default false)."},
                },
                "required": ["filename", "content"],
                "additionalProperties": False,
            },
            function=create_file,
            title="Create File",
            permission=Permission.WRITE,
        ),
        Tool(
            name="delete_file",
            description="Delete a file from the user's workspace. Requires user confirmation.",
            parameters={
                "type": "object",
                "properties": {"filename": filename_schema},
                "required": ["filename"],
                "additionalProperties": False,
            },
            function=delete_file,
            title="Delete File",
            permission=Permission.WRITE,
        ),
    ]
