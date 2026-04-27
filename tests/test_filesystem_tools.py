from pathlib import Path

import pytest

from agentskeleton.policy.paths import PathSecurityError, resolve_workspace_path
from agentskeleton.tools.base import ToolContext
from agentskeleton.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool


def test_resolve_workspace_path_blocks_parent_escape(tmp_path: Path) -> None:
    with pytest.raises(PathSecurityError):
        resolve_workspace_path(tmp_path, "../outside.txt")


def test_resolve_workspace_path_blocks_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-target"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(PathSecurityError):
        resolve_workspace_path(tmp_path, "link/secret.txt")


def test_list_dir_lists_direct_children(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "folder").mkdir()

    result = ListDirTool().execute({"path": "."}, ToolContext(workspace=tmp_path))

    assert result.success is True
    assert result.payload["entries"] == [
        {"name": "a.txt", "type": "file"},
        {"name": "folder", "type": "directory"},
    ]


def test_read_file_reads_utf8_text(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")

    result = ReadFileTool().execute(
        {"path": "note.txt"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert result.payload["content"] == "hello"
    assert result.summary == "Read 5 characters from note.txt"


def test_read_file_rejects_binary_content(tmp_path: Path) -> None:
    (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02")

    result = ReadFileTool().execute(
        {"path": "data.bin"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.error == "Binary file rejected"


def test_write_file_writes_text_and_reports_bytes(tmp_path: Path) -> None:
    result = WriteFileTool().execute(
        {"path": "nested/out.txt", "content": "hello", "create_parent_dirs": True},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert (tmp_path / "nested" / "out.txt").read_text(encoding="utf-8") == "hello"
    assert result.payload["bytes_written"] == 5
