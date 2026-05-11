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


def test_list_dir_returns_error_when_listing_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_iterdir = Path.iterdir

    def fail_iterdir(path: Path):
        if path == tmp_path:
            raise OSError("permission denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_iterdir)

    result = ListDirTool().execute({"path": "."}, ToolContext(workspace=tmp_path))

    assert result.success is False
    assert result.summary == "Directory listing failed: ."
    assert result.error == "Directory listing failed"


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


def test_read_file_returns_error_when_read_fails(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello", encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def fail_read_bytes(path: Path) -> bytes:
        if path == target:
            raise OSError("permission denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    result = ReadFileTool().execute(
        {"path": "note.txt"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.summary == "File read failed: note.txt"
    assert result.error == "File read failed"


def test_write_file_writes_text_and_reports_bytes(tmp_path: Path) -> None:
    result = WriteFileTool().execute(
        {"path": "nested/out.txt", "content": "hello", "create_parent_dirs": True},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert (tmp_path / "nested" / "out.txt").read_text(encoding="utf-8") == "hello"
    assert result.payload["bytes_written"] == 5


def test_write_file_does_not_clobber_existing_temp_sibling(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    temp_sibling = tmp_path / "out.txt.tmp"
    temp_sibling.write_text("keep me", encoding="utf-8")

    result = WriteFileTool().execute(
        {"path": "out.txt", "content": "hello"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "hello"
    assert temp_sibling.read_text(encoding="utf-8") == "keep me"


def test_write_file_returns_error_when_write_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_write_bytes = Path.write_bytes

    def fail_write_bytes(path: Path, data: bytes) -> int:
        if path.name.startswith(".out.txt."):
            raise OSError("disk full")
        return original_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_write_bytes)

    result = WriteFileTool().execute(
        {"path": "out.txt", "content": "hello"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.summary == "File write failed: out.txt"
    assert result.error == "File write failed"
    assert not (tmp_path / "out.txt").exists()


def test_write_file_rejects_directory_target(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()

    result = WriteFileTool().execute(
        {"path": "nested", "content": "hello"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.error == "Not a file"


def test_write_file_rejects_parent_that_is_file(tmp_path: Path) -> None:
    (tmp_path / "parent").write_text("not a directory", encoding="utf-8")

    result = WriteFileTool().execute(
        {"path": "parent/out.txt", "content": "hello"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.error == "Parent is not a directory"
