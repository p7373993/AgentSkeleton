from pathlib import Path

import pytest

from agentskeleton.policy.paths import PathSecurityError, resolve_workspace_path
from agentskeleton.tools.base import ToolContext
from agentskeleton.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool


def test_resolve_workspace_path_blocks_parent_escape(tmp_path: Path) -> None:
    with pytest.raises(PathSecurityError):
        resolve_workspace_path(tmp_path, "../outside.txt")


def test_resolve_workspace_path_blocks_absolute_paths_inside_workspace(
    tmp_path: Path,
) -> None:
    target = tmp_path / "inside.txt"

    with pytest.raises(PathSecurityError, match="workspace-relative"):
        resolve_workspace_path(tmp_path, str(target))


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


@pytest.mark.parametrize(
    "requested_path",
    [
        "NUL",
        "CON.txt",
        "folder/COM1.log",
    ],
)
def test_resolve_workspace_path_blocks_windows_reserved_device_names(
    tmp_path: Path,
    requested_path: str,
) -> None:
    with pytest.raises(PathSecurityError, match="reserved Windows device name"):
        resolve_workspace_path(tmp_path, requested_path)


@pytest.mark.parametrize(
    "requested_path",
    [
        "note.txt:secret",
        "folder./note.txt",
        "folder /note.txt",
    ],
)
def test_resolve_workspace_path_blocks_ambiguous_windows_path_parts(
    tmp_path: Path,
    requested_path: str,
) -> None:
    with pytest.raises(PathSecurityError, match="ambiguous Windows path part"):
        resolve_workspace_path(tmp_path, requested_path)


def test_list_dir_lists_direct_children(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "folder").mkdir()

    result = ListDirTool().execute({"path": "."}, ToolContext(workspace=tmp_path))

    assert result.success is True
    assert result.payload["entries"] == [
        {"name": "a.txt", "type": "file"},
        {"name": "folder", "type": "directory"},
    ]


def test_list_dir_limits_large_directory_payload(tmp_path: Path) -> None:
    for index in range(205):
        (tmp_path / f"{index:03}.txt").write_text("a", encoding="utf-8")

    result = ListDirTool().execute({"path": "."}, ToolContext(workspace=tmp_path))

    assert result.success is True
    assert len(result.payload["entries"]) == 200
    assert result.payload["total_entries"] == 205
    assert result.payload["truncated"] is True
    assert result.summary == "Listed 200 of 205 entries in ."


@pytest.mark.parametrize(
    ("args", "summary"),
    [
        ({}, "Path invalid: path must be a string"),
        ({"path": 123}, "Path invalid: path must be a string"),
        ({"path": "   "}, "Path invalid: path cannot be blank"),
    ],
)
def test_list_dir_rejects_invalid_path(
    tmp_path: Path,
    args: dict[str, object],
    summary: str,
) -> None:
    result = ListDirTool().execute(args, ToolContext(workspace=tmp_path))

    assert result.success is False
    assert result.error == "Path invalid"
    assert result.summary == summary
    assert result.payload == {}


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        (ListDirTool(), {"path": "bad\x00name"}),
        (ReadFileTool(), {"path": "bad\x00name"}),
        (WriteFileTool(), {"path": "bad\x00name", "content": "hello"}),
    ],
)
def test_filesystem_tools_reject_control_characters_in_paths(
    tmp_path: Path,
    tool,
    args: dict[str, object],
) -> None:
    result = tool.execute(args, ToolContext(workspace=tmp_path))

    assert result.success is False
    assert result.error == "Path invalid"
    assert result.summary == "Path invalid: path cannot contain control characters"
    assert result.payload == {}


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


def test_list_dir_returns_error_when_exists_stat_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_exists = Path.exists

    def fail_exists(path: Path) -> bool:
        if path == tmp_path:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_exists)

    result = ListDirTool().execute({"path": "."}, ToolContext(workspace=tmp_path))

    assert result.success is False
    assert result.summary == "Directory listing failed: ."
    assert result.error == "Directory listing failed"


def test_list_dir_returns_error_when_directory_stat_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_is_dir = Path.is_dir

    def fail_is_dir(path: Path) -> bool:
        if path == tmp_path:
            raise OSError("permission denied")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", fail_is_dir)

    result = ListDirTool().execute({"path": "."}, ToolContext(workspace=tmp_path))

    assert result.success is False
    assert result.summary == "Directory listing failed: ."
    assert result.error == "Directory listing failed"


def test_list_dir_returns_error_when_entry_type_check_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "a.txt"
    target.write_text("a", encoding="utf-8")
    original_is_dir = Path.is_dir

    def fail_is_dir(path: Path) -> bool:
        if path == target:
            raise OSError("permission denied")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", fail_is_dir)

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


@pytest.mark.parametrize(
    ("args", "summary"),
    [
        ({}, "Path invalid: path must be a string"),
        ({"path": 123}, "Path invalid: path must be a string"),
        ({"path": "   "}, "Path invalid: path cannot be blank"),
    ],
)
def test_read_file_rejects_invalid_path(
    tmp_path: Path,
    args: dict[str, object],
    summary: str,
) -> None:
    result = ReadFileTool().execute(args, ToolContext(workspace=tmp_path))

    assert result.success is False
    assert result.error == "Path invalid"
    assert result.summary == summary
    assert result.payload == {}


def test_read_file_rejects_binary_content(tmp_path: Path) -> None:
    (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02")

    result = ReadFileTool().execute(
        {"path": "data.bin"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.error == "Binary file rejected"


def test_read_file_rejects_large_text_files(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_bytes(b"a" * 1_048_577)

    result = ReadFileTool().execute(
        {"path": "large.txt"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.error == "File too large"
    assert result.summary == "File too large: large.txt"


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


def test_read_file_returns_error_when_exists_stat_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello", encoding="utf-8")
    original_exists = Path.exists

    def fail_exists(path: Path) -> bool:
        if path == target:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_exists)

    result = ReadFileTool().execute(
        {"path": "note.txt"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.summary == "File read failed: note.txt"
    assert result.error == "File read failed"


def test_read_file_returns_error_when_file_stat_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello", encoding="utf-8")
    original_is_file = Path.is_file

    def fail_is_file(path: Path) -> bool:
        if path == target:
            raise OSError("permission denied")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", fail_is_file)

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


def test_write_file_allows_multiline_content(tmp_path: Path) -> None:
    result = WriteFileTool().execute(
        {"path": "out.txt", "content": "hello\nworld"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hello\nworld"


@pytest.mark.parametrize(
    ("args", "summary"),
    [
        ({"content": "hello"}, "Path invalid: path must be a string"),
        ({"path": 123, "content": "hello"}, "Path invalid: path must be a string"),
        ({"path": "   ", "content": "hello"}, "Path invalid: path cannot be blank"),
    ],
)
def test_write_file_rejects_invalid_path(
    tmp_path: Path,
    args: dict[str, object],
    summary: str,
) -> None:
    result = WriteFileTool().execute(args, ToolContext(workspace=tmp_path))

    assert result.success is False
    assert result.error == "Path invalid"
    assert result.summary == summary
    assert result.payload == {}


@pytest.mark.parametrize(
    ("args", "summary"),
    [
        ({"path": "out.txt"}, "Content invalid: content must be a string"),
        (
            {"path": "out.txt", "content": 123},
            "Content invalid: content must be a string",
        ),
    ],
)
def test_write_file_rejects_invalid_content(
    tmp_path: Path,
    args: dict[str, object],
    summary: str,
) -> None:
    result = WriteFileTool().execute(args, ToolContext(workspace=tmp_path))

    assert result.success is False
    assert result.error == "Content invalid"
    assert result.summary == summary
    assert result.payload == {}
    assert not (tmp_path / "out.txt").exists()


def test_write_file_rejects_large_content(tmp_path: Path) -> None:
    result = WriteFileTool().execute(
        {"path": "out.txt", "content": "a" * 1_048_577},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.error == "Content too large"
    assert result.summary == "Content too large: out.txt"
    assert not (tmp_path / "out.txt").exists()


def test_write_file_rejects_non_bool_create_parent_dirs(tmp_path: Path) -> None:
    result = WriteFileTool().execute(
        {"path": "nested/out.txt", "content": "hello", "create_parent_dirs": "true"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.error == "Create parent dirs invalid"
    assert (
        result.summary
        == "Create parent dirs invalid: create_parent_dirs must be a boolean"
    )
    assert result.payload == {}
    assert not (tmp_path / "nested").exists()


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


def test_write_file_uses_bounded_temp_name_for_long_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target_name = f"{'a' * 70}.txt"
    original_write_bytes = Path.write_bytes

    def fail_oversized_temp_name(path: Path, data: bytes) -> int:
        if path.name.startswith(".") and len(path.name) > 80:
            raise OSError("temp filename too long")
        return original_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_oversized_temp_name)

    result = WriteFileTool().execute(
        {"path": target_name, "content": "hello"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert (tmp_path / target_name).read_text(encoding="utf-8") == "hello"


def test_write_file_returns_error_when_write_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_write_bytes = Path.write_bytes

    def fail_write_bytes(path: Path, data: bytes) -> int:
        if path.name.startswith(".write-") and path.name.endswith(".tmp"):
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


def test_write_file_returns_error_when_parent_exists_stat_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parent = tmp_path / "nested"
    original_exists = Path.exists

    def fail_exists(path: Path) -> bool:
        if path == parent:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_exists)

    result = WriteFileTool().execute(
        {"path": "nested/out.txt", "content": "hello"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.summary == "File write failed: nested/out.txt"
    assert result.error == "File write failed"


def test_write_file_returns_error_when_parent_type_stat_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parent = tmp_path / "nested"
    parent.mkdir()
    original_is_dir = Path.is_dir

    def fail_is_dir(path: Path) -> bool:
        if path == parent:
            raise OSError("permission denied")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", fail_is_dir)

    result = WriteFileTool().execute(
        {"path": "nested/out.txt", "content": "hello"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.summary == "File write failed: nested/out.txt"
    assert result.error == "File write failed"


def test_write_file_returns_error_when_target_exists_stat_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "out.txt"
    original_exists = Path.exists

    def fail_exists(path: Path) -> bool:
        if path == target:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_exists)

    result = WriteFileTool().execute(
        {"path": "out.txt", "content": "hello"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.summary == "File write failed: out.txt"
    assert result.error == "File write failed"


def test_write_file_returns_error_when_target_type_stat_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "out.txt"
    target.write_text("", encoding="utf-8")
    original_is_file = Path.is_file

    def fail_is_file(path: Path) -> bool:
        if path == target:
            raise OSError("permission denied")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", fail_is_file)

    result = WriteFileTool().execute(
        {"path": "out.txt", "content": "hello"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.summary == "File write failed: out.txt"
    assert result.error == "File write failed"


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
