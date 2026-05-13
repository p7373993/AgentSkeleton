from pathlib import Path

_WINDOWS_RESERVED_PATH_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class PathSecurityError(ValueError):
    """Raised when a requested path escapes the configured workspace."""


def resolve_workspace_path(workspace: Path, requested_path: str) -> Path:
    path_text = _inspect_requested_path(requested_path)
    raw_path = Path(path_text)
    if raw_path.is_absolute() or raw_path.drive:
        raise PathSecurityError(f"Path must be workspace-relative: {path_text}")
    if _contains_windows_reserved_basename(raw_path):
        raise PathSecurityError(
            f"Path contains reserved Windows device name: {path_text}"
        )
    if _contains_ambiguous_windows_path_part(raw_path):
        raise PathSecurityError(
            f"Path contains ambiguous Windows path part: {path_text}"
        )

    root = workspace.expanduser().resolve()
    candidate = (root / path_text).resolve(strict=False)

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PathSecurityError(f"Path escapes workspace: {path_text}") from exc

    return candidate


def _inspect_requested_path(requested_path: str) -> str:
    try:
        path_text = str(requested_path)
        path_text.encode("utf-8")
    except Exception as exc:
        raise PathSecurityError("Path could not be inspected") from exc
    return path_text


def _contains_windows_reserved_basename(path: Path) -> bool:
    for part in path.parts:
        base_name = part.split(".", 1)[0].rstrip(" ").upper()
        if base_name in _WINDOWS_RESERVED_PATH_BASENAMES:
            return True
    return False


def _contains_ambiguous_windows_path_part(path: Path) -> bool:
    return any(
        ":" in part or part.endswith(" ") or part.endswith(".") for part in path.parts
    )
