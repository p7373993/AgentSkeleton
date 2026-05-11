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
    raw_path = Path(requested_path)
    if raw_path.is_absolute() or raw_path.drive:
        raise PathSecurityError(
            f"Path must be workspace-relative: {requested_path}"
        )
    if _contains_windows_reserved_basename(raw_path):
        raise PathSecurityError(
            f"Path contains reserved Windows device name: {requested_path}"
        )
    if _contains_ambiguous_windows_path_part(raw_path):
        raise PathSecurityError(
            f"Path contains ambiguous Windows path part: {requested_path}"
        )

    root = workspace.expanduser().resolve()
    candidate = (root / requested_path).resolve(strict=False)

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PathSecurityError(
            f"Path escapes workspace: {requested_path}"
        ) from exc

    return candidate


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
