from pathlib import Path


class PathSecurityError(ValueError):
    """Raised when a requested path escapes the configured workspace."""


def resolve_workspace_path(workspace: Path, requested_path: str) -> Path:
    raw_path = Path(requested_path)
    if raw_path.is_absolute() or raw_path.drive:
        raise PathSecurityError(
            f"Path must be workspace-relative: {requested_path}"
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
