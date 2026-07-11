import hashlib
from pathlib import Path


def file_fingerprint(path: Path) -> str:
    """Return a SHA-256 hex digest of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
