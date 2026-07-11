from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config):
    """Fail fast with a helpful message when tests run outside the project venv."""
    if not (PROJECT_ROOT / ".venv").exists():
        return

    venv_path = (PROJECT_ROOT / ".venv").resolve()
    if not Path(sys.prefix).resolve().is_relative_to(venv_path):
        pytest.exit(
            "Tests must run with the unit-1-RAG virtual environment.\n"
            "Your shell is using a different pytest (common with conda + venv).\n"
            "Use one of:\n"
            "  python -m pytest\n"
            "  .venv/bin/pytest",
            returncode=1,
        )


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def ingest_root(tmp_path, monkeypatch) -> Path:
    """Point ingest paths at a temporary directory."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    monkeypatch.setattr("ragu.ingest.ROOT", tmp_path)
    monkeypatch.setattr("ragu.ingest.PARSE_MANIFEST_PATH", parsed_dir / ".parse_manifest.json")
    return tmp_path


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Point index manifest paths at a temporary directory."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    monkeypatch.setattr("ragu.index.INDEX_MANIFEST_PATH", parsed_dir / ".index_manifest.json")
    return tmp_path
