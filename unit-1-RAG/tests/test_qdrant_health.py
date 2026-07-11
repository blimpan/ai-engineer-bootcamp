import pytest

from ragu.index import ensure_qdrant_running


def test_ensure_qdrant_running_requires_url(monkeypatch):
    monkeypatch.setattr("ragu.index.QDRANT_URL", None)

    with pytest.raises(RuntimeError, match="QDRANT_URL is not set"):
        ensure_qdrant_running()


def test_ensure_qdrant_running_raises_when_unreachable(monkeypatch):
    monkeypatch.setattr("ragu.index.QDRANT_URL", "http://localhost:6333")

    def fail_info():
        raise ConnectionError("connection refused")

    monkeypatch.setattr("ragu.index.qdrant_client.info", fail_info)

    with pytest.raises(RuntimeError, match="Cannot connect to Qdrant"):
        ensure_qdrant_running()


def test_ensure_qdrant_running_succeeds_when_reachable(monkeypatch):
    monkeypatch.setattr("ragu.index.QDRANT_URL", "http://localhost:6333")
    monkeypatch.setattr("ragu.index.qdrant_client.info", lambda: {"version": "1.18.0"})

    ensure_qdrant_running()
