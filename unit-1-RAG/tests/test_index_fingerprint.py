from ragu.chunking import CHUNK_OVERLAP, CHUNK_SIZE, CHUNK_STRATEGY
from ragu.embed import EMBEDDING_MODEL
from ragu.fingerprint import file_fingerprint
from ragu.index import (
    indexing_fingerprint,
    load_index_manifest,
    needs_index,
    save_index_manifest,
)


def test_indexing_fingerprint_includes_chunk_and_embed_config(tmp_path):
    markdown_file = tmp_path / "sample.md"
    markdown_file.write_text("# Title\n\nBody", encoding="utf-8")

    expected = (
        f"{file_fingerprint(markdown_file)}:{CHUNK_SIZE}:{CHUNK_OVERLAP}:"
        f"{CHUNK_STRATEGY}:{EMBEDDING_MODEL}"
    )
    assert indexing_fingerprint(markdown_file) == expected


def test_needs_index_when_markdown_content_changes(tmp_path):
    markdown_file = tmp_path / "sample.md"
    markdown_file.write_text("# Title\n\nBody", encoding="utf-8")
    manifest = {markdown_file.name: indexing_fingerprint(markdown_file)}

    markdown_file.write_text("# Title\n\nUpdated body", encoding="utf-8")
    assert needs_index(markdown_file, manifest, "ragu-files") is True


def test_skips_index_when_fingerprint_matches(tmp_path):
    markdown_file = tmp_path / "sample.md"
    markdown_file.write_text("# Title\n\nBody", encoding="utf-8")
    manifest = {markdown_file.name: indexing_fingerprint(markdown_file)}

    assert needs_index(markdown_file, manifest, "ragu-files") is False


def test_backfills_legacy_index_when_points_already_exist(tmp_path, monkeypatch):
    markdown_file = tmp_path / "sample.md"
    markdown_file.write_text("# Title\n\nBody", encoding="utf-8")
    manifest: dict[str, str] = {}

    monkeypatch.setattr("ragu.index.has_points_for_source", lambda *_args, **_kwargs: True)

    assert needs_index(markdown_file, manifest, "ragu-files") is False
    assert manifest[markdown_file.name] == indexing_fingerprint(markdown_file)


def test_index_manifest_round_trip(index_root):
    save_index_manifest({"sample.md": "abc123"})
    assert load_index_manifest() == {"sample.md": "abc123"}
