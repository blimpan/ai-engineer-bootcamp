from ragu.fingerprint import file_fingerprint
from ragu.ingest import (
    load_parse_manifest,
    needs_parse,
    output_path_for,
    save_parse_manifest,
)


def test_needs_parse_when_markdown_missing(ingest_root):
    pdf_path = ingest_root / "data" / "sample.pdf"
    pdf_path.parent.mkdir()
    pdf_path.write_bytes(b"pdf-v1")

    assert needs_parse(pdf_path, {}) is True


def test_skips_unchanged_pdf_with_matching_fingerprint(ingest_root):
    pdf_path = ingest_root / "data" / "sample.pdf"
    pdf_path.parent.mkdir()
    pdf_path.write_bytes(b"pdf-v1")

    markdown_file = output_path_for(pdf_path)
    markdown_file.write_text("# Parsed", encoding="utf-8")

    manifest = {pdf_path.name: file_fingerprint(pdf_path)}
    assert needs_parse(pdf_path, manifest) is False


def test_reparses_when_pdf_content_changes(ingest_root):
    pdf_path = ingest_root / "data" / "sample.pdf"
    pdf_path.parent.mkdir()
    pdf_path.write_bytes(b"pdf-v1")

    markdown_file = output_path_for(pdf_path)
    markdown_file.write_text("# Parsed", encoding="utf-8")

    manifest = {pdf_path.name: file_fingerprint(pdf_path)}
    pdf_path.write_bytes(b"pdf-v2")

    assert needs_parse(pdf_path, manifest) is True


def test_backfills_legacy_parsed_file_without_manifest_entry(ingest_root):
    pdf_path = ingest_root / "data" / "sample.pdf"
    pdf_path.parent.mkdir()
    pdf_path.write_bytes(b"pdf-v1")

    markdown_file = output_path_for(pdf_path)
    markdown_file.write_text("# Parsed", encoding="utf-8")

    manifest: dict[str, str] = {}
    assert needs_parse(pdf_path, manifest) is False
    assert manifest[pdf_path.name] == file_fingerprint(pdf_path)


def test_parse_manifest_round_trip(ingest_root):
    save_parse_manifest({"sample.pdf": "abc123"})
    assert load_parse_manifest() == {"sample.pdf": "abc123"}
