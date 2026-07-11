import os

import pytest
from llama_cloud import LlamaCloud

from ragu.ingest import parse_pdf


SAMPLE_PDF = "Knowledge about neuroscience doesnt protect teachers from myths.pdf"


@pytest.mark.integration
def test_parse_pdf_produces_nonempty_markdown(project_root, ingest_root):
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        pytest.skip("LLAMA_CLOUD_API_KEY is not configured")

    source_pdf = project_root / "data" / SAMPLE_PDF
    if not source_pdf.exists():
        pytest.skip(f"Sample PDF not found: {source_pdf}")

    pdf_path = ingest_root / "data" / SAMPLE_PDF
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(source_pdf.read_bytes())

    llama_client = LlamaCloud(base_url="https://api.cloud.eu.llamaindex.ai")
    parsed_document = parse_pdf(llama_client, pdf_path, manifest={})

    assert parsed_document.markdown_file.exists()
    assert parsed_document.markdown_file.read_text(encoding="utf-8").strip() != ""
