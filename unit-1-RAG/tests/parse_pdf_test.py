from pathlib import Path
from dotenv import load_dotenv
import sys
from pathlib import Path

from httpx import delete

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragu.ingest import parse_pdf

load_dotenv()

from llama_cloud import LlamaCloud

ROOT = Path(__file__).parent.parent
pdf_path = ROOT / "data" / "Knowledge about neuroscience doesnt protect teachers from myths.pdf"
out_path = ROOT / "parsed" / "Knowledge about neuroscience doesnt protect teachers from myths.md"
out_path.parent.mkdir(exist_ok=True)

if out_path.exists():
    out_path.unlink()


llama_client = LlamaCloud(base_url="https://api.cloud.eu.llamaindex.ai")
parsed_document = parse_pdf(llama_client, pdf_path)

assert parsed_document.markdown_file.exists()
assert parsed_document.markdown_file.read_text(encoding="utf-8").strip() != ""
print("PDF parsing test passed")