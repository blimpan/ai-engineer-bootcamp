import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from llama_cloud import LlamaCloud

from ragu.fingerprint import file_fingerprint

load_dotenv()


@dataclass
class ParsedDocument:
    source_file: Path
    markdown_file: Path


ROOT = Path(__file__).resolve().parent.parent
PARSE_MANIFEST_PATH = ROOT / "parsed" / ".parse_manifest.json"


def output_path_for(pdf_path: Path) -> Path:
    """Return the output path for a PDF file."""
    return ROOT / "parsed" / f"{pdf_path.stem}.md"


def load_parse_manifest() -> dict[str, str]:
    """Load PDF filename -> content hash mappings from disk."""
    if not PARSE_MANIFEST_PATH.exists():
        return {}

    return json.loads(PARSE_MANIFEST_PATH.read_text(encoding="utf-8"))


def save_parse_manifest(manifest: dict[str, str]) -> None:
    """Persist the parse manifest to disk."""
    PARSE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARSE_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def needs_parse(pdf_path: Path, manifest: dict[str, str]) -> bool:
    """Return True if the PDF must be parsed (missing output or changed content)."""
    markdown_file = output_path_for(pdf_path)
    current_fingerprint = file_fingerprint(pdf_path)
    key = pdf_path.name
    stored_fingerprint = manifest.get(key)

    if not markdown_file.exists():
        return True

    if stored_fingerprint is None:
        # Backfill legacy parsed files so we don't re-bill LlamaParse on first run.
        manifest[key] = current_fingerprint
        return False

    return stored_fingerprint != current_fingerprint


def record_parse(pdf_path: Path, manifest: dict[str, str]) -> None:
    """Store the current PDF fingerprint after a successful parse."""
    manifest[pdf_path.name] = file_fingerprint(pdf_path)


def parse_pdf(client, pdf_path: Path, manifest: dict[str, str]) -> ParsedDocument:
    """Parse a PDF file into a Markdown file."""

    if not needs_parse(pdf_path, manifest):
        print(f"Skipping {pdf_path.name} (unchanged)")
        return ParsedDocument(source_file=pdf_path, markdown_file=output_path_for(pdf_path))

    try:
        uploaded_file = client.files.create(file=pdf_path, purpose="parse")
        result = client.parsing.parse(
            file_id=uploaded_file.id,
            tier="agentic",
            version="latest",
            expand=["markdown"],
        )
        markdown_content = "\n\n".join(
            md
            for page in result.markdown.pages
            if (md := getattr(page, "markdown", None))
        )
        output_path_for(pdf_path).write_text(markdown_content, encoding="utf-8")
        record_parse(pdf_path, manifest)
        print(f"Parsed {pdf_path.name}")
        return ParsedDocument(source_file=pdf_path, markdown_file=output_path_for(pdf_path))
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")


def ingest_directory(client: LlamaCloud, dir_path: Path) -> list[ParsedDocument]:
    """Ingest all PDF files in a directory into a list of ParsedDocuments."""
    manifest = load_parse_manifest()
    parsed_documents = []
    files_to_parse = list(dir_path.glob("*.pdf"))

    for pdf_path in files_to_parse:
        try:
            parsed_documents.append(parse_pdf(client, pdf_path, manifest))
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {e}")
        finally:
            print(f"{len(parsed_documents)} document(s) processed out of {len(files_to_parse)}")

    save_parse_manifest(manifest)
    return parsed_documents


if __name__ == "__main__":
    llama_client = LlamaCloud(base_url="https://api.cloud.eu.llamaindex.ai")
    parsed_documents = ingest_directory(llama_client, Path(ROOT / "data"))
    print(f"Parsed {len(parsed_documents)} documents from {Path(ROOT / 'data')}")
