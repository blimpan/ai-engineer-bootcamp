from dataclasses import dataclass
from pathlib import Path
from llama_cloud import LlamaCloud
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ParsedDocument:
    source_file: Path
    markdown_file: Path


ROOT = Path(__file__).resolve().parent.parent

def output_path_for(pdf_path: Path) -> Path:
    """Return the output path for a PDF file."""
    return ROOT / 'parsed' / f'{pdf_path.stem}.md'


def should_skip(pdf_path: Path) -> bool:
    """Return True if the PDF should be skipped."""
    return output_path_for(pdf_path).exists()


def parse_pdf(client, pdf_path: Path) -> ParsedDocument:
    """Parse a PDF file into a Markdown file."""

    if should_skip(pdf_path):
        print(f"Skipping {pdf_path} because it already has a parsed file")
        return ParsedDocument(source_file=pdf_path, markdown_file=output_path_for(pdf_path))
    
    try:
        uploaded_file = client.files.create(file=pdf_path, purpose="parse")
        result = client.parsing.parse(
            file_id=uploaded_file.id,
            tier="agentic",
            version="latest",
            expand=["markdown"]
        )
        markdown_content = "\n\n".join(
            md
            for page in result.markdown.pages
            if (md := getattr(page, "markdown", None))
        )
        output_path_for(pdf_path).write_text(markdown_content, encoding="utf-8")
        return ParsedDocument(source_file=pdf_path, markdown_file=output_path_for(pdf_path))
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")


def ingest_directory(client, dir_path: Path) -> list[ParsedDocument]:
    """Ingest all PDF files in a directory into a list of ParsedDocuments."""
    parsed_documents = []
    files_to_parse = list(dir_path.glob("*.pdf"))
    for pdf_path in files_to_parse:
        try:
            parsed_documents.append(parse_pdf(client, pdf_path))
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {e}")
        finally:
            print(f"{len(parsed_documents)} document(s) parsed out of {len(files_to_parse)}")
    return parsed_documents


if __name__ == "__main__":
    llama_client = LlamaCloud(base_url="https://api.cloud.eu.llamaindex.ai")
    parsed_documents = ingest_directory(llama_client, Path(ROOT / "data"))
    print(f"Parsed {len(parsed_documents)} documents from {Path(ROOT / 'data')}")