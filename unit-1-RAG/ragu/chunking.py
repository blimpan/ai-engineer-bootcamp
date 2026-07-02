from pathlib import Path
import re
from tiktoken import get_encoding
from dataclasses import dataclass


encoder = get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    source_file: Path
    chunk_index: int
    section_title: str | None = None

    def __str__(self) -> str:
        return f"<< Text >>\n{self.text}\n<< Source File >>\n{self.source_file}\n<< Chunk Index >>\n{self.chunk_index}\n<< Section Title >>\n{self.section_title}"


def get_section_boundaries(text: str) -> list[tuple[int, str]]:
    """Get the boundaries of the sections in the text."""
    boundaries: list[tuple[int, str]] = []

    for match in re.finditer(r"## (.*)", text, re.MULTILINE):
        title = match.group(1).strip()
        token_index = len(encoder.encode(text[:match.start()]))
        boundaries.append((token_index, title))
    return boundaries


def chunk_text_sliding_window(text: str, source_file: Path, chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
    """Chunk text into chunks using a sliding window approach."""
    tokens = encoder.encode(text)
    stride = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    section_boundaries = get_section_boundaries(text)

    for window_index, window_start in enumerate(range(0, len(tokens), stride)):
        window = tokens[window_start:window_start + chunk_size]
        if not window:
            break
        chunk_text = encoder.decode(window)

        current_section_title = None
        for section_start, section_title in reversed(section_boundaries):
            if section_start < window_start + chunk_size:
                current_section_title = section_title
                break
        
        chunks.append(Chunk(text=chunk_text, source_file=source_file, chunk_index=window_index, section_title=current_section_title))
    return chunks


def chunk_text_by_section_title(text: str, source_file: Path) -> list[Chunk]:
    """Chunk text into chunks by section title."""
    boundaries = get_section_boundaries(text)
    tokens = encoder.encode(text)
    chunks: list[Chunk] = []
    for i, (section_start, section_title) in enumerate(boundaries):
        if i == len(boundaries) - 1:
            chunk_text = encoder.decode(tokens[section_start:])
        else:
            next_section_start, _ = boundaries[i + 1]
            chunk_text = encoder.decode(tokens[section_start:next_section_start])
        chunks.append(Chunk(text=chunk_text, source_file=source_file, chunk_index=i, section_title=section_title))
    return chunks


def chunk_markdown_file(markdown_file: Path) -> list[Chunk]:
    """Chunk a markdown file into chunks."""
    markdown_content = markdown_file.read_text(encoding="utf-8")
    return chunk_text_sliding_window(markdown_content, markdown_file)


if __name__ == "__main__":
    markdown_file = Path("parsed/No evidence that accomodating learning preferences improves learning.md")
    markdown_content = markdown_file.read_text(encoding="utf-8")
    chunks: list[Chunk] = []

    chunk_type = "section_title"
    if chunk_type == "sliding_window":
        chunks = chunk_text_sliding_window(markdown_content, markdown_file)
    elif chunk_type == "section_title":
        chunks = chunk_text_by_section_title(markdown_content, markdown_file)

    print(chunks[0])
    print(chunks[-1])
    print(f"Number of tokens in chunk 0: {len(encoder.encode(chunks[0].text))}")
    print(f"Number of tokens in chunk 1: {len(encoder.encode(chunks[1].text))}")
    print(f"Total number of chunks: {len(chunks)}")