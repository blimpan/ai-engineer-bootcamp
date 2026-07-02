from pathlib import Path
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


def chunk_text_sliding_window(text: str, source_file: Path, chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
    """Chunk text into chunks using a sliding window approach."""
    tokens = encoder.encode(text)
    stride = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    current_section_title = None
    for i, start in enumerate(range(0, len(tokens), stride)):
        window = tokens[start:start + chunk_size]
        if not window:
            break
        chunk_text = encoder.decode(window)

        for line in chunk_text.splitlines(): # not ideal since there can be multiple section titles in the same chunk
            if line.startswith("## "):
                current_section_title = line[3:]
                break

        chunks.append(Chunk(text=chunk_text, source_file=source_file, chunk_index=i, section_title=current_section_title))
    return chunks


def chunk_markdown_file(markdown_file: Path) -> list[Chunk]:
    """Chunk a markdown file into chunks."""
    markdown_content = markdown_file.read_text(encoding="utf-8")
    return chunk_text_sliding_window(markdown_content, markdown_file)


if __name__ == "__main__":
    chunks = chunk_markdown_file(Path("parsed/No evidence that accomodating learning preferences improves learning.md"))
    print(chunks[0])
    print(chunks[-1])
    print(f"Number of tokens in chunk 0: {len(encoder.encode(chunks[0].text))}")
    print(f"Number of tokens in chunk 1: {len(encoder.encode(chunks[1].text))}")
    print(f"Total number of chunks: {len(chunks)}")