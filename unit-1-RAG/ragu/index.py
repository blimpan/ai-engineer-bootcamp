import json
import os
from pathlib import Path
from uuid import UUID, uuid5

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)
from ragu.chunking import CHUNK_OVERLAP, CHUNK_SIZE, CHUNK_STRATEGY, Chunk
from ragu.embed import EMBEDDING_MODEL, embed_text
from ragu.fingerprint import file_fingerprint

QDRANT_COLLECTION_NAME = "ragu-files"
QDRANT_URL = os.getenv("QDRANT_URL")
UUID_NAMESPACE = UUID(os.getenv("UUID_NAMESPACE"))
ROOT = Path(__file__).resolve().parent.parent
INDEX_MANIFEST_PATH = ROOT / "parsed" / ".index_manifest.json"
qdrant_client = QdrantClient(url=QDRANT_URL)


class ScoredChunk(Chunk):
    score: float

    def __init__(self, chunk: Chunk, score: float):
        super().__init__(chunk.text, chunk.source_file, chunk.chunk_index, chunk.section_title)
        self.score = score

    def __str__(self) -> str:
        return f"<< Text >>\n{self.text[:100]}...\n<< Source File >>\n{self.source_file}\n<< Chunk Index >>\n{self.chunk_index}\n<< Section Title >>\n{self.section_title}\n<< Score >>\n{self.score}"


def ensure_qdrant_running() -> None:
    """Verify Qdrant is reachable before starting the pipeline."""
    if not QDRANT_URL:
        raise RuntimeError(
            "QDRANT_URL is not set. Add it to your .env file (e.g. http://localhost:6333)."
        )

    try:
        qdrant_client.info()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Qdrant at {QDRANT_URL}. "
            "Start it with: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant"
        ) from exc


def indexing_fingerprint(markdown_file: Path) -> str:
    """Fingerprint parsed content plus all settings that affect chunk vectors."""
    content_hash = file_fingerprint(markdown_file)
    return (
        f"{content_hash}:{CHUNK_SIZE}:{CHUNK_OVERLAP}:"
        f"{CHUNK_STRATEGY}:{EMBEDDING_MODEL}"
    )


def load_index_manifest() -> dict[str, str]:
    """Load markdown filename -> indexing fingerprint mappings from disk."""
    if not INDEX_MANIFEST_PATH.exists():
        return {}

    return json.loads(INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))


def save_index_manifest(manifest: dict[str, str]) -> None:
    """Persist the index manifest to disk."""
    INDEX_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _source_file_filter(source_file: Path) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="source_file",
                match=MatchValue(value=str(source_file)),
            )
        ]
    )


def has_points_for_source(collection_name: str, source_file: Path) -> bool:
    """Return True if Qdrant already contains points for this markdown file."""
    if not qdrant_client.collection_exists(collection_name):
        return False

    result = qdrant_client.count(
        collection_name=collection_name,
        count_filter=_source_file_filter(source_file),
    )
    return result.count > 0


def needs_index(
    markdown_file: Path,
    manifest: dict[str, str],
    collection_name: str,
) -> bool:
    """Return True if the markdown file must be chunked and embedded."""
    current_fingerprint = indexing_fingerprint(markdown_file)
    key = markdown_file.name
    stored_fingerprint = manifest.get(key)

    if stored_fingerprint == current_fingerprint:
        return False

    if stored_fingerprint is None and has_points_for_source(collection_name, markdown_file):
        manifest[key] = current_fingerprint
        return False

    return True


def record_index(markdown_file: Path, manifest: dict[str, str]) -> None:
    """Store the current indexing fingerprint after a successful upsert."""
    manifest[markdown_file.name] = indexing_fingerprint(markdown_file)


def create_collection(collection_name: str):
    if qdrant_client.collection_exists(collection_name):
        print(f"Collection {collection_name} already exists")
        return

    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    except Exception as e:
        print(f"Error creating collection: {e}")


def delete_points_for_source(collection_name: str, source_file: Path) -> None:
    """Remove stale vectors for a markdown file before re-indexing it."""
    if not has_points_for_source(collection_name, source_file):
        return

    try:
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=_source_file_filter(source_file)),
        )
    except Exception as e:
        print(f"Error deleting points for {source_file.name}: {e}")


def chunk_to_point(chunk: Chunk) -> PointStruct:
    return PointStruct(
        id=uuid5(UUID_NAMESPACE, chunk.text),
        vector=embed_text(chunk.text),
        payload={
            "source_file": str(chunk.source_file),
            "chunk_index": chunk.chunk_index,
            "section_title": chunk.section_title,
            "chunk_text": chunk.text,
        },
    )


def upsert_points(collection_name: str, points: list[PointStruct]):
    """Upsert points into a collection."""
    if not points:
        return

    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
        )
    except Exception as e:
        print(f"Error upserting points: {e}")


def search_points(collection_name: str, query: str, top_k: int = 3, score_threshold: float | None = None) -> list[ScoredChunk]:
    """Search for points in a collection."""
    try:
        query_vector = embed_text(query)
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        )

        scored_chunks: list[ScoredChunk] = []
        for point in results.points:
            if not point.payload:
                continue

            chunk = Chunk(
                point.payload["chunk_text"],
                Path(point.payload["source_file"]),
                point.payload["chunk_index"],
                point.payload["section_title"],
            )

            scored_chunks.append(ScoredChunk(chunk, point.score))
        return scored_chunks

    except Exception as e:
        print(f"Error searching points: {e}")
        return []
