import os
from pathlib import Path
from uuid import UUID, uuid4, uuid5
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, ScoredPoint, VectorParams, Distance
from ragu.chunking import Chunk
from ragu.embed import embed_text


QDRANT_URL = os.getenv("QDRANT_URL")
UUID_NAMESPACE = UUID(os.getenv("UUID_NAMESPACE"))
qdrant_client = QdrantClient(url=QDRANT_URL)


class ScoredChunk(Chunk):
    score: float

    def __init__(self, chunk: Chunk, score: float):
        super().__init__(chunk.text, chunk.source_file, chunk.chunk_index, chunk.section_title)
        self.score = score

    def __str__(self) -> str:
        return f"<< Text >>\n{self.text}\n<< Source File >>\n{self.source_file}\n<< Chunk Index >>\n{self.chunk_index}\n<< Section Title >>\n{self.section_title}\n<< Score >>\n{self.score}"


def create_collection(collection_name: str):

    if qdrant_client.collection_exists(collection_name):
        print(f"Collection {collection_name} already exists")
        return

    try:
        qdrant_client.create_collection(
        collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    except Exception as e:
        print(f"Error creating collection: {e}")


def chunk_to_point(chunk: Chunk) -> PointStruct:
    return PointStruct(
        id = uuid5(UUID_NAMESPACE, chunk.text),
        vector=embed_text(chunk.text),
        payload={
            "source_file": str(chunk.source_file),
            "chunk_index": chunk.chunk_index,
            "section_title": chunk.section_title,
            "chunk_text": chunk.text
        }
    )


def upsert_points(collection_name: str, points: list[PointStruct]):
    """Upsert points into a collection."""

    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
    except Exception as e:
        print(f"Error upserting points: {e}")


def search_points(collection_name: str, query: str, top_k: int = 10) -> list[ScoredChunk]:
    """Search for points in a collection."""
    try:
        query_vector = embed_text(query)
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        scored_chunks: list[ScoredChunk] = []
        for point in results.points:
            if not point.payload:
                continue

            chunk = Chunk(
                point.payload["chunk_text"],
                Path(point.payload["source_file"]),
                point.payload["chunk_index"],
                point.payload["section_title"]
            )

            scored_chunks.append(ScoredChunk(chunk, point.score))
        return scored_chunks

    except Exception as e:
        print(f"Error searching points: {e}")
        return []