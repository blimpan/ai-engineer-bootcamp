from pathlib import Path

from llama_cloud import LlamaCloud

from ragu.chunking import chunk_markdown_file
from ragu.ingest import ingest_directory
from ragu.index import (
    QDRANT_COLLECTION_NAME,
    chunk_to_point,
    create_collection,
    delete_points_for_source,
    ensure_qdrant_running,
    load_index_manifest,
    needs_index,
    record_index,
    save_index_manifest,
    search_points,
    upsert_points,
)


ROOT = Path(__file__).parent.parent


def ensured_indexed():
    """Make sure documents are parsed, chunked, and stored in Qdrant."""
    create_collection(QDRANT_COLLECTION_NAME)

    llama_client = LlamaCloud(base_url="https://api.cloud.eu.llamaindex.ai")
    parsed_documents = ingest_directory(llama_client, Path(ROOT / "data"))
    if not parsed_documents:
        print("No parsed documents found")
        raise ValueError("No parsed documents found")

    manifest = load_index_manifest()

    for parsed_document in parsed_documents:
        markdown_file = parsed_document.markdown_file
        try:
            if not needs_index(markdown_file, manifest, QDRANT_COLLECTION_NAME):
                print(f"Skipping index for {markdown_file.name} (unchanged)")
                continue

            delete_points_for_source(QDRANT_COLLECTION_NAME, markdown_file)
            chunks = chunk_markdown_file(markdown_file)
            points = [chunk_to_point(chunk) for chunk in chunks]
            upsert_points(collection_name=QDRANT_COLLECTION_NAME, points=points)
            record_index(markdown_file, manifest)
            print(f"Indexed {markdown_file.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"Error chunking or upserting {markdown_file.name}: {e}")

    save_index_manifest(manifest)


def query_loop():
    """Take user input and return relevant chunks."""
    while True:
        user_input = input("Enter a query: ")
        if user_input.lower() == "exit":
            break
        chunks = search_points(collection_name=QDRANT_COLLECTION_NAME, query=user_input, score_threshold=0.2)
        for chunk in chunks:
            print(chunk, "\n")


def main():
    ensure_qdrant_running()
    ensured_indexed()
    query_loop()


if __name__ == "__main__":
    main()
