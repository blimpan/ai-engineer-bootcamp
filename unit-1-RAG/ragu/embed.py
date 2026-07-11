from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI


client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_text(text: str) -> list[float]:
    """Embed text using OpenAI's text-embedding-3-small model."""
    try:
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL,
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error embedding text: {e}")
        return []


if __name__ == "__main__":
    text = "Hello, world!"
    embedding = embed_text(text)
    print(embedding)