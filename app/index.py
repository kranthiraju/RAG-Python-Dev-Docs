import pickle
from pathlib import Path

from dotenv import load_dotenv

from app.ingestion import (
    load_documents,
    split_documents,
)

from app.vectorstore import (
    create_vectorstore,
)


CHUNKS_FILE = "data/chunks.pkl"


def main():

    load_dotenv()

    documents = load_documents()

    print(
        f"Loaded {len(documents)} documents"
    )

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    # Save chunks for BM25
    Path("data").mkdir(
        exist_ok=True
    )

    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print(
        f"Saved chunks to {CHUNKS_FILE}"
    )

    # Create vector database
    create_vectorstore(
        chunks,
        batch_size=1000,
    )

    print("\nIndexing completed!")


if __name__ == "__main__":
    main()