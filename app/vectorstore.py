# app/vectorstore.py

import os

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

COLLECTION_NAME = "rag_documents"


def get_embeddings():

    model = os.environ["EMBEDDINGS_MODEL"]

    print(f"EMBEDDINGS_MODEL = {model}")

    return OllamaEmbeddings(
        model=model
    )


def create_vectorstore(
    documents,
    batch_size: int = 100,
):
    """
    Create PostgreSQL vector store and
    add documents in batches.
    """

    embeddings = get_embeddings()

    connection = os.environ["POSTGRES_CONNECTION"]

    total = len(documents)

    print(f"Total chunks = {total}")
    print(f"Batch size = {batch_size}")

    vectorstore = None

    for start in range(0, total, batch_size):

        end = min(
            start + batch_size,
            total,
        )

        batch = documents[start:end]

        print(
            f"\nEmbedding batch "
            f"{start + 1}-{end} / {total}"
        )

        if vectorstore is None:

            vectorstore = PGVector.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                connection=connection,
                use_jsonb=True,
            )

        else:

            vectorstore.add_documents(
                batch
            )

        print(
            f"Completed {end}/{total}"
        )

    return vectorstore


def get_vectorstore():

    """
    Connect to an existing vector store.

    No documents are embedded here.
    """

    embeddings = get_embeddings()

    connection = os.environ["POSTGRES_CONNECTION"]

    vectorstore = PGVector(
        collection_name=COLLECTION_NAME,
        connection=connection,
        embeddings=embeddings,
        use_jsonb=True,
    )

    return vectorstore


def reset_vectorstore():
    connection = os.environ["POSTGRES_CONNECTION"]

    engine = create_engine(connection)

    with engine.begin() as conn:
        conn.execute(
            text(
                "DROP TABLE IF EXISTS "
                "langchain_pg_embedding CASCADE"
            )
        )

        conn.execute(
            text(
                "DROP TABLE IF EXISTS "
                "langchain_pg_collection CASCADE"
            )
        )

    print("Vector store reset successfully.")