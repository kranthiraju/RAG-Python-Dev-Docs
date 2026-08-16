# app/ingestion.py

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(
    data_dir: str = "data/documents",
):
    documents = []

    for file_path in Path(data_dir).rglob("*.txt"):

        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = str(file_path)
            doc.metadata["filename"] = file_path.name
            doc.metadata["folder"] = file_path.parent.name

        documents.extend(docs)

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    return splitter.split_documents(documents)