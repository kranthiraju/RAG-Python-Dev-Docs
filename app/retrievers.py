import pickle

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from app.vectorstore import get_vectorstore


CHUNKS_FILE = "data/chunks.pkl"


def load_chunks():

    with open(
        CHUNKS_FILE,
        "rb",
    ) as f:
        return pickle.load(f)


def create_vector_retriever():

    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
        },
    )


def create_bm25_retriever():

    chunks = load_chunks()

    retriever = BM25Retriever.from_documents(
        chunks
    )

    retriever.k = 5

    return retriever


def create_ensemble_retriever():

    vector_retriever = create_vector_retriever()

    bm25_retriever = create_bm25_retriever()

    ensemble = EnsembleRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever,
        ],
        weights=[
            0.7,
            0.3,
        ],
    )

    return ensemble