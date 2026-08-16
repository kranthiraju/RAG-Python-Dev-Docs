import os

from langchain_ollama import ChatOllama

from langchain_core.output_parsers import (
    StrOutputParser,
)

from langchain_core.runnables import (
    RunnablePassthrough,
)

from app.prompts import RAG_PROMPT


def format_documents(documents):

    return "\n\n".join(
        document.page_content
        for document in documents
    )


def create_rag_chain(retriever):

    model = os.environ["LLM_MODEL"]

    print(
        f"LLM_MODEL = {model}"
    )

    llm = ChatOllama(
        model=model,
        temperature=0,
    )

    chain = (
        {
            "context": (
                retriever
                | format_documents
            ),
            "question": (
                RunnablePassthrough()
            ),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain