from dotenv import load_dotenv

from app.retrievers import (
    create_ensemble_retriever,
)

from app.pipeline import (
    create_rag_chain,
)


def main():

    load_dotenv()

    # -------------------------
    # 1. Create retriever
    # -------------------------

    retriever = (
        create_ensemble_retriever()
    )

    # -------------------------
    # 2. Create RAG chain
    # -------------------------

    rag_chain = create_rag_chain(
        retriever
    )

    # -------------------------
    # 3. Interactive questions
    # -------------------------

    while True:

        question = input(
            "\nAsk a question "
            "(type 'exit' to quit): "
        )

        if question.lower() in {
            "exit",
            "quit",
        }:
            print("Goodbye!")
            break

        if not question.strip():
            continue

        print("\nThinking...\n")

        answer = rag_chain.invoke(
            question
        )

        print("Answer:")
        print(answer)


if __name__ == "__main__":
    main()