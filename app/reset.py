from dotenv import load_dotenv

from app.vectorstore import reset_vectorstore


def main():
    load_dotenv()

    reset_vectorstore()


if __name__ == "__main__":
    main()