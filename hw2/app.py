"""NotebookLM-like RAG application.

Loads documents from a folder into a Chroma vector store and answers questions
about them with a retrieval-augmented generation (RAG) chain.

Based on 02_LangChain/07_RAG/07_rag_loaddb.py (loading) and 09_rag_query.py
(querying), using Gemini embeddings and chat model via langchain-google-genai.

Usage:
    uv run python hw2/app.py [--data-dir PATH] [--reload]
"""

import argparse
import os
import readline  # noqa: F401  (enables arrow-key line editing in input())
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFDirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

HW_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HW_DIR.parent / "02_LangChain" / "07_RAG" / "rag_data"
PERSIST_DIR = HW_DIR / ".chromadb"
EMBEDDING_MODEL = "models/gemini-embedding-001"

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""
)


def create_vectorstore(persist_dir=PERSIST_DIR):
    """Open (or create) the persistent Chroma vector store with Gemini embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(embedding_function=embeddings, persist_directory=str(persist_dir))


def load_folder(directory):
    """Load all supported files (txt, pdf, docx, md, csv) from a folder into Documents."""
    directory = str(directory)
    loaders = [
        DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
        PyPDFDirectoryLoader(directory),
        DirectoryLoader(directory, glob="**/*.docx", loader_cls=Docx2txtLoader),
        DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        DirectoryLoader(directory, glob="**/*.csv", loader_cls=CSVLoader),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def add_documents(vectorstore, docs, chunk_size=1000, chunk_overlap=150):
    """Split documents into chunks and store their embeddings in the vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    if splits:
        vectorstore.add_documents(documents=splits)
    return len(splits)


def list_sources(vectorstore):
    """Return the sorted set of unique document sources stored in the vector store."""
    metadatas = vectorstore.get(include=["metadatas"])["metadatas"]
    return sorted({metadata.get("source", "unknown") for metadata in metadatas})


def format_docs(docs):
    """Join retrieved document contents into one prompt context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore):
    """Build the chain: retrieve context, fill the prompt, call the LLM, parse text."""
    llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL"))
    retriever = vectorstore.as_retriever()
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


def question_loop(rag_chain):
    """Prompt the user for questions and print answers until an empty line is entered."""
    while True:
        question = input("llm>> ").strip()
        if not question:
            break
        print(rag_chain.invoke(question))


def parse_args():
    """Parse command-line arguments for the data folder and reload option."""
    parser = argparse.ArgumentParser(description="Ask questions about documents in a folder.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="folder of documents to index")
    parser.add_argument("--reload", action="store_true", help="clear the database and re-index the folder")
    return parser.parse_args()


def main():
    """Index the document folder if needed, then answer questions from the command line."""
    load_dotenv()
    args = parse_args()
    vectorstore = create_vectorstore()

    if args.reload:
        vectorstore.reset_collection()
    if args.reload or not list_sources(vectorstore):
        print(f"Indexing documents from: {args.data_dir}")
        count = add_documents(vectorstore, load_folder(args.data_dir))
        print(f"Stored {count} chunks.")

    print("Welcome to my RAG application. Ask a question about these documents (empty line to quit):")
    for source in list_sources(vectorstore):
        print(f"  {source}")
    question_loop(build_rag_chain(vectorstore))


if __name__ == "__main__":
    main()
