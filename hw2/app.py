"""
Uzi Mermelstein
Z23806462

NotebookLM-like RAG application.

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
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFDirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

HW_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HW_DIR.parent / "02_LangChain" / "07_RAG" / "rag_data"
PERSIST_DIR = HW_DIR / ".chromadb"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBED_BATCH_SIZE = 40   # chunks per batch (free tier allows 100 embed requests/minute)
EMBED_BATCH_PAUSE = 30  # seconds to wait between batches, and before retrying a rate-limited batch

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
    """Load every supported file (txt, pdf, docx, md, csv, pptx, html) in a folder."""
    directory = str(directory)
    loaders = [
        DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
        PyPDFDirectoryLoader(directory),
        DirectoryLoader(directory, glob="**/*.docx", loader_cls=Docx2txtLoader),
        DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        DirectoryLoader(directory, glob="**/*.csv", loader_cls=CSVLoader),
        DirectoryLoader(directory, glob="**/*.pptx", loader_cls=UnstructuredPowerPointLoader),
        DirectoryLoader(directory, glob="**/*.htm*", loader_cls=BSHTMLLoader),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def load_file(path):
    """Load a single local file, choosing the loader that matches its extension."""
    path = Path(path)
    loaders = {
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".csv": CSVLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".html": BSHTMLLoader,
        ".htm": BSHTMLLoader,
    }
    loader_cls = loaders.get(path.suffix.lower())
    if loader_cls is None:
        raise ValueError(f"unsupported file type: {path.suffix or path.name}")
    return loader_cls(str(path)).load()


def load_webpage(url):
    """Load the readable text of a web page into Documents."""
    return WebBaseLoader(url).load()


def youtube_video_id(url):
    """Return the video id from a YouTube URL, or None if the URL is not YouTube."""
    parts = urlparse(url)
    if parts.hostname in ("youtu.be", "www.youtu.be"):
        return parts.path.lstrip("/") or None
    if parts.hostname and parts.hostname.endswith("youtube.com"):
        return parse_qs(parts.query).get("v", [None])[0]
    return None


def load_youtube(video_id):
    """Load a YouTube video transcript as a single Document."""
    transcript = YouTubeTranscriptApi().fetch(video_id)
    text = " ".join(entry.text for entry in transcript)
    return [Document(page_content=text, metadata={"source": f"youtube:{video_id}"})]


def load_source(source):
    """Load a folder, file, web page, or YouTube video, chosen by what the source looks like."""
    if source.startswith(("http://", "https://")):
        video_id = youtube_video_id(source)
        return load_youtube(video_id) if video_id else load_webpage(source)
    path = Path(source).expanduser()
    if path.is_dir():
        return load_folder(path)
    if path.is_file():
        return load_file(path)
    raise ValueError(f"no such file or folder: {source}")


def add_batch_with_retry(vectorstore, batch, max_retries=5):
    """Add one batch of chunks, waiting and retrying if the embedding rate limit is hit."""
    for attempt in range(max_retries + 1):
        try:
            vectorstore.add_documents(documents=batch)
            return
        except GoogleGenerativeAIError as error:
            if "RESOURCE_EXHAUSTED" not in str(error) or attempt == max_retries:
                raise
            print(f"  rate limit hit, retrying in {EMBED_BATCH_PAUSE}s...")
            time.sleep(EMBED_BATCH_PAUSE)


def add_documents(vectorstore, docs, chunk_size=1000, chunk_overlap=150):
    """Split documents into chunks and store their embeddings in the vector store.

    Chunks are embedded in small batches with pauses in between to stay under the
    Gemini free-tier limit of 100 embedding requests per minute.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    for start in range(0, len(splits), EMBED_BATCH_SIZE):
        if start:
            time.sleep(EMBED_BATCH_PAUSE)
        batch = splits[start:start + EMBED_BATCH_SIZE]
        add_batch_with_retry(vectorstore, batch)
        print(f"  embedded {start + len(batch)}/{len(splits)} chunks")
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


def handle_add_command(vectorstore, argument):
    """Load a new source named by the /add command into the vector store."""
    if not argument:
        print("usage: /add <url, YouTube link, file path, or folder>")
        return
    try:
        docs = load_source(argument)
    except Exception as error:  # loaders raise many different errors for bad input
        print(f"could not load {argument}: {error}")
        return
    count = add_documents(vectorstore, docs)
    print(f"added {argument} ({count} chunks)")


def question_loop(vectorstore, rag_chain):
    """Answer questions and handle /add commands until an empty line is entered."""
    while True:
        line = input("llm>> ").strip()
        if not line:
            break
        if line.startswith("/add"):
            handle_add_command(vectorstore, line[len("/add"):].strip())
        else:
            print(rag_chain.invoke(line))


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

    print("Welcome to my RAG application. Ask a question about these documents (empty line to quit).")
    print("Add a new source at any time with: /add <url, YouTube link, file path, or folder>")
    for source in list_sources(vectorstore):
        print(f"  {source}")
    question_loop(vectorstore, build_rag_chain(vectorstore))


if __name__ == "__main__":
    main()
