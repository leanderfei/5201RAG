from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch
import re
import json
import shutil
from pathlib import Path

def load_documents():
    """
    Auto-load all supported document types under knowledge_base.

    Returns:
        list: A list containing all loaded documents.
    """
    knowledge_dir = Path("knowledge_base")
    if not knowledge_dir.exists():
        raise FileNotFoundError("knowledge_base directory does not exist. Please create it and add documents first.")

    supported_suffixes = {".txt", ".md", ".pdf", ".docx"}
    all_files = sorted(
        [
            path for path in knowledge_dir.iterdir()
            if path.is_file() and path.suffix.lower() in supported_suffixes
        ]
    )

    if not all_files:
        raise ValueError("No available files found in knowledge_base (supported: txt/md/pdf/docx).")

    documents = []
    for path in all_files:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            loaded = TextLoader(str(path), encoding="utf8").load()
        elif suffix == ".md":
            loaded = UnstructuredMarkdownLoader(str(path)).load()
        elif suffix == ".pdf":
            loaded = UnstructuredPDFLoader(
                str(path),
                mode="elements",  # Element mode
                strategy="hi_res",  # High-resolution strategy
                languages=["eng", "chi_sim"],  # Supports English and Simplified Chinese
            ).load()
        else:  # .docx
            loaded = UnstructuredWordDocumentLoader(str(path)).load()

        documents.extend(loaded)

    print(f"Loaded files: {len(all_files)}, parsed document chunks: {len(documents)}")
    return documents

def clean_content(documents: list):

    """Text cleaning."""
    cleaned_docs = []

    for doc in documents:

    # 1) Process page_content: remove extra newlines and spaces
        text = doc.page_content

        # Replace consecutive newlines with two newlines (regex pattern: r"\n{2,}")
        # r"" means raw string to avoid escaping issues
        # \n means newline
        # {2,} is a quantifier meaning the previous token appears 2 or more times
        text = re.sub(r"\n{2,}", "\n\n", text)
        text = text.strip()

        # 2) Process metadata: convert unsupported types to JSON strings for Chroma
        allowed_types = (str, int, float, bool)
        for key, value in doc.metadata.items():
            if not isinstance(value, allowed_types):
                try:
                    doc.metadata[key] = json.dumps(value, ensure_ascii=False)
                except (TypeError, ValueError):
                    # If json.dumps fails (e.g., non-serializable objects), fallback to str
                    doc.metadata[key] = str(value)

        # 3) Update document content
        doc.page_content = text
        cleaned_docs.append(doc)

    return cleaned_docs

def text_split(documents):
    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "。"], # Separator list
        chunk_size=500, # Maximum length per chunk
        chunk_overlap=50, # Overlap length between chunks
    )
    texts = text_splitter.split_documents(documents)
    return texts

def save_to_db(texts):
    persist_dir = Path("vectorstore")
    if persist_dir.exists():
        # To avoid duplicated accumulation across repeated indexing runs, rebuild from scratch.
        shutil.rmtree(persist_dir)

    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="./bge-base-zh-v1.5",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        }, # Output normalized vectors, better for cosine similarity
    )
    # Embed and store into vector database
    vectorstore = Chroma.from_documents(
        texts, # Document list
        embedding_model, # Embedding model
        persist_directory=str(persist_dir), # Storage path
    )
    return vectorstore

if __name__ == '__main__':
    # 1) Load documents
    documents = load_documents()
    # 2) Clean documents
    cleaned_docs = clean_content(documents)
    # 3) Split text
    texts = text_split(cleaned_docs)
    # 4) Save to database
    vectorstore = save_to_db(texts)
    # 5) Inspect database content
    print(vectorstore.get().keys()) # Show all fields
    print(vectorstore.get(include=["embeddings"])["embeddings"][:5, :5]) # Show embedding vectors