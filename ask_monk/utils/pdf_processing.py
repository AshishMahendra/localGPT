import logging
import os
import shutil
import httpx
from io import BytesIO
import traceback
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import re
from utils.embeddings import save_embeddings


def file_log(logentry):
    with open("file_ingest.log", "a") as file1:
        file1.write(logentry + "\n")
    print(logentry + "\n")


def clear_existing_data():
    persist_directory = os.getenv("PERSIST_DIRECTORY", "persist")
    if os.path.exists(persist_directory):
        for file in os.listdir(persist_directory):
            file_path = os.path.join(persist_directory, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):  # remove directory if directory persistence is used
                shutil.rmtree(file_path)
    logging.info("Cleared existing data from the database.")


def load_pdf_document(file_content: BytesIO, pdf_path):
    """Read PDF and return list of Document objects for all pages."""
    documents = []
    with pdfplumber.open(file_content) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                normalized_text = preprocess_text(text)
                documents.append(
                    Document(page_content=normalized_text, metadata={"source": pdf_path, "page_number": i + 1})
                )
    return documents


def preprocess_text(text):
    """Remove non-alphanumeric characters, except necessary punctuation, and normalize spaces and case."""
    text = re.sub(r"[^\w\s-]", "", text)  # Keep alphanumeric, whitespace, hyphens
    text = re.sub(r"\s+", " ", text)  # Reduce multiple spaces to a single space
    return text.lower().strip()


async def process_documents(file_urls):
    clear_existing_data()
    async with httpx.AsyncClient() as client:
        for url in file_urls:
            try:
                response = await client.get(url)
                response.raise_for_status()
                with BytesIO(response.content) as pdf_file:
                    logging.info(f"Loading documents from {url}")
                    documents = load_pdf_document(pdf_file, url)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    texts = text_splitter.split_documents(documents)
                    logging.info(f"Loaded {len(texts)} chunks of text from documents")
                    save_embeddings(texts)
            except Exception as e:
                traceback.print_exc()
                logging.error(f"Error processing PDF from {url}: {e}")
