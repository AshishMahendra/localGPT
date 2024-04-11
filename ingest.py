import logging
import os
import shutil

import httpx
from io import BytesIO
import traceback
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from utils import get_embeddings
from PyPDF2 import PdfReader
from constants import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
)


def file_log(logentry):
    file1 = open("file_ingest.log", "a")
    file1.write(logentry + "\n")
    file1.close()
    print(logentry + "\n")


def clear_existing_data():
    if os.path.exists(PERSIST_DIRECTORY):
        for file in os.listdir(PERSIST_DIRECTORY):
            file_path = os.path.join(PERSIST_DIRECTORY, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):  # remove directory if directory persistence is used
                shutil.rmtree(file_path)
    logging.info("Cleared existing data from the database.")


def load_pdf_document(file_content: str, url) -> list[Document]:
    documents = []
    try:
        reader = PdfReader(file_content)
        num_pages = len(reader.pages)
        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            if text:
                doc = Document(page_content=text, metadata={"source": url, "page_number": i + 1})
                documents.append(doc)
    except Exception as ex:
        logging.error(f"Error loading PDF document from {url}: {ex}")
    return documents


async def process_documents(file_urls, device_type):
    results = []
    full_texts = []
    clear_existing_data()
    async with httpx.AsyncClient() as client:
        for url in file_urls:
            try:
                response = await client.get(url)
                response.raise_for_status()  # ensure successful response
                with BytesIO(response.content) as pdf_file:
                    logging.info(f"Loading documents from {pdf_file}")
                    documents = load_pdf_document(pdf_file, url)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    texts = text_splitter.split_documents(documents)
                    logging.info(f"Loaded {len(documents)} documents from {pdf_file}")
                    logging.info(f"Split into {len(texts)} chunks of text")
                    full_texts.extend(texts)
            except Exception as e:
                traceback.print_exc()
                logging.error(f"Error processing PDF from {url}: {e}")
                results.append({"url": url, "error": str(e)})

    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")
    db = Chroma.from_documents(
        full_texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
