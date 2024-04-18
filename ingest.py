import logging
import os
import shutil
import tempfile
import httpx
from io import BytesIO
import traceback
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from utils import get_embeddings
from langchain.document_loaders import UnstructuredFileLoader
import re
from PyPDF2 import PdfReader
from constants import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
)

import pdfplumber


def file_log(logentry):
    file1 = open("file_ingest.log", "a")
    file1.write(logentry + "\n")
    file1.close()
    print(logentry + "\n")


def fix_broken_words(text):
    hyphenated_words_pattern = r"(\w+-\n\w+)"

    def _fix_match(match):
        return match.group(1).replace("-\n", "")

    fixed_text = re.sub(hyphenated_words_pattern, _fix_match, text)
    return fixed_text


def clear_existing_data():
    if os.path.exists(PERSIST_DIRECTORY):
        for file in os.listdir(PERSIST_DIRECTORY):
            file_path = os.path.join(PERSIST_DIRECTORY, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):  # remove directory if directory persistence is used
                shutil.rmtree(file_path)
    logging.info("Cleared existing data from the database.")


def load_pdf_document(file_content: BytesIO, url) -> list[Document]:
    documents = []
    try:
        with pdfplumber.open(file_content) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    doc = Document(page_content=text, metadata={"source": url, "page_number": i + 1})
                    documents.append(doc)
    except Exception as ex:
        logging.error(f"Error loading PDF document from {url}: {ex}")
    return documents


# def load_pdf_document(file_content: str, url) -> list[Document]:
#     documents = []
#     try:
#         reader = PdfReader(file_content)
#         num_pages = len(reader.pages)
#         for i in range(num_pages):
#             page = reader.pages[i]
#             text = page.extract_text()
#             if text:
#                 text = fix_broken_words(text)
#                 doc = Document(page_content=text, metadata={"source": url, "page_number": i + 1})
#                 documents.append(doc)
#     except Exception as ex:
#         logging.error(f"Error loading PDF document from {url}: {ex}")
#     return documents


# def load_pdf_document(file_content: bytes, url: str) -> list[Document]:
#     documents = []
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(file_content)
#             tmp_file_path = tmp_file.name

#         # Use the temporary file path with UnstructuredFileLoader
#         loader = UnstructuredFileLoader(tmp_file_path)
#         loaded_documents = loader.load()  # Adjust based on how the loader returns content

#         for loaded_doc in loaded_documents:
#             # Assuming each loaded_doc is a Document object, we retrieve its page_content.
#             text = loaded_doc.page_content if hasattr(loaded_doc, "page_content") else None
#             if text:
#                 text = fix_broken_words(text)
#                 # Create a new Document object with fixed text and the correct metadata
#                 doc = Document(
#                     page_content=text, metadata={"source": url, "page_number": loaded_doc.metadata["page_number"]}
#                 )
#                 documents.append(doc)

#     except Exception as ex:
#         logging.error(f"Error loading PDF document from {url}: {ex}")
#     finally:
#         # Clean up the temporary file
#         if tmp_file_path:
#             os.unlink(tmp_file_path)

#     return documents


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
                    logging.info(f"Loaded {len(documents)} documents from {url}")
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
