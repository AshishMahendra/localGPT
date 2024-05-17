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
from constants import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
)
import difflib
import pdfplumber
import pdfplumber
import re


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


def debug_extract_words(page):
    """Debug function to extract words from the page, with added details."""
    words = page.extract_words(keep_blank_chars=True)
    # print(f"Extracted Words: {[word['text'] for word in words]}")
    return words


def preprocess_text(text):
    """Remove non-alphanumeric characters, except necessary punctuation, and normalize spaces and case."""
    text = re.sub(r"[^\w\s-]", "", text)  # Keep alphanumeric, whitespace, hyphens
    text = re.sub(r"\s+", " ", text)  # Reduce multiple spaces to a single space
    return text.lower().strip()


def find_text_positions(page, search_text):
    """Improved logic to handle text matches that start or end mid-word."""
    # print("Original Search Text:", search_text)
    search_text = preprocess_text(search_text)
    # print("Preprocessed Search Text:", search_text)

    words = debug_extract_words(page)  # Assuming this function returns word boundaries accurately
    page_text = preprocess_text(" ".join([word["text"] for word in words]))
    # print("Full Page Text:", page_text)

    # Create a continuous index for word positions in the page text
    word_positions = []
    current_index = 0
    for word in words:
        start = current_index
        end = start + len(preprocess_text(word["text"]))
        word_positions.append((start, end, word))
        current_index = end + 1  # Account for space between words

    # Use regex for finding overlapping matches
    pattern = re.compile(re.escape(search_text), re.IGNORECASE)
    matches = list(pattern.finditer(page_text))

    bounding_boxes = []
    for match in matches:
        match_start, match_end = match.start(), match.end()
        # Collect words that overlap with the match
        for start, end, word in word_positions:
            if start < match_end and end > match_start:  # Check for any overlap
                bounding_boxes.append((word["x0"], word["top"], word["x1"], word["bottom"]))

    return bounding_boxes


def highlight_text_in_pdf(pdf_path, page_number, highlight_text):
    """Highlight the specified text on the given page with enhanced debugging."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            words = debug_extract_words(page)  # Extract words with visual debugging

            # print(f"Page {page_number} - Extracted Text: {' '.join([word['text'] for word in words])}")
            bounding_boxes = find_text_positions(page, highlight_text)

            if not bounding_boxes:
                print("No matching text found.")
                return None

            page_image = page.to_image(resolution=400)
            for box in bounding_boxes:
                page_image.draw_rect(box, fill=(255, 255, 0, 64), stroke="orange", stroke_width=3)

            output_file_path = f"highlighted_page_{page_number}.png"
            page_image.save(output_file_path, quality=95)
            print(f"Highlighted text saved to {output_file_path}")

            return output_file_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return str(e)


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
