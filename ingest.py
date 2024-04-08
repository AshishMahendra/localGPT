import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import shutil
import click
import torch
from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from utils import get_embeddings
from langchain.document_loaders import UnstructuredFileLoader
from PyPDF2 import PdfReader  # Assuming PyPDF2 is acceptable for PDF processing
from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)


def file_log(logentry):
    file1 = open("file_ingest.log", "a")
    file1.write(logentry + "\n")
    file1.close()
    print(logentry + "\n")


# def load_single_document(file_path: str) -> Document:
#     # Loads a single document from a file path
#     try:
#         file_extension = os.path.splitext(file_path)[1]
#         loader_class = DOCUMENT_MAP.get(file_extension)
#         if loader_class:
#             file_log(file_path + " loaded.")
#             loader = loader_class(file_path)
#         else:
#             file_log(file_path + " document type is undefined.")
#             raise ValueError("Document type is undefined")
#         return loader.load()[0]
#     except Exception as ex:
#         file_log("%s loading error: \n%s" % (file_path, ex))
#         return None
def clear_existing_data():
    # This function should contain the logic to clear the database
    # Example for file-based storage:
    if os.path.exists(PERSIST_DIRECTORY):
        for file in os.listdir(PERSIST_DIRECTORY):
            file_path = os.path.join(PERSIST_DIRECTORY, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):  # remove directory if directory persistence is used
                shutil.rmtree(file_path)
    logging.info("Cleared existing data from the database.")


def load_pdf_document(file_path: str) -> list[Document]:
    documents = []
    try:
        reader = PdfReader(file_path)
        num_pages = len(reader.pages)
        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            if text:
                doc = Document(
                    page_content=text,
                    metadata={"source": file_path, "page_number": i + 1},  # Page numbers are generally 1-based
                )
                documents.append(doc)
    except Exception as ex:
        file_log(f"{file_path} loading error: {ex}")
    return documents


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_pdf_document, name) for name in filepaths]
        # collect data
        if futures is None:
            file_log(name + " failed to submit")
            return None
        else:
            data_list = [future.result() for future in futures]
            # return data and file paths
            return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            print("Importing: " + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)
    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            try:
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                file_log("executor task failed: %s" % (ex))
                future = None
            if future is not None:
                futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                file_log("Exception: %s" % (ex))

    return docs


def split_documents(documents_list: list) -> tuple[list[Document], list[Document]]:
    text_docs, python_docs = [], []
    for docs in documents_list:  # Here, each `docs` is expected to be a list of Document objects
        for doc in docs:  # Iterate over each Document in the list
            if doc is not None:
                file_extension = os.path.splitext(doc.metadata["source"])[1]
                if file_extension == ".py":
                    python_docs.append(doc)
                else:
                    text_docs.append(doc)
    return text_docs, python_docs


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--folder_path",
    default=SOURCE_DIRECTORY,
    type=str,
    help="Path to the folder containing documents to ingest. (Default is 'SOURCE_DOCUMENTS')",
)
def main(device_type, folder_path):
    clear_existing_data()
    logging.info(f"Loading documents from {folder_path}")
    documents = load_documents(folder_path)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {folder_path}")
    logging.info(f"Split into {len(texts)} chunks of text")

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within fun_localGPT.py.
    
    (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """

    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
