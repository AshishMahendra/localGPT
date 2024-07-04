# PDF Question Answering API

This project provides an API for processing PDF documents and answering questions based on their content.

## Features

- Ingest PDF documents and process them into semantic chunks.
- Answer questions based on the content of the ingested documents.
- Provide feedback on the answers.
- Highlight text in PDFs and return the modified files.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/yourrepository.git
    ```

2. Navigate to the project directory:
    ```bash
    cd yourrepository
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the spaCy model:
    ```bash
    python -m spacy download en_core_web_trf
    ```

## Usage

1. Start the FastAPI application:
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```

2. Use the API endpoints to process documents, ask questions, provide feedback, and highlight text in PDFs.

## Endpoints

- **POST /api/run_ingest**: Ingest and process PDF documents.
- **POST /api/prompt_route**: Ask questions based on the content of the ingested documents.
- **POST /api/feedback**: Provide feedback on the answers.
- **POST /api/highlight_pdf**: Highlight text in PDFs and return the modified files.

## License

This project is licensed under the MIT License.
