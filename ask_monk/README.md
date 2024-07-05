# ASK MonkS

This project provides an API for processing PDF documents and answering questions based on their content. It leverages advanced NLP techniques to extract meaningful data from documents, enabling users to interact with the information in an intuitive way.

## Features

- **Document Ingestion**: Ingest PDF documents and process them into semantic chunks that can be easily analyzed.
- **Question Answering**: Automatically answer questions based on the content of the ingested documents.
- **Feedback Mechanism**: Users can provide feedback on the answers, which can be used to improve the system.
- **PDF Highlighting**: Highlight text in PDFs based on query results and return the modified files.

## Installation

Follow these steps to get the environment set up:

1. Clone the repository:
    ```bash
    git clone https://github.com/AshishMahendra/localGPT.git
    ```

2. Navigate to the project directory:
    ```bash
    cd ask_monks
    ```

3. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the API, follow these steps:

1. Start the FastAPI server:
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```

2. Once the server is running, you can interact with the API through its endpoints. Use the provided endpoints to process documents, ask questions, provide feedback, and highlight text in PDFs.

## API Endpoints

- **POST /api/run_ingest**: Endpoint to ingest and process PDF documents. It converts PDFs into manageable chunks and prepares them for analysis.
- **POST /api/prompt_route**: Use this endpoint to ask questions regarding the ingested PDF documents. It uses the processed data to generate responses.
- **POST /api/feedback**: Provide feedback on the quality of the answers received. This helps in refining the accuracy and relevance of the responses.
- **POST /api/highlight_pdf**: Highlight specific text in the PDFs based on the queries and return the modified PDF files.

## License

This project is available under the MIT License, allowing flexibility for both personal and commercial use.
