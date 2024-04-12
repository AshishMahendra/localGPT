### API Usage Documentation for the Local GPT Module

#### Overview
This API enables document management, document ingestion, and interaction with a Language Learning Model (LLM). The endpoints allow for file management, ingestion processes, and querying the LLM for responses based on user inputs.

#### Activate Environment
```bash
conda activate localGPT
```
#### Running the API Server
```bash
uvicorn local_gpt_fast_api:app --host 0.0.0.0 --port 8500
```
#### Base URL
All URLs referenced in the documentation have the following base:
```
http://localhost:8500
```

#### API Endpoints

1. **Delete Source Directory**
   - **Endpoint**: `POST /api/delete_source/{folder_path}`
   - **Description**: Deletes the specified folder and its contents, then recreates the folder.
   - **Path Parameters**:
     - `folder_path` (string): The path of the folder to delete relative to the source directory.
   - **Example Request**:
     ```bash
     curl -X POST http://localhost:8500/api/delete_source/myfolder
     ```
   - **Response**:
     ```json
     {
       "message": "Folder 'SOURCE_DOCUMENTS/myfolder' successfully deleted and recreated."
     }
     ```

2. **Save Document**
   - **Endpoint**: `POST /api/save_document`
   - **Description**: Saves a document to a specified folder within the source directory.
   - **Request Body**:
     - Form data with `file` (file to be uploaded) and optional `folder_path` (subdirectory path).
   - **Example Request**:
     ```bash
     curl -F "file=@path/to/file.pdf" -F "folder_path=subfolder" http://localhost:8500/api/save_document
     ```
   - **Response**:
     ```json
     {
       "message": "File saved successfully"
     }
     ```

3. **Run Ingestion Process**
   - **Endpoint**: `GET /api/run_ingest`
   - **Description**: Initiates the ingestion process for documents within the specified folder.
   - **Query Parameters**:
     - `folder_path` (optional, string): The path to the folder to ingest documents from.
   - **Example Request**:
     ```bash
     curl http://localhost:8500/api/run_ingest?folder_path=myfolder
     ```
   - **Response**:
     ```json
     {
       "message": "Script executed successfully: [details from stdout]"
     }
     ```

4. **Prompt Handling**
   - **Endpoint**: `POST /api/prompt_route`
   - **Description**: Processes a user prompt and returns the LLM's response along with any relevant source documents.
   - **Request Body**:
     ```json
     {
       "user_prompt": "Describe the process of photosynthesis."
     }
     ```
   - **Example Request**:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"user_prompt": "Describe the process of photosynthesis."}' http://localhost:8500/api/prompt_route
     ```
   - **Response**:
     ```json
     {
       "Prompt": "Describe the process of photosynthesis.",
       "Answer": "Photosynthesis is a process used by plants and other organisms to convert light energy...",
       "Sources": [
         {"PDF": "biology_textbook.pdf", "PageNumber": 123, "Text": "The light-dependent reactions of photosynthesis..."}
       ]
     }
     ```

#### Error Handling
Each endpoint responds with appropriate HTTP status codes. For example:
- `200 OK` for successful requests.
- `400 Bad Request` if there is a problem with the input data.
- `500 Internal Server Error` if there are issues on the server.

#### Security Considerations
- Ensure that all interactions with the API are over HTTPS to protect data integrity and privacy.
- Implement authentication and authorization mechanisms if the API is exposed publicly.

#### Versioning
The current version of the API is v1. Future changes that might break backward compatibility will result in a new version number.

This documentation provides the necessary details for developers to effectively integrate and interact with your API, covering how to format requests and what responses to expect. Adjust as necessary for any specific security implementations or additional functionalities you might have.
