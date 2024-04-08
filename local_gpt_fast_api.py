from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from typing import List
import uvicorn
from pydantic import BaseModel
import subprocess
import shutil
from threading import Lock
import os
import logging
import torch
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from fastapi import Query
import traceback

# from langchain.embeddings import HuggingFaceEmbeddings
from run_localGPT import load_model
from prompt_template_utils import get_prompt_template

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from werkzeug.utils import secure_filename

from constants import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    SOURCE_DIRECTORY,
)


# API queue addition
from threading import Lock

request_lock = Lock()


if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"

SHOW_SOURCES = True
logging.info(f"Running on: {DEVICE_TYPE}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")

EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})

DB = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=EMBEDDINGS,
    client_settings=CHROMA_SETTINGS,
)

RETRIEVER = DB.as_retriever()

LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)

QA = RetrievalQA.from_chain_type(
    llm=LLM,
    chain_type="stuff",
    retriever=RETRIEVER,
    return_source_documents=SHOW_SOURCES,
    chain_type_kwargs={
        "prompt": prompt,
    },
)


app = FastAPI()


# API endpoint to delete the source directory and recreate it
@app.post("/api/delete_source/{folder_path:path}")
def delete_source_route(folder_path: str = "SOURCE_DOCUMENTS"):
    full_path = os.path.join("SOURCE_DOCUMENTS", folder_path)
    shutil.rmtree(full_path, ignore_errors=True)
    return {"message": f"Folder '{full_path}' successfully deleted and recreated."}


@app.post("/api/save_document")
def save_document_route(file: UploadFile = File(...), folder_path: str = Query(default="")):
    filename = secure_filename(file.filename)
    full_folder_path = os.path.join("SOURCE_DOCUMENTS", folder_path)
    os.makedirs(full_folder_path, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(full_folder_path, filename)

    with open(file_path, "wb") as out_file:
        content = file.file.read()  # Synchronous read
        out_file.write(content)  # Synchronous write

    return {"message": "File saved successfully"}


# API endpoint to run ingestion
@app.get("/api/run_ingest")
def run_ingest_route(folder_path: str = Query(default="")):
    # As before, except use subprocess.run synchronously
    try:
        global DB
        global RETRIEVER
        global QA
        specific_folder_to_ingest = (
            os.path.join(SOURCE_DIRECTORY, folder_path.strip("/")) if folder_path else SOURCE_DIRECTORY
        )

        # Pass the specific folder path to the ingestion script
        run_langest_commands = [
            "python",
            "ingest.py",
            "--device_type",
            DEVICE_TYPE,
            "--folder_path",
            specific_folder_to_ingest,  # Pass the dynamic folder path to the script
        ]
        result = subprocess.run(run_langest_commands, capture_output=True, text=True)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Script execution failed: {result.stderr}")

        DB = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=EMBEDDINGS,
            client_settings=CHROMA_SETTINGS,
        )
        RETRIEVER = DB.as_retriever()
        prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)

        QA = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            retriever=RETRIEVER,
            return_source_documents=SHOW_SOURCES,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )
        return {"message": f"Script executed successfully: {result.stdout}"}
    except Exception as e:
        traceback.print_stack()
        logging.error(f"Error occurred during /api/run_ingest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# API endpoint for prompt handling
class PromptModel(BaseModel):
    user_prompt: str


# @app.post("/api/prompt_route")
# def prompt_route(prompt_model: PromptModel):
#     global QA
#     user_prompt = prompt_model.user_prompt

#     # Acquire the lock asynchronously before processing the prompt
#     with request_lock:
#         # Get the answer from the chain
#         print(f"User Prompt: {user_prompt}")
#         res = QA(user_prompt)
#         answer, docs = res["result"], res["source_documents"]

#         prompt_response_dict = {
#             "Prompt": user_prompt,
#             "Answer": answer,
#             "Sources": [
#                 (os.path.basename(str(document.metadata["source"])), str(document.page_content)) for document in docs
#             ],
#         }

#     return prompt_response_dict


# @app.post("/api/prompt_route")
# def prompt_route(prompt_model: PromptModel):
#     global QA
#     user_prompt = prompt_model.user_prompt
#     with request_lock:
#         # Get the answer from the chain
#         print(f"User Prompt: {user_prompt}")
#         res = QA(user_prompt)
#         answer = res["result"]
#         docs = res["source_documents"]

#         prompt_response_dict = {
#             "Prompt": user_prompt,
#             "Answer": answer,
#             "Sources": [
#                 {
#                     "PDF": os.path.basename(str(document.metadata["source"])),
#                     "PageNumber": document.metadata.get("page_number"),  # Assuming this metadata is available
#                     "Text": str(document.page_content),
#                 }
#                 for document in docs
#             ],
#         }

#     return prompt_response_dict


def suggest_refined_query(user_prompt, sources):
    return "Could you specify which aspect of the 'slam tool' you are interested in?"


@app.post("/api/prompt_route")
def prompt_route(prompt_model: PromptModel):
    global QA
    user_prompt = prompt_model.user_prompt
    with request_lock:
        # Get the answer from the chain
        print(f"User Prompt: {user_prompt}")
        res = QA(user_prompt)
        answer = res["result"].strip()
        docs = res["source_documents"]

        # Use a set to avoid duplicate file names in the source list
        seen_files = set()
        source_list = []

        if not answer or "I apologize" in answer or "there is no information" in answer:
            for document in docs:
                pdf_name = os.path.basename(document.metadata["source"])
                if pdf_name not in seen_files:
                    seen_files.add(pdf_name)
                    source_list.append({"PDF": pdf_name})
            answer = "I apologize, but I'm unable to find detailed information on this topic. Please refer to the following sources for more information."
        else:
            for document in docs:
                pdf_name = os.path.basename(document.metadata["source"])
                source_list.append(
                    {
                        "PDF": pdf_name,
                        "PageNumber": document.metadata.get("page_number"),
                        "Text": str(document.page_content),
                    }
                )

        prompt_response_dict = {
            "Prompt": user_prompt,
            "Answer": answer,
            "Sources": source_list,
        }

    return prompt_response_dict


# @app.post("/api/prompt_route")
# def prompt_route(prompt_model: PromptModel):
#     global QA
#     user_prompt = prompt_model.user_prompt
#     with request_lock:
#         print(f"User Prompt: {user_prompt}")
#         res = QA(user_prompt)
#         answer = res["result"]
#         docs = res["source_documents"]

#         if not answer.strip():
#             answer = "I apologize, but I'm not familiar with your query. " + suggest_refined_query(user_prompt, docs)

#         prompt_response_dict = {
#             "Prompt": user_prompt,
#             "Answer": answer,
#             "Sources": [
#                 {
#                     "PDF": os.path.basename(str(document.metadata["source"])),
#                     "PageNumber": document.metadata.get("page_number"),
#                     "Text": str(document.page_content),
#                 }
#                 for document in docs
#             ],
#         }
#     return prompt_response_dict


class FeedbackModel(BaseModel):
    user_prompt: str
    feedback: str


@app.post("/api/feedback")
def receive_feedback(feedback: FeedbackModel):
    print(f"Received feedback for '{feedback.user_prompt}': {feedback.feedback}")
    return {"message": "Thank you for your feedback!"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8500)
