from fastapi import FastAPI, HTTPException
from typing import List
import uvicorn
from threading import Lock
import os
import logging
import torch
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
import traceback
from fastapi.middleware.cors import CORSMiddleware

from run_localGPT import load_model
from prompt_template_utils import get_prompt_template

from langchain.vectorstores import Chroma
from ingest import process_documents
from constants import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
)

from threading import Lock
from pydantic import BaseModel, HttpUrl
from typing import List
import traceback


class FileInfo(BaseModel):
    id: int
    name: str
    file: HttpUrl


class FolderInfo(BaseModel):
    id: int
    uid: str
    name: str
    slug: str
    user_id: int
    url: HttpUrl
    files: List[FileInfo]


class DocumentData(BaseModel):
    data: FolderInfo


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def reinitialize_components():
    global DB, RETRIEVER, QA
    DB = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=EMBEDDINGS,  # Ensure this function is suitable for your needs
        client_settings=CHROMA_SETTINGS,
    )
    RETRIEVER = DB.as_retriever()
    prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)
    QA = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=RETRIEVER,
        return_source_documents=SHOW_SOURCES,
        chain_type_kwargs={"prompt": prompt},
    )


@app.post("/api/run_ingest")
async def ingest_from_json(document_data: DocumentData):
    file_urls = [file.file for file in document_data.data.files]
    try:
        results = await process_documents(file_urls, DEVICE_TYPE)
        reinitialize_components()
        return {"message": "Documents processed and vectorstore updated successfully", "results": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class PromptModel(BaseModel):
    user_prompt: str


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


class FeedbackModel(BaseModel):
    user_prompt: str
    feedback: str


@app.post("/api/feedback")
def receive_feedback(feedback: FeedbackModel):
    print(f"Received feedback for '{feedback.user_prompt}': {feedback.feedback}")
    return {"message": "Thank you for your feedback!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8500)
