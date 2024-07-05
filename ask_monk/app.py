from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from utils.pdf_processing import process_documents
from utils.question_answering import answer_question
from utils.highlight import highlight_text_in_pdf
import os
import logging
import traceback
import tempfile
import httpx
import boto3
from urllib.parse import urlparse
from threading import Lock
from models.schemas import DocumentData, QuestionRequest, FeedbackModel, HighlightRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

request_lock = Lock()


@app.post("/api/run_ingest")
async def ingest_from_json(document_data: DocumentData):
    file_urls = [file.file for file in document_data.data.files]
    try:
        await process_documents(file_urls)
        return {"message": "Documents processed and vectorstore updated successfully"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/prompt_route")
async def prompt_route(question_request: QuestionRequest):
    with request_lock:
        try:
            answer, references = answer_question(question_request.question)
            # Use a set to avoid duplicate file names in the source list
            seen_files = set()
            source_list = []

            if not answer or "I apologize" in answer or "there is no information" in answer:
                for document in references:
                    pdf_name = document.metadata["source"]
                    if pdf_name not in seen_files:
                        seen_files.add(pdf_name)
                        source_list.append({"PDF": pdf_name})
                answer = "I apologize, but I'm unable to find detailed information on this topic. Please refer to the following sources for more information."
            else:
                for document in references:
                    pdf_name = document.metadata["source"]
                    source_list.append(
                        {
                            "filename": pdf_name,
                            "pageNumber": document.metadata.get("page_number"),
                            "highlightText": str(document.page_content),
                        }
                    )

            prompt_response_dict = {
                "Prompt": question_request.question,
                "Answer": answer,
                "Sources": source_list,
            }

            return prompt_response_dict

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def receive_feedback(feedback: FeedbackModel):
    print(f"Received feedback for '{feedback.user_prompt}': {feedback.feedback}")
    return {"message": "Thank you for your feedback!"}


@app.post("/api/highlight_pdf")
async def highlight_pdf_endpoint(highlight_requests: List[HighlightRequest]):
    results = []
    for request in highlight_requests:
        pdf_path = None
        try:
            parsed_url = urlparse(request.pdf_name)
            bucket = parsed_url.netloc.split(".")[0]
            key_prefix = os.path.dirname(parsed_url.path).strip("/")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                response = httpx.get(request.pdf_name)
                response.raise_for_status()
                tmp_file.write(response.content)
                pdf_path = tmp_file.name

            highlighted_pdf = highlight_text_in_pdf(pdf_path, request.page_number, request.highlight_text)

            if highlighted_pdf:
                image_name = f"{key_prefix}/highlighted_page_{request.page_number}.png"
                s3_image_url = upload_image_to_s3(highlighted_pdf, bucket, image_name)
                results.append({"pdf_name": request.pdf_name, "highlighted_image": s3_image_url})
            else:
                results.append(
                    {"pdf_name": request.pdf_name, "error": "No matching text found or failed to create highlight."}
                )
        except Exception as e:
            logging.error(f"Error highlighting {request.pdf_name}: {e}")
            results.append({"pdf_name": request.pdf_name, "error": str(e)})
        finally:
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)
            if highlighted_pdf and os.path.exists(highlighted_pdf):
                os.remove(highlighted_pdf)

    return results


def upload_image_to_s3(image_path, bucket, object_name):
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(image_path, bucket, object_name)
        return f"https://{bucket}.s3.amazonaws.com/{object_name}"
    except Exception as e:
        logging.error(f"Failed to upload {image_path} to {bucket}/{object_name}: {e}")
        return str(e)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
