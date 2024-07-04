from pydantic import BaseModel, HttpUrl
from typing import List


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


class QuestionRequest(BaseModel):
    question: str
    document_url: str


class FeedbackModel(BaseModel):
    user_prompt: str
    feedback: str


class HighlightRequest(BaseModel):
    pdf_name: str
    page_number: int
    highlight_text: str
