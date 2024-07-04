from openai import OpenAI
import os
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_openai_embeddings(text):
    response = client.embeddings.create(model="text-embedding-ada-002", input=text)
    return response.data[0].embedding


def get_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        embedding = get_openai_embeddings(chunk)
        embeddings.append(embedding)
    return embeddings


def save_embeddings(embeddings, texts):
    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)

    db = Chroma.from_embeddings(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
