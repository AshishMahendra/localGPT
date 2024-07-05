from openai import OpenAI
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from .constants import CHROMA_SETTINGS, PERSIST_DIRECTORY

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EmbeddingWrapper(OpenAIEmbeddings):
    def embed_documents(self, documents):
        # This will now handle multiple documents correctly
        return get_openai_embeddings(documents)

    def embed_query(self, query):
        # Handle a single query by wrapping it in a list
        return get_openai_embeddings([query])[0]


def get_openai_embeddings(texts):
    embeddings = []
    for text in texts:
        # Ensure that each text is properly formatted and sent separately
        response = client.embeddings.create(model="text-embedding-ada-002", input=[text])
        embeddings.append(response.data[0].embedding)
    return embeddings


def get_single_embedding(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        embedding = get_openai_embeddings(chunk)
        embeddings.append(embedding)
    return embeddings


def save_embeddings(texts):
    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)

    embeddings = EmbeddingWrapper()
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
