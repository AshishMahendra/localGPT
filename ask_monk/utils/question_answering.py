from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        response = client.embeddings.create(model="text-embedding-ada-002", input=chunk)
        embeddings.append(response.data[0].embedding)
    return embeddings


def get_most_relevant_chunks(question, chunk_embeddings, chunks, top_k=5):
    question_embedding = client.embeddings.create(model="text-embedding-ada-002", input=question).data[0].embedding

    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]

    top_k_indices = similarities.argsort()[-top_k:][::-1]
    relevant_chunks = [chunks[i] for i in top_k_indices]

    return relevant_chunks


def generate_answer(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = client.completions.create(model="gpt-3.5-turbo-instruct-0914", prompt=prompt, max_tokens=200)

    answer = response.choices[0].text.strip()
    return answer


def answer_question_from_pdf(pdf_path, question):
    from .pdf_processing import extract_text_from_pdf, semantic_chunk_text

    document_text = extract_text_from_pdf(pdf_path)
    chunks = semantic_chunk_text(document_text)
    chunk_embeddings = get_embeddings(chunks)
    relevant_chunks = get_most_relevant_chunks(question, chunk_embeddings, chunks)
    answer = generate_answer(question, relevant_chunks)
    references = relevant_chunks

    return answer, references
