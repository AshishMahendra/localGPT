from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
import os
from .constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from .embeddings import EmbeddingWrapper
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
import traceback


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Setup
api_key = os.getenv("OPENAI_API_KEY")
chroma_db_path = PERSIST_DIRECTORY
model_name = "gpt-3.5-turbo-instruct-0914"

embedding_func = EmbeddingWrapper()
# Initialize the components
llm = OpenAI(model=model_name)
chroma_db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_func,
    client_settings=CHROMA_SETTINGS,
)


# Define the prompt structure for the QA
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on 
the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question.
    Context: {context}
    Question: {question}
    """,
)

# retrieval_qa = RetrievalQA.from_chain_type(
#     retriever=chroma_db.as_retriever(), chain_type="stuff", llm=model, prompt_template=prompt_template
# )
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=chroma_db.as_retriever(),
    return_source_documents=True,
    callbacks=callback_manager,
    chain_type_kwargs={
        "prompt": prompt_template,
    },
)


# Function to answer questions
def answer_question(question):
    try:
        result = retrieval_qa(question)
        answer = result["result"]
        return answer, result["source_documents"]
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")
        raise e
