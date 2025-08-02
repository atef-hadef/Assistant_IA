import os
import faiss
import pandas as pd
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.schema import Document
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def extract_text_from_csv(file) -> str:
    df = pd.read_csv(file)
    return df.to_string(index=False)

def split_into_chunks(text: str):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)

def ask_with_groq(question: str, docs: list[Document]) -> str:
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""
Réponds à la question suivante à partir des documents ci-dessous.

Documents:
{context}

Question:
{question}
"""
    chat_completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "Tu es un assistant intelligent et rigoureux."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return chat_completion.choices[0].message.content
