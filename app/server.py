from fastapi import FastAPI, UploadFile, File
from langserve import add_routes
import uvicorn
from dotenv import load_dotenv
from pydantic.v1 import BaseModel
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import shutil
import os

from app.chain import get_conversational_rag_chain, get_vectorstore


load_dotenv()


app = FastAPI(title="Smart Contract Assistant API")


class ChatInput(BaseModel):
    input: str
    session_id: Optional[str] = "default"

class ChatOutput(BaseModel):
    answer: str


rag_chain = get_conversational_rag_chain()

add_routes(
    app,
    rag_chain,
    path="/chat",
    input_type=ChatInput,
    output_type=ChatOutput,
)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    upload_dir = "uploaded_docs"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = text_splitter.split_documents(documents)

    # Load existing vectorstore
    vectorstore = get_vectorstore()

    # Add new documents
    vectorstore.add_documents(docs)

    # Persist changes
    vectorstore.persist()

    return {
        "message": f"{file.filename} uploaded and indexed successfully."
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
