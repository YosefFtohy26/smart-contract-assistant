from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_chunks(documents, chunk_size=1000, chunk_overlap=150):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def store_in_vector_db(chunks, persist_directory="vector_db/chroma_data"):

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    vectorstore.add_documents(chunks)
    vectorstore.persist()

    return vectorstore 