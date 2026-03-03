import os
from dotenv import load_dotenv
from typing import Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

from langchain_community.chat_message_histories import ChatMessageHistory

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain

from app.utils import (
    format_source_citations,
    is_query_in_domain,
    validate_response_grounding,
    LEGAL_DISCLAIMER,
)


# -------------------------
# Environment
# -------------------------

load_dotenv()

PERSIST_DIRECTORY = "vectorstore"

# -------------------------
# Embeddings (GLOBAL)
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -------------------------
# Session Memory Store
# -------------------------

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# -------------------------
# Vector Store Loader
# -------------------------

def get_vectorstore():
    """
    Always load the persisted Chroma DB.
    """
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )

# -------------------------
# Main Conversational RAG Chain
# -------------------------

def get_conversational_rag_chain():

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )



    llm = ChatOllama(model="mistral", temperature=0)

    # -------------------------
    # Query Rephrasing Prompt
    # -------------------------

    rephrase_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Generate a search query based on the conversation above.")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        rephrase_prompt
    )

    # -------------------------
    # Answer Prompt
    # -------------------------

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer ONLY using the provided context. "
         "If the answer is not in the context, say you cannot find it.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(
        llm,
        answer_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain
    )

    # -------------------------
    # Guardrail Layer
    # -------------------------

    def guardrailed_chain(inputs: Dict[str, Any]):

        user_query = inputs["input"]

        # Domain restriction
        if not is_query_in_domain(user_query):
            return {
                "answer": "This assistant only answers contract-related questions."
            }

        result = retrieval_chain.invoke(inputs)
        print("DEBUG RESULT:", result)
        answer = result["answer"]
        source_docs = result.get("source_documents", [])

        grounding_warning = validate_response_grounding(
            answer,
            source_docs,
            embeddings
        )

        citations = format_source_citations(source_docs)

        final_answer = answer

        if grounding_warning:
            final_answer += f"\n\n⚠ {grounding_warning}"

        if citations:
            final_answer += f"\n\n---\n{citations}"

        final_answer += f"\n\n{LEGAL_DISCLAIMER}"

        return {"answer": final_answer}

    guardrailed_runnable = RunnableLambda(guardrailed_chain)

    conversational_chain = RunnableWithMessageHistory(
        guardrailed_runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_chain
