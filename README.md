# 📜 Smart Contract Summary & Q&A Assistant

An end-to-end RAG (Retrieval Augmented Generation) application built for the NVIDIA DLI Workshop Project alignment. This tool allows users to upload complex legal documents and interact with them using a grounded, conversational AI.

## 🚀 Key Features
- **Multi-Format Ingestion:** Supports PDF and DOCX parsing.
- **Advanced RAG Pipeline:** Uses Recursive Character Splitting and ChromaDB for semantic search.
- **Microservice Architecture:** Decoupled Backend (LangServe/FastAPI) and Frontend (Gradio).
- **Guard-rails:** Domain-specific filtering and strict grounding to prevent hallucinations.
- **Conversational Memory:** Remembers context for follow-up questions.
- **Summarization:** Map-Reduce strategy for long-form contract overviews.

## 🏗️ Project Architecture
The system is divided into three main layers:
1. **Ingestion Layer:** Extracts and transforms raw text into high-dimensional vectors.
2. **Logic Layer (API):** A FastAPI/LangServe service that manages retrieval and LLM chain logic.
3. **Interface Layer:** A dual-tab Gradio UI for document management and chat.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd smart-contract-assistant