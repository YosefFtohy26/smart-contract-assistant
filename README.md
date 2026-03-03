# 📜 Smart Contract Assistant – RAG with Local LLM

A production‑ready **Retrieval‑Augmented Generation (RAG)** system for querying legal contracts.  
Upload PDF or DOCX contracts, and ask natural‑language questions – the assistant answers **strictly from the retrieved contract text**, with source citations and a legal disclaimer.

**Key features:**
- 📤 Upload and index contracts (PDF, DOCX) into a Chroma vector database.
- 💬 Conversational Q&A with session memory.
- 🛡️ Guardrails: domain check, grounding validation, source citations.
- 🔒 Fully local – uses Ollama with Mistral (or any other local model) – no API quotas, no data leaks.
- 🌐 Dual interface: FastAPI (LangServe) backend + Gradio web UI.
- 📊 Built‑in evaluation script for semantic similarity testing.

---

## 🧱 Project Structure

```
smart-contract-assistant/
├── app/
│   ├── chain.py          # Main RAG chain (retriever, prompts, guardrails)
│   ├── server.py         # FastAPI + LangServe server
│   └── utils.py          # Domain check, grounding, citations, disclaimer
├── ingestion/
│   ├── loader.py         # PDF/DOCX extraction with metadata
│   └── processing.py     # Chunking and vector store insertion
├── ui/
│   └── gradio_app.py     # Simple chat UI
├── vectorstore/          # Persisted Chroma database (created at runtime)
├── uploaded_docs/        # Temporary storage for uploaded files
├── requirements.txt
└── README.md
```

---

## ⚙️ Prerequisites

- **Python 3.10 – 3.11** (the project has been tested with 3.10)
- **Ollama** – install from [ollama.com](https://ollama.com/)
- **Git** (to clone the repository)

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YosefFothy26/smart-contract-assistant.git
cd smart-contract-assistant
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and set up Ollama
- Download and install Ollama from [ollama.com](https://ollama.com/).
- Pull the model you want to use (e.g., Mistral):
  ```bash
  ollama pull mistral
  ```
- Keep Ollama running in the background (it starts automatically as a service on most systems).  
  You can verify with `ollama list`.

### 5. Environment variables (optional)
Create a `.env` file in the project root if you need to set:
- `HF_TOKEN` – to avoid Hugging Face rate limits when downloading the embedding model. Get a free token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Example `.env`:
```
HF_TOKEN=your_hf_token_here
```

No other API keys are required because the LLM runs locally via Ollama.

---

## 🔧 Running the Application

### Start the FastAPI backend server
```bash
python -m app.server
```
You should see output similar to:
```
INFO:     Started server process [xxxx]
INFO:     Application startup complete.
LANGSERVE: Playground for chain "/chat/" is live at:
LANGSERVE:  └──> /chat/playground/
...
Uvicorn running on http://0.0.0.0:8000
```

The API is now available at `http://localhost:8000`.

### (Optional) Launch the Gradio UI
In another terminal (with the same virtual environment activated):
```bash
python ui/gradio_app.py
```
The UI will open at `http://localhost:7860`. Use it to upload contracts and chat interactively.

---

## 📡 API Endpoints

### `POST /upload`
Upload a PDF or DOCX file to index it into the vector database.

**Request** (multipart/form-data):
- `file`: the contract file

**Response** (JSON):
```json
{
  "message": "filename.pdf uploaded and indexed successfully."
}
```

### `POST /chat/invoke`
Send a question with a session ID to maintain conversation history.

**Request body** (JSON):
```json
{
  "input": "What is the termination clause?",
  "session_id": "user123"
}
```
`session_id` is optional – defaults to `"default"`.

**Response** (JSON):
```json
{
  "answer": "The contract can be terminated with 30 days written notice.\n\n---\n📄 contract.pdf (Page 5)\n\n---\n*DISCLAIMER: This assistant provides summaries ...*"
}
```

### Interactive Playground
Visit [http://localhost:8000/chat/playground/](http://localhost:8000/chat/playground/) to test the chain visually.

---

## 💻 Usage Examples

### Upload a contract with `curl`
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/contract.pdf"
```

### Ask a question with `curl`
```bash
curl -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": "What is the payment schedule?", "session_id": "test"}'
```

### Use the chain from Python
```python
from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/chat/")
response = chain.invoke(
    {"input": "Who are the parties?"},
    config={"configurable": {"session_id": "abc123"}}
)
print(response["answer"])
```

---


## 🛠️ Troubleshooting

### `ModuleNotFoundError` / Import errors
Ensure all dependencies are installed and you're using the **correct versions**. The pinned `requirements.txt` is essential – do not upgrade packages arbitrarily, as newer versions may break compatibility.

### Ollama connection refused
Make sure the Ollama service is running. You can test it with:
```bash
ollama list
```
If it's not running, start it with `ollama serve` (or restart the Ollama application).

### No answer / empty retrieval
- The vector store might be empty. Upload at least one contract first.
- Check that the retriever returns documents by adding a `print` inside `guardrailed_chain` (already there: `print("DEBUG RESULT:", result)`). Look at the console output to see if `source_documents` are present.

### "This assistant only answers contract-related questions."
The domain guardrail (`is_query_in_domain`) is triggered. The question must contain at least one of the keywords defined in `utils.py`. You can extend the keyword list there if needed.

### Pydantic warnings about mixing V1/V2
If you see warnings like:
```
UserWarning: Mixing V1 and V2 models is not supported.
```
This is usually harmless but indicates some packages are out of sync. Stick to the pinned versions in `requirements.txt`.

---

