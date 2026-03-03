import gradio as gr
import uuid
from langserve import RemoteRunnable

from ingestion.loader import load_document
from ingestion.processing import create_chunks, store_in_vector_db

# Connect to LangServe backend
remote_chain = RemoteRunnable("http://localhost:8000/contract-qa")


def process_file(file):

    if file is None:
        return "⚠ Please upload a PDF or DOCX file."

    try:
        # Load structured documents (with metadata)
        documents = load_document(file.name)

        # Chunk documents
        chunks = create_chunks(documents)

        # Store in vector DB (append mode)
        store_in_vector_db(chunks)

        return f"✅ Successfully indexed {len(chunks)} chunks from {file.name}"

    except Exception as e:
        return f"❌ Error during ingestion: {str(e)}"


def chat_with_assistant(message, chat_history, session_id):

    if not message:
        return "", chat_history

    try:
        response = remote_chain.invoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}}
        )

        answer = response["answer"]

    except Exception as e:
        answer = f"❌ Error contacting backend: {str(e)}"

    chat_history.append((message, answer))

    return "", chat_history


with gr.Blocks(title="Smart Contract Assistant") as demo:

    gr.Markdown("# 📜 Smart Contract Summary & Q&A Assistant")
    gr.Markdown("Upload a contract and ask questions grounded strictly in its content.")

    session_state = gr.State(str(uuid.uuid4()))

    with gr.Tabs():

        with gr.TabItem("1️⃣ Upload Contract"):
            file_input = gr.File(
                label="Upload PDF or DOCX",
                file_types=[".pdf", ".docx"]
            )

            upload_button = gr.Button("Process and Index")
            upload_status = gr.Textbox(label="Status")

            upload_button.click(
                process_file,
                inputs=file_input,
                outputs=upload_status
            )

        with gr.TabItem("2️⃣ Chat with Contract"):
            chatbot = gr.Chatbot(label="Assistant")
            msg = gr.Textbox(label="Ask a question about the contract")
            clear = gr.Button("Clear Chat")

            msg.submit(
                chat_with_assistant,
                inputs=[msg, chatbot, session_state],
                outputs=[msg, chatbot]
            )

            clear.click(
                lambda: ([], str(uuid.uuid4())),
                outputs=[chatbot, session_state],
                queue=False
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
