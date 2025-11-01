from dotenv import load_dotenv
import os
load_dotenv()
import gradio as gr
from build_workflow import app

# --- Load environment variables ---


if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
if not os.getenv("TAVILY_API_KEY"):
    print("⚠️ TAVILY_API_KEY not found (web search may fail).")


def chat_fn(question, history):
    """Handles chat messages, shows processing message, and clears input field."""
    messages = history or []

    # Step 1: show “Processing…” message
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": "⏳ Processing... please wait."})
    yield messages, messages, ""  # immediately clears the input box

    # Step 2: run LangGraph pipeline
    result = None
    for output in app.stream({"question": question}):
        for _, value in output.items():
            result = value

    generation = result.get("generation", "⚠️ No answer generated.")

    # Step 3: replace placeholder with real answer
    messages[-1] = {"role": "assistant", "content": generation}
    yield messages, messages, ""  # keeps input box empty


with gr.Blocks() as demo:
    gr.Markdown("## 🤖 LangGraph CRAG Chatbot")

    chatbot = gr.Chatbot(height=400, type="messages")
    msg = gr.Textbox(label="Ask a question", placeholder="Type your question here…")
    clear = gr.Button("Clear Chat")

    # Note: now we include msg in both input and output lists
    msg.submit(chat_fn, [msg, chatbot], [chatbot, chatbot, msg], queue=True)
    clear.click(lambda: [], None, chatbot, queue=False)

demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
