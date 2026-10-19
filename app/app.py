import sys
import uuid
import traceback
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
USER_AVATAR = str(ROOT / "assets" / "user_avatar.png")
sys.path.insert(0, str(ROOT))

from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vector_store import build_vector_store
from src.rag_pipeline import answer_question
from src.llm import get_llm
from src.web_search import answer_from_web

DOCS_DIR = ROOT / "data" / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Nexa AI",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.app-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 4px 0 18px;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(128,128,128,.16);
}
.app-topbar-title {
    font-size: 22px;
    font-weight: 650;
    letter-spacing: -0.5px;
}
.app-topbar-subtitle {
    font-size: 12px;
    color: #888b96;
    margin-top: 2px;
}
.app-topbar-actions {
    color: #777b85;
    font-size: 21px;
}

#MainMenu, footer {visibility:hidden;}
header[data-testid="stHeader"] {background:transparent;}
.block-container {
    max-width: 1050px;
    padding-top: 0.6rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,.18);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.6rem;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.45rem;
}
[data-testid="stSidebar"] hr {
    margin-top: 0.45rem;
    margin-bottom: 0.45rem;
}
[data-testid="stSidebar"] .brand {
    margin-top: 0;
    margin-bottom: 0;
}
[data-testid="stSidebar"] .brand-sub {
    margin-bottom: 0.4rem;
}
[data-testid="stSidebar"] div.stButton > button {
    min-height: 36px;
    padding-top: 0.35rem;
    padding-bottom: 0.35rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] {
    gap: 0.25rem;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    padding: 0.25rem;
}
.brand {
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -1.2px;
    margin: 8px 0 2px;
}
.muted {color: #8b8d99; font-size: 13px;}
.hero {padding: 40px 0 22px;}
.hero h1 {
    font-size: clamp(30px, 4vw, 43px);
    font-weight: 750;
    letter-spacing: -1.6px;
    line-height: 1.15;
    margin-bottom: 10px;
}
.hero p {color: #898b98; font-size: 16px;}
.section-label {
    font-size: 12px;
    letter-spacing: 1px;
    font-weight: 700;
    color: #898b98;
    margin: 14px 0 8px;
}
div[data-testid="stChatMessage"] {
    padding: 16px 20px;
    border: 1px solid rgba(128,128,128,.14);
    border-radius: 15px;
    margin-bottom: 12px;
}
div[data-testid="stChatInput"] {
    border-radius: 17px;
}
div.stButton > button {
    border-radius: 10px;
    min-height: 40px;
}
div[data-testid="stFileUploader"] {
    border-radius: 12px;
}
hr {opacity: .25;}
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
if "chats" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.chats = {
        cid: {"title": "New chat", "messages": []}
    }
    st.session_state.active_chat = cid

if "active_chat" not in st.session_state:
    st.session_state.active_chat = next(iter(st.session_state.chats))

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "General AI"

chats = st.session_state.chats
active_id = st.session_state.active_chat
messages = chats[active_id]["messages"]

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="brand">Nexa AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">YOUR PERSONAL AI WORKSPACE</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if st.button("+ New chat", use_container_width=True, type="primary"):
        cid = str(uuid.uuid4())
        st.session_state.chats[cid] = {
            "title": "New chat",
            "messages": [],
        }
        st.session_state.active_chat = cid
        st.rerun()

    st.divider()
    st.markdown('<div class="section-label">WORKSPACE</div>',
                unsafe_allow_html=True)

    st.session_state.chat_mode = st.radio(
        "Chat mode",
        ["General AI", "Chat with documents"],
        index=0 if st.session_state.chat_mode == "General AI" else 1,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<div class="section-label">RECENT CHATS</div>',
                unsafe_allow_html=True)

    for cid, chat in list(chats.items())[::-1]:
        label = chat["title"][:30]
        if st.button(label, key=f"chat_{cid}", use_container_width=True):
            st.session_state.active_chat = cid
            st.rerun()

    st.divider()

    if st.button("Clear current chat", use_container_width=True):
        chats[st.session_state.active_chat]["messages"] = []
        chats[st.session_state.active_chat]["title"] = "New chat"
        st.rerun()

    st.divider()
    st.markdown("**Local AI models**")
    st.caption("Chat: Qwen 2.5 3B (local Ollama)")
    st.caption("Embeddings: nomic-embed-text")
    st.caption("No hosted API key required for local chat.")

# ---------- Main area ----------
st.markdown("""
<div class="app-topbar">
  <div>
    <div class="app-topbar-title">Nexa AI</div>
    <div class="app-topbar-subtitle">Personal AI workspace</div>
  </div>
  <div class="app-topbar-actions"></div>
</div>
""", unsafe_allow_html=True)

messages = chats[st.session_state.active_chat]["messages"]
mode = st.session_state.chat_mode

if not messages:
    st.markdown("""
    <div class="hero">
      <h1>What can I help you with?</h1>
      <p>Ask, learn, code, write or explore your documents.</p>
    </div>
    """, unsafe_allow_html=True)

    if mode == "Chat with documents":
        if st.session_state.document_ready:
            st.info("Document mode is ready. Ask a question about your indexed file.")
        else:
            st.info("Upload and index a document in the sidebar to get started.")
    else:
        st.markdown("#### Try asking")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Write Python code", use_container_width=True):
                st.session_state.pending_prompt = "Write a Python program to add two numbers."
                st.rerun()
            if st.button("Explain a concept", use_container_width=True):
                st.session_state.pending_prompt = "Explain machine learning in simple terms."
                st.rerun()
        with col2:
            if st.button("Improve my writing", use_container_width=True):
                st.session_state.pending_prompt = "Help me write a clear and professional message."
                st.rerun()
            if st.button("Analyze some data", use_container_width=True):
                st.session_state.pending_prompt = "Explain how to analyze a dataset using Python and pandas."
                st.rerun()
else:
    st.caption(
        "DOCUMENT MODE Â· " + st.session_state.document_name
        if mode == "Chat with documents" and st.session_state.document_ready
        else ("GENERAL AI Â· Local Ollama" if mode == "General AI"
              else "DOCUMENT MODE Â· Upload and index a document first")
    )

for msg in messages:
    with st.chat_message(msg["role"], avatar=chr(0x1F468) if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    if isinstance(source, dict):
                        title = source.get("title", "Web result")
                        url = source.get("url", "")
                        date = source.get("date", "")
                        label = f"{title} — Published: {date}" if date else title
                        if url:
                            st.markdown(f"- [{label}]({url})")
                        else:
                            st.write(f"- {label}")
                    else:
                        st.write(f"- {source}")

pending = st.session_state.pending_prompt
chat_input = st.chat_input(
    "Message Nexa AI...",
    accept_file=True,
    file_type=["pdf", "txt"],
)
question = chat_input.text if chat_input else ""
if pending:
    question = pending
    st.session_state.pending_prompt = ""

# Process files attached directly in the chat input.
if chat_input and chat_input.files:
    try:
        all_docs = []
        uploaded_names = []

        with st.spinner("Reading attached files and building the knowledge base..."):
            for attached in chat_input.files:
                filename = Path(attached.name).name
                suffix = Path(filename).suffix.lower()
                if suffix not in {".pdf", ".txt"}:
                    raise ValueError("Only PDF and TXT files are supported.")

                file_bytes = attached.getvalue()
                if not file_bytes:
                    raise ValueError("The uploaded file is empty.")
                if len(file_bytes) > 10 * 1024 * 1024:
                    raise ValueError("Each uploaded file must be 10 MB or smaller.")

                if suffix == ".pdf" and not file_bytes.startswith(b"%PDF-"):
                    raise ValueError("The uploaded PDF has an invalid file signature.")
                if suffix == ".txt":
                    try:
                        file_bytes.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ValueError("TXT files must use UTF-8 encoding.") from exc

                file_path = ROOT / "data" / "documents" / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(file_bytes)

                all_docs.extend(load_documents(file_path))
                uploaded_names.append(filename)

            chunks = split_documents(all_docs)
            if not chunks:
                st.error("No readable text found in the attached files.")
            else:
                build_vector_store(chunks)
                st.session_state.document_ready = True
                st.session_state.document_name = ", ".join(uploaded_names)
                st.session_state.chat_mode = "Chat with documents"

                if not question.strip():
                    question = "Summarize the uploaded documents."

                st.success("Files attached and indexed successfully.")

    except Exception as exc:
        print(traceback.format_exc())
        if isinstance(exc, (ValueError, FileNotFoundError)):
            st.error(str(exc))
        else:
            st.error("Attachment processing failed. Check the file and try again.")

mode = st.session_state.chat_mode

# ---------- Answer generation ----------
if question:
    active_id = st.session_state.active_chat
    chat = st.session_state.chats[active_id]
    chat["messages"].append({"role": "user", "content": question})

    if chat["title"] == "New chat":
        chat["title"] = question[:30]

    with st.chat_message("user", avatar=chr(0x1F468)):
        st.markdown(question)

    with st.chat_message("assistant", avatar="assistant"):
        try:
            with st.spinner("Thinking..."):
                sources = []

                if mode == "Chat with documents":
                    if not st.session_state.document_ready:
                        answer = "Please upload and index a PDF or TXT document first."
                    else:
                        result = answer_question(question)
                        answer = str(result.get("answer", "No answer returned."))
                        sources = result.get("sources", [])
                else:
                    result = answer_from_web(question)
                    answer = str(result.get("answer", "No answer returned."))
                    sources = result.get("sources", [])

                    st.markdown(answer)

            if mode == "Chat with documents":
                st.markdown(answer)
            if sources:
                heading = "Document Sources" if mode == "Chat with documents" else "Web Sources"
                with st.expander(heading):
                    for source in sources:
                        if isinstance(source, dict):
                            title = source.get("title", "Web result")
                            url = source.get("url", "")
                            date = source.get("date", "")
                            label = f"{title} — Published: {date}" if date else title
                            if url:
                                st.markdown(f"- [{label}]({url})")
                            else:
                                st.write(f"- {label}")
                        else:
                            st.write(f"- {source}")

            chat["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception:
            st.error("The answer could not be generated. Error details:")
            st.code(traceback.format_exc())

