import streamlit as st
import html
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.embeddings import create_embeddings, EMBEDDING_MODEL_NAME
from utils.vector_store import (store_chunks, clear_database)
from utils.retriever import search_chunks

from utils.gemini import ask_gemini

# Avatars used consistently everywhere chat_message() is called
USER_AVATAR = "🧑‍💻"
ASSISTANT_AVATAR = "🤖"


# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide"
)


# ---------------------------------
# Custom Theme (CSS)
# ---------------------------------
# Streamlit doesn't expose fine-grained styling controls, so we inject
# CSS directly. This only changes appearance - no widget behavior,
# layout logic, or session state is affected by this block.

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Gradient page title */
    h1 {
        background: linear-gradient(90deg, #FF6B6B 0%, #FFA94D 50%, #FFD43B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        padding-bottom: 4px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] h2 {
        font-weight: 600;
    }

    /* Sidebar document cards (see sidebar section below) */
    .file-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 10px;
        transition: border-color 0.2s ease;
    }

    .file-card:hover {
        border-color: #FFA94D;
    }

    .file-card .file-name {
        font-weight: 600;
        font-size: 0.92em;
        word-break: break-word;
    }

    .file-card .file-meta {
        opacity: 0.65;
        font-size: 0.8em;
        margin-top: 2px;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.15);
        transition: all 0.2s ease;
        font-weight: 500;
    }

    div.stButton > button:hover {
        border-color: #FFA94D;
        color: #FFA94D;
        transform: translateY(-1px);
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 6px 10px;
        margin-bottom: 10px;
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* Chat input box */
    div[data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
    }

    /* Source citation expander */
    details {
        border-radius: 10px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------
# Session State
# ---------------------------------

if "processed" not in st.session_state:
    st.session_state.processed = False

if "text" not in st.session_state:
    st.session_state.text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_info" not in st.session_state:
    st.session_state.file_info = {}

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ---------------------------------
# Helper Function
# ---------------------------------

def split_text(text, chunk_size=1000, overlap=200):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks = text_splitter.split_text(text)

    return chunks

# ---------------------------------
# UI
# ---------------------------------

st.title("📄 AI PDF Chatbot")
st.caption("Upload your PDFs and chat with them — grounded answers, real citations, zero guesswork.")

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"pdf_uploader_{st.session_state.uploader_key}"
)


# ---------------------------------
# Detect New PDF
# ---------------------------------

if uploaded_files:

    st.session_state.processed = False

# ---------------------------------
# Process PDF ONLY ONCE
# ---------------------------------
if uploaded_files and not st.session_state.processed:

    with st.spinner("Processing PDFs — extracting text, chunking, and creating embeddings..."):

        clear_database()

        text = ""
        all_chunks = []
        file_info = {}

        for uploaded_file in uploaded_files:

            reader = PdfReader(uploaded_file)

            file_page_count = len(reader.pages)
            file_chunk_count = 0

            for page_number, page in enumerate(reader.pages, start=1):

                extracted = page.extract_text()

                if not extracted:
                    continue

                text += extracted + "\n"

                page_chunks = split_text(extracted)

                for chunk_number, chunk in enumerate(page_chunks, start=1):

                    all_chunks.append(
                        {
                            "text": chunk,
                            "page": page_number,
                            "chunk": chunk_number,
                            "file": uploaded_file.name
                        }
                    )

                    file_chunk_count += 1

            file_info[uploaded_file.name] = {
                "pages": file_page_count,
                "chunks": file_chunk_count
            }

        chunk_texts = [item["text"] for item in all_chunks]

        embeddings = create_embeddings(chunk_texts)

        store_chunks(all_chunks, embeddings)

        st.session_state.text = text
        st.session_state.chunks = all_chunks
        st.session_state.file_info = file_info
        st.session_state.processed = True

    st.success("All PDFs processed successfully!")

# ---------------------------------
# Sidebar
# ---------------------------------
# NOTE: This block runs AFTER PDF processing above, on purpose. Streamlit
# re-runs the whole script top-to-bottom on every interaction, so if the
# sidebar were rendered earlier in the file, it would display file_info
# from BEFORE this run's processing updated it - i.e. it would always be
# one step behind. Rendering it here guarantees it shows this run's data.

with st.sidebar:

    st.header("📚 Documents")

    if st.session_state.file_info:

        total_chunks = 0

        for filename, info in st.session_state.file_info.items():

            safe_name = html.escape(filename)

            st.markdown(
                f"""
                <div class="file-card">
                    <div class="file-name">📄 {safe_name}</div>
                    <div class="file-meta">{info['pages']} pages · {info['chunks']} chunks</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            total_chunks += info["chunks"]

        st.divider()

        st.write(f"**Total files:** {len(st.session_state.file_info)}")
        st.write(f"**Total chunks:** {total_chunks}")

    else:
        st.caption("No PDFs uploaded yet.")

    st.divider()

    st.write(f"**Embedding model:** `{EMBEDDING_MODEL_NAME}`")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        clear_chat_clicked = st.button("🧹 Clear Chat", use_container_width=True)

    with col2:
        clear_db_clicked = st.button("🗑️ Clear Database", use_container_width=True)

    if clear_chat_clicked:
        st.session_state.messages = []
        st.toast("Chat cleared.", icon="🧹")
        st.rerun()

    if clear_db_clicked:
        clear_database()
        st.session_state.processed = False
        st.session_state.text = ""
        st.session_state.chunks = []
        st.session_state.file_info = {}
        st.session_state.messages = []

        # Bump the uploader's key so Streamlit mounts a brand-new,
        # empty file_uploader widget on the next run. This is what
        # actually removes the files from the upload box - resetting
        # session state alone does NOT clear an st.file_uploader.
        st.session_state.uploader_key += 1

        st.toast("Database cleared. All uploaded PDFs removed.", icon="🗑️")
        st.rerun()

# ---------------------------------
# Show PDF Information
# ---------------------------------

if st.session_state.processed:

    st.divider()

    st.subheader("Extracted Text")

    st.text_area(
        "PDF Content",
        st.session_state.text,
        height=300
    )

    st.divider()

    st.subheader("Text Chunks")

    st.write(f"Total Chunks: {len(st.session_state.chunks)}")

    for chunk in st.session_state.chunks:

        with st.expander(
            f"{chunk['file']} | Page {chunk['page']} | Chunk {chunk['chunk']}"
        ):

            st.write(chunk["text"])

# ---------------------------------
# Ask Questions (Chat Interface)
# ---------------------------------

if st.session_state.processed:

    st.divider()

    st.header("Ask Questions")

    # ---- Render existing chat history ----
    for message in st.session_state.messages:

        avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR

        with st.chat_message(message["role"], avatar=avatar):

            st.write(message["content"])

            # Re-display sources under past assistant answers, if it has any
            if message["role"] == "assistant" and message.get("sources"):

                with st.expander("📄 View Source Chunks"):

                    for i, chunk in enumerate(message["sources"], start=1):

                        st.markdown(f"### Source {i}")

                        st.write(f"Page : {chunk['page']}")
                        st.write(f"File : {chunk['file']}")
                        st.write(f"Chunk : {chunk['chunk']}")

                        st.write(chunk["text"])

    # ---- Chat input box (pinned at the bottom by Streamlit) ----
    question = st.chat_input("Ask something about the PDF")

    if question:

        # Snapshot the conversation BEFORE adding the current question,
        # so ask_gemini receives only prior turns as "history" (not a
        # duplicate of the question it's currently answering).
        conversation_history = st.session_state.messages.copy()

        # 1. Show and store the user's message immediately
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user", avatar=USER_AVATAR):
            st.write(question)

        # 2. Show the assistant bubble immediately with a "thinking" spinner
        #    inside it, then fill it in once retrieval + generation finish.
        #    This replaces the silent pause with a live status indicator.
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):

            with st.spinner("🤖 Reading through the document..."):

                results = search_chunks(question)

                answer = ask_gemini(question, results, history=conversation_history)

            st.write(answer)

            with st.expander("📄 View Source Chunks"):

                for i, chunk in enumerate(results, start=1):

                    st.markdown(f"### Source {i}")

                    st.write(f"Page : {chunk['page']}")
                    st.write(f"File : {chunk['file']}")
                    st.write(f"Chunk : {chunk['chunk']}")

                    st.write(chunk["text"])

        # 3. Store the assistant's message, with its sources attached,
        #    AFTER rendering above (so the values used above are final)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": results
            }
        )