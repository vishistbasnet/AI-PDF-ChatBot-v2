<div align="center">

# 📄 AI PDF Chatbot

**A Retrieval-Augmented Generation chatbot that answers questions from your PDFs — with citations, not guesses.**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-8E75B2)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-2E8B57)

</div>

---

## Why this project

Most "chat with your PDF" demos fall apart in two places: they hallucinate when the answer isn't actually in the document, and they forget what you asked two messages ago. This project was built to solve both.

- Every answer is generated **only** from retrieved chunks — the prompt explicitly instructs Gemini to say *"I couldn't find the answer in the uploaded PDF"* rather than fill gaps with outside knowledge.
- Conversation history is fed back into the model on every turn, so follow-ups like *"what about its examples?"* resolve correctly instead of being answered blind.
- Every claim is traceable — each answer shows exactly which file, page, and chunk it came from.

It started as a single-file, single-PDF prototype (v1) and has been rebuilt incrementally into the multi-document, memory-aware assistant it is now (v2) — see [Version History](#-version-history) below.

---

## ✨ Features

| | |
|---|---|
| 📂 **Multi-PDF upload** | Upload several documents at once; retrieval works across all of them |
| 💬 **Real chat interface** | `st.chat_message` / `st.chat_input`, persistent history in-session |
| 🧵 **Conversation memory** | Gemini sees recent turns to understand follow-up questions |
| 📌 **Grounded, cited answers** | Every response cites its source filename, page, and chunk |
| 🚫 **Refuses to hallucinate** | Explicitly says when the PDF doesn't contain the answer |
| 🧭 **Sidebar document manager** | Live file/page/chunk counts, embedding model, Clear Chat / Clear Database |
| ⏳ **Visible loading states** | Spinners for both PDF processing and answer generation |
| 🎨 **Custom UI theme** | Styled chat bubbles, gradient header, sidebar cards |

---

## 🏗️ How it works

```
 Upload PDF(s)
      │
      ▼
 pypdf extracts text, page by page
      │
      ▼
 RecursiveCharacterTextSplitter chunks each page
      │
      ▼
 Sentence-Transformers embeds each chunk  (all-MiniLM-L6-v2)
      │
      ▼
 ChromaDB stores {text, embedding, file, page, chunk}
      │
      ▼
 ── User asks a question ──
      │
      ▼
 Query embedded → top-5 chunks retrieved by similarity → best 3 kept
      │
      ▼
 Gemini answers using ONLY the retrieved chunks
 + recent chat history (for context, never as a source of facts)
      │
      ▼
 Answer + source citations rendered in the chat
```

---

## 🧱 Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| PDF parsing | pypdf |
| Chunking | LangChain Text Splitters |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent, local) |
| LLM | Google Gemini via the official `google-genai` SDK |

---

## 📁 Project Structure

```
AI-PDF-ChatBot-v2/
├── app.py                 # Streamlit UI — upload, processing, sidebar, chat
├── requirements.txt
├── .env.example
├── .gitignore
├── database/               # ChromaDB persistent storage (gitignored)
├── assets/
└── utils/
    ├── embeddings.py       # Embedding creation
    ├── vector_store.py     # ChromaDB storage + clear_database()
    ├── retriever.py        # Similarity search
    └── gemini.py           # Prompt construction + generation
```

---

## 🚀 Getting Started

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/AI-PDF-ChatBot-v2.git
cd AI-PDF-ChatBot-v2

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# then open .env and add your key from https://aistudio.google.com/apikey

# 5. Run
streamlit run app.py
```

Open the local URL Streamlit prints, upload a PDF, and start asking questions.

---

## 🗺️ Version History

**v2 — current**
Multi-PDF support · chat UI with persistent history · conversation memory · sidebar document manager · source citations · `google-genai` SDK migration · custom theme

**v1**
Single-PDF upload · text extraction · chunking · embeddings · ChromaDB storage · semantic search · single-turn Gemini Q&A

**Next up**
Retrieval tuning (chunk-size experiments, re-ranking) · prompt refinement for edge cases · deployment guide

---

## 👨‍💻 Author

**Vishist Chhetri**
Computer Science Engineering Student

---

## 📄 License

This project is intended for educational and learning purposes.