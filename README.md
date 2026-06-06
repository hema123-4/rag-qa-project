# 📄 RAG-Powered Document Q&A System

A production-ready Retrieval-Augmented Generation (RAG) pipeline that answers questions from documents with zero hallucination. Built with LangChain, ChromaDB, and Google Gemini API — evaluated using the RAGAS framework.

---

## 🚀 Live Demo

🔗 [Try it on Render](https://rag-app-t8dv.onrender.com)

---

## 📌 Project Overview

This project implements a full RAG pipeline that:
- Ingests and chunks PDF documents
- Embeds and stores them in a ChromaDB vector store
- Retrieves relevant context for any user question
- Generates grounded answers using Google Gemini LLM
- Evaluates answer quality using RAGAS metrics

---

## 🏗️ Architecture

```
User Question
      ↓
 Retriever (ChromaDB + HuggingFace Embeddings)
      ↓
 Relevant Context Chunks
      ↓
 Prompt Template
      ↓
 Gemini LLM (gemini-3.1-flash-lite)
      ↓
 Grounded Answer with Source Reference
```

---

## 📊 Evaluation Results (RAGAS)

Evaluated on the *"Attention Is All You Need"* paper using the RAGAS framework.

| Metric | Q1 | Q2 | Q3 | Average |
|--------|----|----|-----|---------|
| Faithfulness | 1.0 | 1.0 | 1.0 | **1.0** |
| Answer Relevancy | 0.53 | 0.38 | 0.78 | **0.56** |

**Faithfulness: 1.0 / 1.0** — The system never hallucinated. Every answer was fully grounded in the retrieved document context.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Google Gemini (via LangChain) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| Evaluation | RAGAS (Faithfulness, Answer Relevancy) |
| Frontend | Streamlit |
| Deployment | Render |

---

## 📁 Project Structure

```
rag-qa-project/
├── app.py              # Streamlit chat UI
├── rag_chain.py        # RAG pipeline (retriever + LLM chain)
├── embeddings.py       # ChromaDB vector store setup
├── ingest.py           # Document ingestion and chunking
├── eval_data.py        # RAGAS evaluation script
├── requirements.txt    # Dependencies
├── render.yaml         # Render deployment config
└── .gitignore
```

---

## ⚙️ Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/hema123-4/rag-qa-project.git
cd rag-qa-project
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Ingest your document
```bash
python ingest.py
```

### 6. Run the app
```bash
streamlit run app.py
```

### 7. Run evaluation
```bash
python eval_data.py
```

---

## 🌐 Deployment

Deployed on **Render** using `render.yaml`. Every push to `master` triggers an automatic redeploy.

---

## 🔑 Key Features

- **Zero hallucination** — Faithfulness score of 1.0 on RAGAS evaluation
- **Source references** — Every answer cites which part of the document supports it
- **Multi-chunk retrieval** — Retrieves top-3 relevant chunks per query
- **Chat UI** — Clean Streamlit interface with conversation history
- **Production-ready** — Deployed on Render with environment variable management

---

## 👩‍💻 Author

**Hema** — AI Engineer  
📍 Bengaluru, India  
🔗 [GitHub](https://github.com/hema123-4)
