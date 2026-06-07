# 🔁 Self-Healing RAG — Document Q&A System

A production-grade, agentic Retrieval-Augmented Generation (RAG) system that critiques its own answers and self-corrects using a LangGraph stateful workflow. Built with LangChain, ChromaDB, Google Gemini API, and deployed on Render with a Streamlit UI.

---

## 🚀 Live Demo

🔗 [Try it on Render](https://rag-app-t8dv.onrender.com)

---

## 📌 What Makes This Different From Basic RAG?

Most RAG systems just retrieve → generate → return. This system goes further:

| Step | Basic RAG | Self-Healing RAG |
|------|-----------|-----------------|
| Retrieve chunks | ✅ | ✅ |
| Generate answer | ✅ | ✅ |
| Critic evaluates grounding | ❌ | ✅ |
| Reformulate query on failure | ❌ | ✅ |
| Honest refusal if ungrounded | ❌ | ✅ |

---

## 🏗️ Architecture

```
User Question
      ↓
  [Retrieve Node] → Fetch top-3 chunks from ChromaDB
      ↓
  [Generate Node] → LLM generates answer from context
      ↓
  [Critic Node] → "Is this answer grounded in the retrieved chunks?"
      ↓
   YES → [Finalize] → Return verified answer ✅
   NO  → [Reformulate Node] → Rewrite query → Retry (max 2x)
      ↓
   Still failing → "I don't have enough information" ⚠️
```

Built as a **stateful, cyclical LangGraph workflow** — not a simple linear chain.

---

## 📊 Evaluation Results (RAGAS)

Evaluated on the *"Attention Is All You Need"* paper using the RAGAS framework.

| Metric | Q1 | Q2 | Q3 | Average |
|--------|----|----|-----|---------|
| Faithfulness | 1.0 | 1.0 | 1.0 | **1.0** |
| Answer Relevancy | 0.53 | 0.38 | 0.78 | **0.56** |

**Faithfulness: 1.0 / 1.0** — Zero hallucination across all test queries.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Google Gemini (via LangChain) |
| Agentic Workflow | LangGraph (stateful, cyclical graph) |
| Embeddings | FastEmbed (lightweight, no torch) |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| Evaluation | RAGAS (Faithfulness, Answer Relevancy) |
| Frontend | Streamlit |
| Deployment | Render |

---

## 📁 Project Structure

```
rag-qa-project/
├── app.py            # Streamlit chat UI
├── rag_chain.py      # LangGraph graph definition
├── nodes.py          # Retrieve, Generate, Critic, Reformulate, Finalize nodes
├── prompts.py        # All LLM prompt templates
├── state.py          # RAGState TypedDict
├── embeddings.py     # ChromaDB vector store setup
├── ingest.py         # Document ingestion and chunking
├── eval_data.py      # RAGAS evaluation script
├── requirements.txt  # Dependencies
├── render.yaml       # Render deployment config
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

### 5. Run the app
```bash
streamlit run app.py
```

### 6. Run evaluation
```bash
python eval_data.py
```

---

## 🔁 Self-Healing Behavior — Example

**Question:** "What is the stock price of Google?"
*(Asked on the Attention Is All You Need paper)*

- **Basic RAG:** Hallucinated answer or irrelevant response
- **Self-Healing RAG:**
  1. Retrieves chunks → generates answer
  2. Critic detects answer is not grounded
  3. Reformulates query → retries
  4. Still not grounded → returns: *"I don't have enough information in the document to answer this accurately."*

---

## 🌐 Deployment

Deployed on **Render** using `render.yaml`. Every push to `master` triggers an automatic redeploy.

**Start command:**
```
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 🔑 Key Features

- **Zero hallucination** — Faithfulness score of 1.0 on RAGAS evaluation
- **Self-healing pipeline** — Critic agent validates every answer before returning
- **Query reformulation** — Automatically rewrites failed queries and retries
- **Honest refusal** — Returns "I don't have enough information" instead of making things up
- **Modular codebase** — Prompts, nodes, state, and graph logic separated into individual files
- **Lightweight deployment** — Uses FastEmbed instead of torch for Render free tier compatibility
- **Any PDF** — Upload any document and ask questions about it

---

## 👩‍💻 Author

**Muchumarri Hemalatha** — AI Engineer
📍 Bengaluru, India
🔗 [GitHub](https://github.com/hema123-4)
