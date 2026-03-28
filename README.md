# 🔮 CodeWhisper

**RAG-powered GitHub Repository Chat & Summarization Tool**

CodeWhisper lets you paste any public GitHub repo URL and instantly chat with the entire codebase using AI. It fetches the source code, builds a searchable vector index, generates an auto-summary, and provides a conversational interface powered by Groq's LLaMA 3 model.

---

## ✨ Features

- **Repo Ingestion** — Fetches up to 60 source files from any public GitHub repo via the REST API
- **Auto Summary** — Generates a structured summary card with tech stack, folder structure, complexity, and entry point detection
- **RAG Pipeline** — Chunks code with LangChain, embeds with HuggingFace `all-MiniLM-L6-v2`, stores in FAISS for fast retrieval
- **AI Chat** — Ask questions about the codebase and get precise, source-attributed answers via Groq (LLaMA 3 8B)
- **Quick Actions** — One-click buttons to explain the repo, find auth flows, understand folder structure, or list core functions
- **Source Attribution** — Every answer shows exactly which files the information came from
- **Dark Developer Theme** — GitHub-inspired dark UI with accent colors

---

## 📸 Screenshot

<!-- Replace with an actual screenshot -->
![CodeWhisper Screenshot](screenshot.png)

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/codewhisper.git
cd codewhisper
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# or
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your Groq API key

1. Go to [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign up / log in and create a new API key
3. Copy the key
4. Create a `.env` file in the project root:

```bash
cp .env.example .env
```

5. Open `.env` and replace the placeholder with your actual key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🛠️ Tech Stack

| Component        | Technology                              |
| ---------------- | --------------------------------------- |
| Frontend/Backend | Python + Streamlit                      |
| Code Chunking    | LangChain RecursiveCharacterTextSplitter|
| Embeddings       | HuggingFace `all-MiniLM-L6-v2`         |
| Vector Store     | FAISS (CPU)                             |
| LLM              | Groq API — `llama3-8b-8192`            |
| GitHub API       | REST API via `requests`                 |

---

## 📁 Project Structure

```
codewhisper/
├── app.py              # Main application (single file)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .env                # Your actual API key (not committed)
└── README.md           # This file
```

---

## ⚠️ Notes

- **Public repos only** — Private repos require authentication tokens (not supported in this version)
- **GitHub rate limits** — Unauthenticated API requests are limited to 60/hour. The app will display a clear message if you hit the limit
- **First run** — The embedding model (~80MB) will be downloaded on first use. Subsequent runs use the cached version
- **File limit** — Repos are capped at 60 files to keep indexing fast. Root and `src/` files are prioritized

---

## 📝 License

MIT
