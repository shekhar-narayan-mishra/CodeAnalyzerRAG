"""
CodeWhisper — RAG-powered GitHub Repository Chat & Summarization Tool
Built with Streamlit, LangChain, FAISS, HuggingFace, and Groq.
"""

import os
import re
import time
import json
import hashlib
from pathlib import PurePosixPath
from collections import Counter, defaultdict

import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path)

# Try getting the key from the system environment first (local testing)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Fallback to Streamlit secrets (for Streamlit Community Cloud)
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

GROQ_MODEL = "llama-3.1-8b-instant"

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
    ".md", ".json", ".yaml", ".yml", ".txt",
}
ALLOWED_BASENAMES = {".env.example"}

SKIP_DIRS = {"node_modules", ".git", "dist", "build", "__pycache__", ".next", "venv", ".venv", "env"}
MAX_FILE_SIZE = 50 * 1024  # 50 KB
MAX_FILES = 60

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a senior engineer analyzing a codebase. "
    "Answer only based on the provided code context. "
    "Be concise, technical, and always mention which part of the code supports your answer."
)

EXT_TO_LANG = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".jsx": "React JSX", ".tsx": "React TSX", ".html": "HTML",
    ".css": "CSS", ".md": "Markdown", ".json": "JSON",
    ".yaml": "YAML", ".yml": "YAML", ".txt": "Text",
}

# ──────────────────────────────────────────────
# SVG Icons (Lucide-style, 18px)
# ──────────────────────────────────────────────

ICONS = {
    "logo": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 16 4-4-4-4"/><path d="m6 8-4 4 4 4"/><path d="m14.5 4-5 16"/></svg>',
    "git_branch": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#8b949e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>',
    "rocket": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg>',
    "trash": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>',
    "summary": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>',
    "zap": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "file_text": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    "shield": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "folder": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>',
    "settings": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>',
    "message": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
    "hash": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#8b949e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="9" x2="20" y2="9"/><line x1="4" y1="15" x2="20" y2="15"/><line x1="10" y1="3" x2="8" y2="21"/><line x1="16" y1="3" x2="14" y2="21"/></svg>',
    "layers": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#8b949e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "globe": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
    "terminal": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>',
    "search": '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#30363d" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>',
}


# ──────────────────────────────────────────────
# Page configuration & CSS
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="CodeWhisper",
    page_icon="</>"  ,
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0e0c;
    --bg-secondary: #131b16;
    --bg-tertiary: #1b261f;
    --bg-card: rgba(19, 27, 22, 0.85);
    --border-color: #24352a;
    --border-subtle: #1b261f;
    --text-primary: #eaf5ec;
    --text-secondary: #90a597;
    --text-muted: #647a6d;
    --accent: #05d96f;
    --accent-muted: #03a855;
    --accent-subtle: rgba(5, 217, 111, 0.12);
    --green: #05d96f;
    --orange: #eab308;
    --red: #ef4444;
    --purple: #06b6d4;
}

html, body, [class*="css"], .stMarkdown, .stText, p, span, li, a {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

.stApp {
    background-color: var(--bg-primary);
}

/* Hide Streamlit "Press Enter to apply" hint */
div[data-testid="InputInstructions"] {
    display: none !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
}

section[data-testid="stSidebar"] .stTextInput input {
    background-color: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    font-size: 0.88em;
    padding: 10px 14px;
    transition: border-color 0.2s ease;
}

section[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-subtle);
}

section[data-testid="stSidebar"] .stTextInput label {
    font-size: 0.82em;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Summary Card */
.summary-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.summary-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--purple), var(--accent));
    opacity: 0.7;
}

.summary-card .card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}

.summary-card .card-title {
    color: var(--text-primary);
    font-size: 1em;
    font-weight: 600;
    letter-spacing: -0.01em;
}

.summary-card .card-body {
    color: var(--text-primary);
    font-size: 0.95em;
    line-height: 1.65;
    margin-bottom: 18px;
}

.summary-card .meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

.summary-card .meta-item {
    padding: 0;
}

.summary-card .meta-label {
    color: var(--text-muted);
    font-weight: 500;
    font-size: 0.72em;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.summary-card .meta-value {
    color: var(--text-secondary);
    font-size: 0.88em;
    line-height: 1.5;
}

/* Tech Badge */
.tech-badge {
    display: inline-flex;
    align-items: center;
    background-color: var(--accent-subtle);
    color: var(--accent);
    border: 1px solid rgba(56, 139, 253, 0.2);
    border-radius: 6px;
    padding: 3px 10px;
    margin: 2px 3px;
    font-size: 0.78em;
    font-weight: 500;
    letter-spacing: 0.01em;
}

/* Source Chip */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background-color: var(--bg-tertiary);
    color: var(--text-muted);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px 3px;
    font-size: 0.73em;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
}

/* Complexity badges */
.complexity-small {
    color: var(--green);
    font-weight: 600;
    background: rgba(63, 185, 80, 0.1);
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.82em;
}
.complexity-medium {
    color: var(--orange);
    font-weight: 600;
    background: rgba(210, 153, 34, 0.1);
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.82em;
}
.complexity-large {
    color: var(--red);
    font-weight: 600;
    background: rgba(248, 81, 73, 0.1);
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.82em;
}

/* Stat Box */
.stat-box {
    background-color: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    text-align: center;
}

.stat-box .stat-number {
    font-size: 1.6em;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.02em;
}

.stat-box .stat-label {
    font-size: 0.72em;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
}

/* Language item */
.lang-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    font-size: 0.85em;
}

.lang-name {
    color: var(--text-secondary);
    font-weight: 500;
}

.lang-count {
    color: var(--text-muted);
    font-size: 0.85em;
    font-family: 'JetBrains Mono', monospace;
}

/* Chat messages */
div[data-testid="stChatMessage"] {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
}

/* Buttons */
.stButton > button {
    background-color: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    font-size: 0.85em;
    font-weight: 500;
    padding: 8px 16px;
    transition: all 0.2s ease;
    letter-spacing: 0.01em;
}

.stButton > button:hover {
    background-color: var(--accent-muted);
    border-color: var(--accent);
    color: #ffffff;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(31, 111, 235, 0.25);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Code blocks */
code, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Title area */
.title-area {
    text-align: center;
    padding: 16px 0 8px 0;
}

.title-area .title-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-bottom: 4px;
}

.title-area h1 {
    background: linear-gradient(135deg, var(--accent) 0%, var(--purple) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 1.8em;
    margin: 0;
    letter-spacing: -0.03em;
}

.title-area .subtitle {
    color: var(--text-muted);
    font-size: 0.88em;
    font-weight: 400;
    margin-top: 2px;
    letter-spacing: 0.02em;
}

/* Sidebar branding */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0;
}

.sidebar-brand .brand-name {
    font-size: 1.15em;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}

.sidebar-brand .brand-sub {
    color: var(--text-muted);
    font-size: 0.8em;
    margin-top: 2px;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
}

.section-header .section-title {
    font-size: 0.88em;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}

/* Welcome state */
.welcome-state {
    text-align: center;
    padding: 80px 20px;
}

.welcome-state .welcome-icon {
    margin-bottom: 16px;
    opacity: 0.4;
}

.welcome-state .welcome-title {
    font-size: 1.05em;
    color: var(--text-secondary);
    font-weight: 500;
    margin-bottom: 8px;
}

.welcome-state .welcome-sub {
    font-size: 0.85em;
    color: var(--text-muted);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Divider */
.subtle-divider {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 16px 0;
}

/* Footer */
.sidebar-footer {
    color: var(--text-muted);
    font-size: 0.7em;
    text-align: center;
    opacity: 0.6;
    letter-spacing: 0.03em;
}

/* Quick action specific */
.qa-btn-label {
    display: flex;
    align-items: center;
    gap: 6px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────

def _init_state():
    defaults = {
        "files": [],
        "repo_meta": {},
        "summary": "",
        "vectorstore": None,
        "chat_history": [],
        "repo_loaded": False,
        "loading": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ──────────────────────────────────────────────
# GitHub helpers
# ──────────────────────────────────────────────

def parse_repo_url(url: str):
    """Extract owner/repo from a GitHub URL. Returns (owner, repo) or None."""
    url = url.strip().rstrip("/")
    patterns = [
        r"github\.com/([^/]+)/([^/]+?)(?:\.git)?$",
        r"^([^/]+)/([^/]+)$",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1), m.group(2)
    return None


def _should_include(path: str) -> bool:
    """Check if a file path should be included based on extension and directory rules."""
    parts = PurePosixPath(path).parts
    for part in parts:
        if part in SKIP_DIRS:
            return False
    basename = PurePosixPath(path).name
    if basename in ALLOWED_BASENAMES:
        return True
    ext = PurePosixPath(path).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def _priority_key(path: str) -> int:
    """Lower value = higher priority. Root and src/ files first."""
    depth = path.count("/")
    if depth == 0:
        return 0
    if path.startswith("src/"):
        return 1
    if path.startswith("app/") or path.startswith("lib/") or path.startswith("pages/"):
        return 2
    return depth + 3


def fetch_repo_files(owner: str, repo: str):
    """
    Fetch file list from GitHub using the Git Trees API (recursive).
    Returns list of dicts: {path, content, size, language}.
    """
    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    headers = {"Accept": "application/vnd.github.v3+json"}

    resp = requests.get(tree_url, headers=headers, timeout=15)

    if resp.status_code == 404:
        raise ValueError(
            "Repository not found. Make sure the URL is correct and the repo is public."
        )
    if resp.status_code == 403:
        remaining = resp.headers.get("X-RateLimit-Remaining", "?")
        if remaining == "0":
            raise ValueError("GitHub rate limit reached. Please try again in 60 seconds.")
        raise ValueError("Access forbidden — the repository may be private.")
    if resp.status_code != 200:
        raise ValueError(f"GitHub API error (HTTP {resp.status_code}).")

    tree = resp.json().get("tree", [])

    # Filter blobs matching our criteria
    candidates = [
        node for node in tree
        if node["type"] == "blob"
        and _should_include(node["path"])
        and node.get("size", 0) <= MAX_FILE_SIZE
    ]

    # Sort by priority and cap
    candidates.sort(key=lambda n: _priority_key(n["path"]))
    candidates = candidates[:MAX_FILES]

    files = []
    progress_bar = st.progress(0, text="Fetching repository files...")
    total = len(candidates)

    for idx, node in enumerate(candidates):
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{node['path']}"
        try:
            r = requests.get(raw_url, timeout=10)
            if r.status_code == 200:
                ext = PurePosixPath(node["path"]).suffix.lower()
                files.append({
                    "path": node["path"],
                    "filename": PurePosixPath(node["path"]).name,
                    "content": r.text,
                    "size": node.get("size", len(r.text)),
                    "language": EXT_TO_LANG.get(ext, "Unknown"),
                })
        except requests.RequestException:
            pass
        progress_bar.progress((idx + 1) / total, text=f"Fetching {node['path']}")

    progress_bar.empty()

    if not files:
        raise ValueError("No supported source files found in this repository.")

    return files


# ──────────────────────────────────────────────
# Repo analysis & summary
# ──────────────────────────────────────────────

def analyse_repo(files: list) -> dict:
    """Build metadata from fetched files."""
    lang_counter = Counter(f["language"] for f in files if f["language"] != "Unknown")
    top_dirs = sorted({PurePosixPath(f["path"]).parts[0] for f in files if "/" in f["path"]})
    root_files = [f["filename"] for f in files if "/" not in f["path"]]

    # Detect tech stack from known manifest files
    tech_stack = set(lang_counter.keys())
    for f in files:
        if f["filename"] == "package.json":
            try:
                pkg = json.loads(f["content"])
                deps = list(pkg.get("dependencies", {}).keys()) + list(pkg.get("devDependencies", {}).keys())
                for d in deps:
                    dl = d.lower()
                    if "react" in dl:
                        tech_stack.add("React")
                    if "next" in dl:
                        tech_stack.add("Next.js")
                    if "vue" in dl:
                        tech_stack.add("Vue")
                    if "express" in dl:
                        tech_stack.add("Express")
                    if "tailwind" in dl:
                        tech_stack.add("Tailwind CSS")
                    if "vite" in dl:
                        tech_stack.add("Vite")
                    if "angular" in dl:
                        tech_stack.add("Angular")
                    if "svelte" in dl:
                        tech_stack.add("Svelte")
                    if "fastapi" in dl:
                        tech_stack.add("FastAPI")
            except (json.JSONDecodeError, AttributeError):
                pass
        if f["filename"] == "requirements.txt":
            lines = f["content"].lower().splitlines()
            for line in lines:
                if "django" in line:
                    tech_stack.add("Django")
                if "flask" in line:
                    tech_stack.add("Flask")
                if "fastapi" in line:
                    tech_stack.add("FastAPI")
                if "streamlit" in line:
                    tech_stack.add("Streamlit")
                if "tensorflow" in line or "torch" in line:
                    tech_stack.add("ML/AI")

    # Detect entry points
    entry_point = None
    entry_candidates = [
        "main.py", "app.py", "index.py", "manage.py",
        "index.js", "index.ts", "main.js", "main.ts",
        "index.html", "App.jsx", "App.tsx",
    ]
    for ec in entry_candidates:
        for f in files:
            if f["filename"] == ec:
                entry_point = f["path"]
                break
        if entry_point:
            break

    # Complexity
    fc = len(files)
    if fc <= 10:
        complexity = "Small"
    elif fc <= 30:
        complexity = "Medium"
    else:
        complexity = "Large"

    return {
        "file_count": fc,
        "languages": dict(lang_counter.most_common()),
        "tech_stack": sorted(tech_stack),
        "top_dirs": top_dirs,
        "root_files": root_files,
        "entry_point": entry_point,
        "complexity": complexity,
    }


def generate_summary(files: list, meta: dict) -> str:
    """Use Groq LLM to generate a concise project summary."""
    filenames = [f["path"] for f in files]
    readme_content = ""
    for f in files:
        if f["filename"].lower().startswith("readme"):
            readme_content = f["content"][:2000]
            break

    prompt = (
        "Given the following information about a GitHub repository, write a 2-3 sentence summary of what this project does.\n\n"
        f"Files ({len(filenames)}): {', '.join(filenames[:40])}\n"
        f"Tech stack: {', '.join(meta['tech_stack'])}\n"
        f"Entry point: {meta.get('entry_point', 'unknown')}\n"
    )
    if readme_content:
        prompt += f"\nREADME excerpt:\n{readme_content}\n"
    prompt += "\nProvide only the summary, no headings or bullet points."

    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise technical writer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return "Summary generation unavailable. Please check your Groq API key."


# ──────────────────────────────────────────────
# RAG pipeline
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def build_vectorstore(files: list):
    """Chunk files and build a FAISS vectorstore."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    texts, metadatas = [], []
    for f in files:
        chunks = splitter.split_text(f["content"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                "filename": f["filename"],
                "filepath": f["path"],
                "language": f["language"],
            })

    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore


def retrieve_context(query: str, vectorstore) -> tuple[str, list[str]]:
    """Retrieve top-K chunks and return formatted context + source filenames."""
    docs = vectorstore.similarity_search(query, k=TOP_K)
    context_parts = []
    sources = []
    for doc in docs:
        fp = doc.metadata.get("filepath", "unknown")
        lang = doc.metadata.get("language", "")
        context_parts.append(f"### File: {fp} ({lang})\n```\n{doc.page_content}\n```")
        if fp not in sources:
            sources.append(fp)
    return "\n\n".join(context_parts), sources


# ──────────────────────────────────────────────
# Groq chat
# ──────────────────────────────────────────────

def ask_groq(query: str, context: str) -> str:
    """Send query + retrieved context to Groq and return the response."""
    if not GROQ_API_KEY:
        return "Groq API key not found. Please add GROQ_API_KEY to your .env file."

    user_message = (
        f"Code context:\n{context}\n\n"
        f"User question: {query}"
    )
    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=1024,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq API error: {e}"


# ──────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────

def render_summary_card(summary: str, meta: dict):
    tech_badges = "".join(f'<span class="tech-badge">{t}</span>' for t in meta.get("tech_stack", []))

    complexity_cls = f"complexity-{meta.get('complexity', 'small').lower()}"
    complexity_val = meta.get("complexity", "Unknown")

    dirs_html = " ".join(
        f'<span class="tech-badge" style="color:var(--text-secondary);background:var(--bg-tertiary);border-color:var(--border-color);">{d}/</span>'
        for d in meta.get("top_dirs", [])[:12]
    ) if meta.get("top_dirs") else '<span style="color:var(--text-muted);font-style:italic;">Flat structure</span>'

    entry_html = (
        f'<code style="color:var(--accent);background:var(--accent-subtle);padding:2px 8px;border-radius:4px;font-size:0.88em;">{meta["entry_point"]}</code>'
        if meta.get("entry_point")
        else '<span style="color:var(--text-muted);font-style:italic;">Not detected</span>'
    )

    html = f"""
    <div class="summary-card">
        <div class="card-header">
            {ICONS['summary']}
            <span class="card-title">Repository Analysis</span>
        </div>
        <div class="card-body">{summary}</div>
        <hr class="subtle-divider">
        <div class="meta-grid">
            <div class="meta-item">
                <div class="meta-label">Tech Stack</div>
                <div class="meta-value">{tech_badges}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Directories</div>
                <div class="meta-value">{dirs_html}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Complexity</div>
                <div class="meta-value">
                    <span class="{complexity_cls}">{complexity_val}</span>
                    <span style="color:var(--text-muted);font-size:0.85em;margin-left:8px;">{meta.get('file_count', 0)} files indexed</span>
                </div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Entry Point</div>
                <div class="meta-value">{entry_html}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_source_chips(sources: list[str]):
    chips = "".join(
        f'<span class="source-chip">{ICONS["hash"]} {PurePosixPath(s).name}</span>'
        for s in sources
    )
    st.markdown(
        f'<div style="margin-top:6px;padding-top:6px;border-top:1px solid var(--border-subtle);">'
        f'<span style="color:var(--text-muted);font-size:0.7em;text-transform:uppercase;letter-spacing:0.06em;margin-right:6px;">Sources</span>'
        f'{chips}</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            {ICONS['logo']}
            <div>
                <div class="brand-name">CodeWhisper</div>
                <div class="brand-sub">Repository Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    repo_url = st.text_input(
        "Repository URL",
        placeholder="https://github.com/owner/repo",
        key="repo_url_input",
    )

    col_load, col_clear = st.columns(2)
    with col_load:
        load_btn = st.button("Load Repo", use_container_width=True)
    with col_clear:
        clear_btn = st.button("Clear", use_container_width=True)

    if clear_btn:
        for k in ["files", "repo_meta", "summary", "vectorstore", "chat_history", "repo_loaded", "loading"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    # Show stats when repo loaded
    if st.session_state.get("repo_loaded") and st.session_state.get("repo_meta"):
        meta = st.session_state["repo_meta"]
        st.markdown("---")
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-number">{meta['file_count']}</div>
                <div class="stat-label">Files Indexed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        langs = meta.get("languages", {})
        if langs:
            st.markdown(
                f'<div style="margin-top:12px;margin-bottom:8px;">'
                f'<span style="color:var(--text-muted);font-size:0.72em;text-transform:uppercase;letter-spacing:0.06em;font-weight:500;">Languages Detected</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for lang, count in langs.items():
                st.markdown(
                    f'<div class="lang-item">'
                    f'<span class="lang-name">{lang}</span>'
                    f'<span class="lang-count">{count}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown(
        '<p class="sidebar-footer">Streamlit / LangChain / FAISS / Groq</p>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Repo loading logic
# ──────────────────────────────────────────────

if load_btn and repo_url:
    parsed = parse_repo_url(repo_url)
    if not parsed:
        st.error("Invalid GitHub URL. Please use a format like https://github.com/owner/repo")
    else:
        owner, repo = parsed
        try:
            with st.spinner("Connecting to GitHub..."):
                files = fetch_repo_files(owner, repo)

            st.session_state["files"] = files
            meta = analyse_repo(files)
            st.session_state["repo_meta"] = meta

            with st.spinner("Generating repository summary..."):
                summary = generate_summary(files, meta)
            st.session_state["summary"] = summary

            with st.spinner("Building vector index (this may take a moment on first run)..."):
                vs = build_vectorstore(files)
            st.session_state["vectorstore"] = vs
            st.session_state["repo_loaded"] = True
            st.session_state["chat_history"] = []
            st.rerun()

        except ValueError as e:
            st.error(str(e))
        except requests.ConnectionError:
            st.error("Network error — could not reach GitHub. Check your internet connection.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


# ──────────────────────────────────────────────
# Main content area
# ──────────────────────────────────────────────

st.markdown(
    f"""
    <div class="title-area">
        <div class="title-row">
            {ICONS['terminal']}
            <h1>CodeWhisper</h1>
        </div>
        <div class="subtitle">RAG-powered repository analysis and chat</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.get("repo_loaded"):
    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="welcome-state">
            <div class="welcome-icon">{ICONS['search']}</div>
            <div class="welcome-title">Paste a public GitHub repository URL in the sidebar to get started</div>
            <div class="welcome-sub">
                CodeWhisper will fetch the source code, build a searchable vector index,
                and let you chat with the entire codebase using AI-powered retrieval.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Summary card ──
render_summary_card(st.session_state["summary"], st.session_state["repo_meta"])

# ── Quick action buttons ──
st.markdown(
    f"""
    <div class="section-header">
        {ICONS['zap']}
        <span class="section-title">Quick Actions</span>
    </div>
    """,
    unsafe_allow_html=True,
)

qa_cols = st.columns(4)
quick_prompts = [
    ("Explain Repo", "Explain this repo to a beginner in simple terms. What does it do, what are the main files, and how is it structured?"),
    ("Auth Flow", "How does authentication work in this codebase? Describe the auth flow, mention relevant files and middleware."),
    ("Folder Structure", "What is the folder structure logic? Explain the purpose of each major directory and how the project is organized."),
    ("Core Functions", "What are the core functions and components in this codebase? List the most important ones and describe what they do."),
]

for col, (label, prompt) in zip(qa_cols, quick_prompts):
    with col:
        if st.button(label, use_container_width=True, key=f"qa_{label}"):
            st.session_state["_pending_query"] = prompt

# ── Chat history display ──
st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="section-header">
        {ICONS['message']}
        <span class="section-title">Chat</span>
    </div>
    """,
    unsafe_allow_html=True,
)

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_source_chips(msg["sources"])

# ── Handle pending quick-action query ──
pending = st.session_state.pop("_pending_query", None)

# ── Chat input ──
user_input = st.chat_input("Ask anything about the codebase...")

query = pending or user_input

if query:
    # Show user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state["chat_history"].append({"role": "user", "content": query})

    # Retrieve & answer
    vs = st.session_state["vectorstore"]
    context, sources = retrieve_context(query, vs)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing codebase..."):
            answer = ask_groq(query, context)
        # Simulate streaming output
        placeholder = st.empty()
        displayed = ""
        for i, char in enumerate(answer):
            displayed += char
            if i % 3 == 0 or i == len(answer) - 1:
                placeholder.markdown(displayed + "▌")
                time.sleep(0.008)
        placeholder.markdown(answer)
        render_source_chips(sources)

    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
