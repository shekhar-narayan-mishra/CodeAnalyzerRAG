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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
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
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --bg-card: rgba(22, 27, 34, 0.85);
    --border-color: #30363d;
    --border-subtle: #21262d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #484f58;
    --accent: #58a6ff;
    --accent-muted: #1f6feb;
    --accent-subtle: rgba(56, 139, 253, 0.1);
    --green: #3fb950;
    --orange: #d29922;
    --red: #f85149;
    --purple: #bc8cff;
}

html, body, [class*="css"], .stMarkdown, .stText, p, span, li, a {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

.stApp {
    background-color: var(--bg-primary);
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
