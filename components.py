"""All Streamlit presentation components for RAG Explorer."""

from __future__ import annotations

import html
import logging
import time
from collections.abc import Sequence

import numpy as np
import streamlit as st

from src.chunker import chunk_documents
from src.embedder import DEFAULT_MODEL, embed_query, embed_texts
from src.experiment_tracker import (
    append_experiment,
    create_experiment_record,
    initialize_experiment_history,
    update_experiment_generation,
    update_experiment_rating,
)
from src.extractive_answer import generate_extractive_answer
from src.generator import (
    GenerationBackendError,
    SYSTEM_PROMPT,
    build_grounded_prompt,
    format_retrieved_context,
    get_gemini_api_key,
    get_model_name,
    generate_answer,
    is_fallback_eligible_generation_error,
)
from src.smart_retriever import smart_retrieve
from src.parser import parse_pdfs
from src.utils import Chunk, Document, SearchResult
from src.vector_store import BruteForceStore, ChromaStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _request_knowledge_base_rebuild() -> None:
    """Lock navigation before applying a new chunking configuration."""
    current = st.session_state.get("knowledge_base_configuration")
    requested = (
        st.session_state.get("chunk_strategy_control"),
        st.session_state.get("chunk_size_control"),
        st.session_state.get("overlap_control"),
    )
    if current is None or tuple(current[1:]) != requested:
        st.session_state["knowledge_base_rebuild_requested"] = True
        st.session_state["knowledge_base_rebuild_existing"] = bool(
            st.session_state.get("knowledge_base_artifacts")
        )
        st.session_state["rebuild_return_tab"] = st.session_state.get(
            "main_navigation_tab", "Knowledge Base"
        )


def render_app_shell() -> None:
    """Apply the product theme and render the animated hero."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');
    :root { --ink:#eaf0ff; --muted:#93a4c7; --violet:#8b5cf6; --cyan:#22d3ee; }
    .stApp { background:
      radial-gradient(circle at 80% 0%, rgba(91,33,182,.20), transparent 34rem),
      radial-gradient(circle at 8% 35%, rgba(8,145,178,.12), transparent 28rem), #070b17; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:#e8eefc; }
    .stApp, .stApp p, .stApp li, .stApp label { color:#cbd5e1; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color:#f8fafc !important; }
    h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.03em; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#0c1224; border-right:1px solid rgba(148,163,184,.22); }
    [data-testid="stSidebar"] * { color:#dbe5f7; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color:#ffffff !important; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#cbd5e1 !important; font-weight:600; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:#9fb0cf !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color:#111827 !important; }
    [data-testid="stSidebar"] [role="slider"] { color:#ffffff !important; }
    [data-testid="stSidebar"] [data-testid="stSlider"] p { color:#e2e8f0 !important; }
    [data-testid="stSidebar"] hr { border-color:rgba(148,163,184,.14); }
    .block-container { max-width:1400px; padding-top:2rem; padding-bottom:5rem; }
    .rag-hero { position:relative; overflow:hidden; padding:2.2rem 2.4rem; border:1px solid rgba(139,92,246,.28);
      border-radius:24px; background:linear-gradient(115deg,rgba(17,24,48,.96),rgba(21,13,48,.82));
      box-shadow:0 24px 80px rgba(0,0,0,.28); margin-bottom:1.4rem; }
    .rag-hero:after { content:''; position:absolute; width:260px;height:260px;right:-50px;top:-110px;border-radius:50%;
      background:linear-gradient(135deg,var(--violet),var(--cyan));filter:blur(65px);opacity:.3;animation:float 5s ease-in-out infinite; }
    .rag-kicker { color:#67e8f9;font-size:.76rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase; }
    .rag-title { color:var(--ink);font-family:'Space Grotesk';font-size:clamp(2.4rem,5vw,4.8rem);line-height:.95;
      letter-spacing:-.065em;font-weight:700;margin:.7rem 0 .8rem;position:relative;z-index:1; }
    .rag-gradient { background:linear-gradient(100deg,#c4b5fd,#67e8f9);-webkit-background-clip:text;color:transparent; }
    .rag-subtitle { color:var(--muted);font-size:1.05rem;max-width:650px;position:relative;z-index:1; }
    .rag-flow { display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;margin-top:1.25rem;position:relative;z-index:1; }
    .rag-node { padding:.38rem .7rem;border-radius:999px;background:rgba(148,163,184,.08);border:1px solid rgba(148,163,184,.14);color:#cbd5e1;font-size:.78rem; }
    .rag-arrow { color:#22d3ee;animation:pulse 1.6s infinite; }
    [data-testid="stTabs"] { overflow:visible !important;contain:none !important; }
    [data-testid="stTabs"] > div:first-child { overflow:visible !important; }
    [data-testid="stTabs"] [role="tablist"] { display:grid !important;grid-template-columns:repeat(7,minmax(0,1fr)) !important;gap:.3rem !important;background:rgba(8,13,26,.98) !important;padding:.55rem .55rem 0 !important;border-radius:16px 16px 0 0 !important;border:1px solid #334155 !important;border-bottom:2px solid #64748b !important;box-shadow:0 12px 32px rgba(0,0,0,.42) !important;position:sticky !important;top:3.75rem !important;z-index:2147483000 !important;overflow:visible !important;backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);pointer-events:auto !important; }
    [data-testid="stTabs"] [role="tabpanel"] { position:relative;z-index:0; }
    [data-testid="stTabs"] [data-testid="stTab"] { box-sizing:border-box !important;width:100% !important;min-width:0 !important;height:3.65rem !important;min-height:3.65rem !important;margin:0 0 -2px !important;padding:.8rem .55rem !important;color:#dbeafe !important;background:linear-gradient(180deg,#202b43,#121a2d) !important;border:1px solid #475569 !important;border-bottom:2px solid #64748b !important;border-radius:12px 12px 0 0 !important;box-shadow:inset 0 1px 0 rgba(255,255,255,.05),0 -2px 9px rgba(0,0,0,.18) !important;opacity:1 !important;pointer-events:auto !important;cursor:pointer !important; }
    [data-testid="stTabs"] [data-testid="stTab"] p { color:#dbeafe !important;font-size:1.05rem !important;line-height:1.15 !important;font-weight:700 !important;white-space:nowrap !important; }
    [data-testid="stTabs"] [data-testid="stTab"]:hover { background:linear-gradient(180deg,#2d3a58,#19233a) !important;border-color:#64748b !important;transform:translateY(-2px); }
    [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] { background:linear-gradient(180deg,#41336f,#1b2745) !important;border-color:#a78bfa !important;border-bottom:2px solid #1b2745 !important;box-shadow:0 -4px 18px rgba(139,92,246,.3),inset 0 3px 0 #67e8f9 !important;position:relative !important;z-index:2 !important; }
    [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] p { color:#ffffff !important;font-size:1.08rem !important; }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { display:none; }
    [data-testid="stFileUploaderDropzone"] { background:#f8fafc; }
    [data-testid="stFileUploaderDropzone"] * { color:#334155 !important; }
    [data-testid="stFileUploaderDropzone"] button:not([aria-label="Add files"]),
    [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] { background:#17213c !important;border:1px solid #334155 !important;border-radius:10px !important;box-shadow:0 4px 12px rgba(15,23,42,.18) !important;opacity:1 !important; }
    [data-testid="stFileUploaderDropzone"] button:not([aria-label="Add files"]) *,
    [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] * { color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;fill:#ffffff !important;stroke:#ffffff !important;font-weight:700 !important;opacity:1 !important; }
    [data-testid="stFileUploaderDropzone"] button:not([aria-label="Add files"]):hover,
    [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"]:hover { background:#263553 !important;border-color:#0891b2 !important; }
    [data-testid="stFileUploader"] label p { color:#dbe5f7 !important; }
    [data-testid="stFileUploaderFile"] { background:#17213c !important;border:1px solid #425277 !important;border-radius:10px !important;padding:.45rem .65rem !important; }
    [data-testid="stFileUploaderFile"] *, [data-testid="stFileUploaderFile"] p,
    [data-testid="stFileUploaderFile"] span, [data-testid="stFileUploaderFileName"] { color:#f8fafc !important;opacity:1 !important; }
    [data-testid="stFileUploaderFile"] small { color:#c7d2e8 !important; }
    [data-testid="stFileUploaderFile"] svg { color:#dbeafe !important;fill:currentColor; }
    [data-testid="stFileUploaderFile"] button { background:#263553 !important;border-radius:50%; }
    [data-testid="stFileUploaderFile"] button:hover { background:#3b4c70 !important; }
    [data-testid="stFileUploaderDropzone"] [data-testid="stFileChip"] { background:#17213c !important;border:1px solid #53678f !important;border-radius:10px !important; }
    [data-testid="stFileUploaderDropzone"] [data-testid="stFileChipName"],
    [data-testid="stFileUploaderDropzone"] .stFileChipName { color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;opacity:1 !important;font-weight:700 !important; }
    [data-testid="stFileUploaderDropzone"] [data-testid="stFileChip"] div { color:#dbeafe !important;-webkit-text-fill-color:#dbeafe !important;opacity:1 !important; }
    [data-testid="stFileUploaderDropzone"] button[aria-label="Add files"] { display:flex !important;align-items:center !important;justify-content:center !important;background:#e0f2fe !important;border:2px solid #0891b2 !important;border-radius:50% !important;width:2rem !important;height:2rem !important;min-width:2rem !important;padding:0 !important;opacity:1 !important; }
    [data-testid="stFileUploaderDropzone"] button[aria-label="Add files"] * { color:#075985 !important;fill:#075985 !important;stroke:#075985 !important;opacity:1 !important; }
    [data-testid="stFileUploaderDropzone"] button[aria-label="Add files"]:hover { background:#a5f3fc !important;transform:scale(1.06); }
    [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(22,29,55,.8),rgba(12,18,36,.75));border:1px solid rgba(148,163,184,.12);border-radius:16px;padding:1rem; }
    [data-testid="stMetricValue"] { color:#e9d5ff; }
    .st-key-chat_question [data-testid="stTextInput"] input { min-height:3.4rem;font-size:1.08rem;border:1px solid rgba(103,232,249,.55);box-shadow:0 0 0 3px rgba(34,211,238,.07); }
    .st-key-chat_question [data-testid="stWidgetLabel"] p { color:#f8fafc !important;font-size:1.05rem;font-weight:700; }
    .st-key-chat_summary [data-testid="stMetric"], .st-key-chat_metrics [data-testid="stMetric"] { padding:.38rem .55rem !important;border-radius:9px !important;min-height:56px !important; }
    .st-key-chat_summary [data-testid="stMetricLabel"] p, .st-key-chat_metrics [data-testid="stMetricLabel"] p { font-size:.66rem !important;line-height:1.05 !important; }
    .st-key-chat_summary [data-testid="stMetricValue"], .st-key-chat_metrics [data-testid="stMetricValue"] { font-size:.98rem !important;line-height:1.1 !important; }
    .st-key-experiment_detail_cards [data-testid="stMetric"] { padding:.38rem .55rem !important;border-radius:9px !important;min-height:56px !important; }
    .st-key-experiment_detail_cards [data-testid="stMetricLabel"] p { font-size:.66rem !important;line-height:1.05 !important; }
    .st-key-experiment_detail_cards [data-testid="stMetricValue"] { font-size:.98rem !important;line-height:1.1 !important; }
    .st-key-experiment_detail_cards [data-testid="stHorizontalBlock"] { gap:.45rem !important; }
    .st-key-experiment_selector [data-baseweb="select"] > div { min-height:3.6rem !important;border:1px solid rgba(181,139,90,.48) !important;font-size:1.02rem !important;box-shadow:0 0 0 3px rgba(181,139,90,.06); }
    .st-key-experiment_selector [data-baseweb="select"] > div, .st-key-experiment_selector [data-baseweb="select"] * { background-color:#111c30 !important;color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;opacity:1 !important; }
    .st-key-experiment_selector [data-testid="stWidgetLabel"] p { color:#f8fafc !important;font-size:1.05rem !important;font-weight:700 !important; }
    .st-key-experiment_rating_form [data-testid="stForm"] { padding:.7rem .85rem !important;border-radius:12px !important; }
    .st-key-experiment_rating_form [data-baseweb="select"] > div { min-height:2.65rem !important; }
    .st-key-experiment_rating_form [data-testid="stWidgetLabel"] p { font-size:.74rem !important; }
    .st-key-experiment_rating_form [data-baseweb="select"] * { font-size:.8rem !important; }
    .st-key-experiment_review_card { background:rgba(14,25,43,.72);border-color:rgba(181,139,90,.24) !important; }
    .chat-banner { position:relative;overflow:hidden;display:flex;align-items:center;gap:1rem;padding:1rem 1.15rem;margin:.2rem 0 1rem;border-radius:16px;background:linear-gradient(115deg,rgba(124,58,237,.18),rgba(8,145,178,.15));border:1px solid rgba(103,232,249,.22); }
    .chat-banner:after { content:'';position:absolute;right:-35px;top:-55px;width:145px;height:145px;border-radius:50%;background:#7c3aed;filter:blur(48px);opacity:.25; }
    .chat-spark { display:grid;place-items:center;width:2.8rem;height:2.8rem;flex:0 0 2.8rem;border-radius:14px;background:linear-gradient(135deg,#7c3aed,#0891b2);font-size:1.35rem;box-shadow:0 0 24px rgba(34,211,238,.22);animation:float 3s ease-in-out infinite; }
    .chat-banner strong { display:block;color:#f8fafc;font-size:1rem; }.chat-banner span { color:#b9c7df;font-size:.88rem; }
    .st-key-chat_cta [data-testid="stFormSubmitButton"] button,
    .st-key-retrieval_cta [data-testid="stFormSubmitButton"] button { min-height:2.85rem !important;font-size:.95rem !important;padding:.6rem 1rem !important;box-shadow:0 8px 28px rgba(124,58,237,.35) !important; }
    [data-testid="stExpander"] { background:rgba(15,23,42,.55);border:1px solid rgba(148,163,184,.12);border-radius:14px;overflow:hidden; }
    .stButton>button { border:0;border-radius:12px;font-weight:700;background:linear-gradient(110deg,#7c3aed,#0891b2);color:white;box-shadow:0 8px 24px rgba(91,33,182,.26);transition:.2s; }
    .stButton>button:hover { transform:translateY(-2px);box-shadow:0 12px 30px rgba(8,145,178,.28);color:white; }
    .strategy-card { padding:1rem 1.1rem;border-radius:14px;border:1px solid rgba(34,211,238,.18);background:linear-gradient(120deg,rgba(8,145,178,.09),rgba(124,58,237,.08));margin:.6rem 0 1rem; }
    .strategy-card strong { color:#a5f3fc;font-size:1rem; }.strategy-card span { color:#9baaca;display:block;margin-top:.25rem; }
    .distance-card { padding:.48rem .5rem;border-radius:9px;border:1px solid rgba(34,211,238,.16);background:linear-gradient(120deg,rgba(8,145,178,.07),rgba(124,58,237,.07));margin:.25rem 0 .45rem;text-align:center;line-height:1.15; }
    .distance-card strong { color:#a5f3fc;font-size:.76rem;display:block; }.distance-card span { color:#9baaca;display:block;font-size:.68rem;margin-top:.1rem; }.distance-card .distance-arrow { font-size:.82rem;line-height:.85;margin:.05rem 0; }
    .scan-stage { padding:1rem;border-radius:14px;background:rgba(7,12,26,.8);border:1px solid rgba(34,211,238,.22);position:relative;overflow:hidden; }
    .scan-stage:before { content:'';position:absolute;left:0;right:0;height:2px;background:#22d3ee;box-shadow:0 0 15px #22d3ee;animation:scan 1.2s linear infinite; }
    .scan-label { color:#67e8f9;font-size:.73rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase; }.scan-text { color:#cbd5e1;margin-top:.45rem; }
    .match-pop { padding:.7rem 1rem;border-radius:12px;background:rgba(16,185,129,.12);border:1px solid rgba(52,211,153,.3);color:#a7f3d0;animation:pop .28s ease-out; }
    .db-stats { display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:.65rem;margin:.7rem 0 1rem; }
    .db-stat { padding:.65rem .8rem;border-radius:11px;background:rgba(17,26,50,.72);border:1px solid rgba(148,163,184,.14); }
    .db-stat-label { color:#93a4c7;font-size:.72rem;font-weight:600; }.db-stat-value { color:#f1f5f9;font-size:1.05rem;font-weight:700;margin-top:.08rem; }
    .kb-intro { display:flex;align-items:center;gap:1rem;padding:1.25rem 1.35rem;margin:.35rem 0 1rem;border-radius:17px;background:linear-gradient(120deg,rgba(8,145,178,.13),rgba(124,58,237,.12));border:1px solid rgba(103,232,249,.2); }
    .kb-intro-icon { display:grid;place-items:center;flex:0 0 3.2rem;width:3.2rem;height:3.2rem;border-radius:14px;background:rgba(34,211,238,.12);font-size:1.55rem; }
    .kb-intro strong { display:block;color:#f8fafc;font-size:1.05rem;margin-bottom:.2rem; }.kb-intro span { color:#aebdd7;font-size:.92rem; }
    .kb-files { display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:.7rem;margin:.65rem 0 1rem; }
    .kb-file { padding:.8rem .9rem;border-radius:12px;background:#111a31;border:1px solid rgba(148,163,184,.17);min-width:0; }
    .kb-file-name { color:#f8fafc;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }.kb-file-meta { color:#93a4c7;font-size:.78rem;margin-top:.2rem; }
    .st-key-knowledge_base_uploader [data-testid="stWidgetLabel"] p { color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;font-size:1.08rem !important;font-weight:800 !important;opacity:1 !important;margin-bottom:.35rem !important; }
    .st-key-knowledge_base_uploader [data-testid="stFileUploader"] > label { opacity:1 !important; }
    .empty-state { text-align:center;padding:3rem 1rem;border:1px dashed rgba(148,163,184,.22);border-radius:18px;background:rgba(15,23,42,.35); }
    .empty-icon { font-size:2.4rem;animation:float 3s ease-in-out infinite; }.empty-state h3 { color:#f1f5f9 !important; }.empty-state p { color:#b7c4dc !important; }
    @keyframes scan { from{top:0} to{top:100%} } @keyframes pulse { 50%{opacity:.3} }
    @keyframes float { 50%{transform:translateY(8px)} } @keyframes pop { from{transform:scale(.96);opacity:.3} }
    /* Low-glare midnight blue + muted bronze product palette */
    .stApp { background:radial-gradient(circle at 84% 0%,rgba(54,78,117,.18),transparent 35rem),radial-gradient(circle at 8% 38%,rgba(181,139,90,.08),transparent 30rem),#080f1d !important; }
    [data-testid="stSidebar"] { background:#0b1424 !important;border-right-color:rgba(181,139,90,.18) !important; }
    .rag-hero { background:linear-gradient(115deg,rgba(18,31,53,.98),rgba(9,17,31,.97)) !important;border-color:rgba(181,139,90,.24) !important; }
    .rag-hero:after { background:linear-gradient(135deg,#263b60,#9a764b) !important; }.rag-kicker,.rag-arrow { color:#d2ad7c !important; }
    .rag-gradient { background:linear-gradient(100deg,#dbe7f7,#c49a68) !important;-webkit-background-clip:text !important;color:transparent !important; }
    [data-testid="stTabs"] [role="tablist"] { background:rgba(8,15,29,.98) !important;border-color:#34445f !important; }
    [data-testid="stTabs"] [data-testid="stTab"] { background:linear-gradient(180deg,#1a2942,#111c30) !important;border-color:#34445f !important; }
    [data-testid="stTabs"] [data-testid="stTab"]:hover { background:linear-gradient(180deg,#243754,#17243b) !important;border-color:#9a764b !important; }
    [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] { background:linear-gradient(180deg,#344b70,#243754) !important;border-color:#b58b5a !important;box-shadow:0 -3px 14px rgba(181,139,90,.16),inset 0 3px 0 #c49a68 !important; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(24,39,65,.88),rgba(12,23,41,.9)) !important;border-color:rgba(181,139,90,.14) !important; }
    .stButton>button,[data-testid="stFormSubmitButton"] button { background:linear-gradient(110deg,#314a72,#8e6c45) !important;box-shadow:0 7px 20px rgba(7,14,27,.34) !important; }
    .chat-banner,.kb-intro,.strategy-card { background:linear-gradient(115deg,rgba(36,55,84,.3),rgba(112,82,49,.12)) !important;border-color:rgba(181,139,90,.18) !important; }
    .chat-spark { background:linear-gradient(135deg,#314a72,#9a764b) !important; }.strategy-card strong { color:#d2ad7c !important; }
    @media (max-width:700px) { [data-testid="stTabs"] [role="tablist"] { display:flex !important;overflow-x:auto !important;overflow-y:hidden !important;scrollbar-width:thin; } [data-testid="stTabs"] [data-testid="stTab"] { min-width:155px !important; } .rag-hero { padding:1.5rem; } }
    </style>
    <div class="rag-hero">
      <div class="rag-kicker">Interactive RAG laboratory</div>
      <div class="rag-title">See retrieval<br><span class="rag-gradient">think in real time.</span></div>
      <div class="rag-subtitle">Turn PDFs into a living knowledge system. Experiment with chunking, inspect vectors, and watch the best evidence rise to the top.</div>
      <div class="rag-flow"><span class="rag-node">PDF</span><span class="rag-arrow">→</span><span class="rag-node">Chunking</span><span class="rag-arrow">→</span><span class="rag-node">Embeddings</span><span class="rag-arrow">→</span><span class="rag-node">Retrieval</span><span class="rag-arrow">→</span><span class="rag-node">Top 5 Chunks</span><span class="rag-arrow">→</span><span class="rag-node">Grounding</span><span class="rag-arrow">→</span><span class="rag-node">Gemini</span><span class="rag-arrow">→</span><span class="rag-node">Answer</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_empty(icon: str, title: str, message: str) -> None:
    st.markdown(f'<div class="empty-state"><div class="empty-icon">{icon}</div><h3>{title}</h3><p>{message}</p></div>', unsafe_allow_html=True)


def render_processing_pipeline(uploaded_files, settings: dict, external_progress=None):
    """Run the real ingestion pipeline while exposing every processing stage."""
    documents, all_chunks, retrieval_chunks, embeddings = [], [], [], None
    file_count = len(uploaded_files)
    interaction_lock = st.empty()
    interaction_lock.markdown(
        """<style>
        [data-testid="stSidebar"] [data-testid="stForm"],
        [data-testid="stTabs"] [role="tablist"] {
            pointer-events: none !important;
            opacity: .62 !important;
            cursor: wait !important;
        }
        [data-testid="stTabs"] [role="tabpanel"] {
            pointer-events: none !important;
            opacity: .28 !important;
            filter: grayscale(.35) !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    with st.status("Building the knowledge base…", expanded=True) as pipeline:
        try:
            progress = st.progress(0.0, text="Starting ingestion pipeline…")
            if external_progress is not None:
                external_progress.progress(0.05, text="Reading PDFs with PyMuPDF…")

            parse_stage = st.empty()
            parse_stage.markdown(
                f"🔵 Parsing {file_count} PDF{'s' if file_count != 1 else ''} with PyMuPDF<span class='rag-arrow'>...</span>",
                unsafe_allow_html=True,
            )
            documents = parse_pdfs(uploaded_files)
            characters = sum(len(document.text) for document in documents)
            parse_stage.markdown(
                f"✅ Parsed successfully with PyMuPDF · {len(documents):,} pages · "
                f"{characters:,} characters extracted"
            )
            progress.progress(1 / 3, text="PDF parsing complete")
            if external_progress is not None:
                external_progress.progress(1 / 3, text="PDF parsing complete · Chunking in progress…")
            time.sleep(0.3)

            chunk_stage = st.empty()
            chunk_stage.markdown(
                f"🔵 Chunking with {settings['strategy']} · size {settings['chunk_size']:,} · "
                f"overlap {settings['overlap']:,}<span class='rag-arrow'>...</span>",
                unsafe_allow_html=True,
            )
            all_chunks = chunk_documents(
                documents, settings["strategy"], settings["chunk_size"], settings["overlap"]
            )
            retrieval_chunks = [chunk for chunk in all_chunks if chunk.level != "parent"]
            chunk_stage.markdown(
                f"✅ Chunking complete · {len(all_chunks):,} chunks created · "
                f"{len(retrieval_chunks):,} chunks ready for retrieval"
            )
            progress.progress(2 / 3, text="Chunking complete")
            if external_progress is not None:
                external_progress.progress(2 / 3, text="Chunking complete · Creating embeddings…")
            time.sleep(0.3)

            embedding_stage = st.empty()
            embedding_stage.markdown(
                f"🔵 Creating embeddings with {DEFAULT_MODEL}<span class='rag-arrow'>...</span>",
                unsafe_allow_html=True,
            )
            embeddings = embed_texts([chunk.text for chunk in retrieval_chunks])
            dimensions = embeddings.shape[1] if embeddings.ndim == 2 and len(embeddings) else 0
            embedding_stage.markdown(
                f"✅ Embeddings created · {len(embeddings):,} vectors · {dimensions:,} dimensions each"
            )
            progress.progress(1.0, text="All processing stages complete")
            if external_progress is not None:
                external_progress.progress(1.0, text="Knowledge base rebuilt successfully")
            pipeline.update(label="Knowledge base ready", state="complete", expanded=False)
        except Exception as error:
            pipeline.update(label="Knowledge base processing failed", state="error", expanded=True)
            st.error(f"Pipeline stopped: {error}")
    interaction_lock.empty()
    return documents, all_chunks, retrieval_chunks, embeddings


def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("### ⚡ Experiment controls")
        st.caption("Tune the pipeline and see every stage react.")
        st.divider()
        with st.form("pipeline_settings", clear_on_submit=False):
            engine = st.selectbox(
                "Search Engine", ["Brute Force", "ChromaDB"], index=1,
                key="search_engine_control",
            )
            strategy = st.selectbox("Chunk Strategy", [
                "Fixed size", "Fixed size with overlap", "Recursive", "Semantic",
                "Parent-child", "Fact & proposition",
            ], key="chunk_strategy_control")
            chunk_size = st.slider(
                "Chunk Size", 100, 2_000, 500, 50, key="chunk_size_control"
            )
            overlap = st.slider(
                "Overlap", 0, min(500, chunk_size - 1), min(100, chunk_size - 1), 10,
                help="Used by Fixed size with overlap and Parent-child strategies.",
                key="overlap_control",
            )
            top_k = st.slider("Top K", 1, 10, 5, 1, key="top_k_control")
            st.form_submit_button(
                "Apply Settings",
                type="primary",
                use_container_width=True,
                on_click=_request_knowledge_base_rebuild,
            )
        st.caption("Settings are applied together to prevent interrupting an active rebuild.")
        st.divider()
        st.caption(f"🧠 `{DEFAULT_MODEL}`")
        st.caption("🔒 Files stay in this session")
    return {
        "engine": engine,
        "strategy": strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "top_k": top_k,
    }


def render_knowledge_base(uploaded_files, settings: dict, rebuild_progress=None):
    st.subheader("Knowledge Base")
    st.markdown(
        '<div class="kb-intro"><div class="kb-intro-icon">📚</div><div>'
        '<strong>Build your searchable knowledge base</strong>'
        '<span>Upload one or more PDFs. RAG Explorer will parse, chunk, and embed them for semantic search.</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    with st.container(key="knowledge_base_uploader"):
        st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_files",
            help="You can select multiple PDF files at once.",
        )
    if not uploaded_files:
        return [], [], [], None

    artifacts = st.session_state.get("knowledge_base_artifacts")
    if artifacts is None:
        documents, all_chunks, chunks, embeddings = render_processing_pipeline(
            uploaded_files, settings, rebuild_progress
        )
        if chunks and embeddings is not None:
            st.session_state["knowledge_base_artifacts"] = (
                documents, all_chunks, chunks, embeddings
            )
    else:
        documents, all_chunks, chunks, embeddings = artifacts
    cols = st.columns(4)
    cols[0].metric("Documents", len(uploaded_files))
    cols[1].metric("Parsed pages", len(documents))
    cols[2].metric("Chunks", len(chunks))
    cols[3].metric("Vectors", len(embeddings) if embeddings is not None else 0)
    st.markdown("#### Uploaded documents")
    file_cards = []
    for file in uploaded_files:
        name = html.escape(file.name)
        size_mb = getattr(file, "size", 0) / (1024 * 1024)
        file_cards.append(
            f'<div class="kb-file"><div class="kb-file-name">📄 {name}</div>'
            f'<div class="kb-file-meta">{size_mb:.2f} MB · Ready</div></div>'
        )
    st.markdown(f'<div class="kb-files">{"".join(file_cards)}</div>', unsafe_allow_html=True)
    return documents, all_chunks, chunks, embeddings


def render_chunking(chunks: Sequence[Chunk], settings: dict) -> None:
    st.subheader("Chunking")
    descriptions = {
        "Fixed size": "Equal character windows with no shared text.",
        "Fixed size with overlap": "Sliding windows; highlighted edges show text carried between chunks.",
        "Recursive": "Preserves paragraphs and sentences before falling back to words or characters.",
        "Semantic": "Groups neighboring sentences and starts a new chunk at a meaning shift.",
        "Parent-child": "Small child chunks are retrieved while their larger parent supplies context.",
        "Fact & proposition": "Produces compact, independently retrievable statements.",
    }
    st.markdown(f'<div class="strategy-card"><strong>🧩 {settings["strategy"]}</strong><span>{descriptions[settings["strategy"]]}</span></div>', unsafe_allow_html=True)
    lengths = [len(chunk.text) for chunk in chunks]
    summary = st.columns(4)
    summary[0].metric("Total chunks", len(chunks))
    summary[1].metric(
        "Average chunk size",
        f"{sum(lengths) / len(lengths):,.0f} chars" if lengths else "0 chars",
    )
    summary[2].metric("Smallest chunk size", f"{min(lengths):,} chars" if lengths else "0 chars")
    summary[3].metric("Largest chunk size", f"{max(lengths):,} chars" if lengths else "0 chars")
    if settings["strategy"] == "Parent-child":
        for parent in [c for c in chunks if c.level == "parent"][:10]:
            with st.expander(
                f"Parent chunk {parent.global_id} · {parent.source} · page {parent.page} · characters {parent.start}–{parent.end}"
            ):
                st.write(parent.text)
                children = [c for c in chunks if c.parent_id == parent.id]
                for child in children:
                    st.markdown(
                        f"**↳ Chunk {child.global_id}** · characters {child.start}–{child.end}  \n{child.text}"
                    )
        return


def render_embeddings(chunks: Sequence[Chunk], embeddings: np.ndarray | None) -> None:
    st.subheader("Embeddings")
    if embeddings is None or not len(chunks):
        render_empty("🧠", "No vectors yet", "Upload documents to reveal the numerical fingerprints behind their meaning.")
        return
    cols = st.columns(3)
    cols[0].metric("Vectors", len(embeddings))
    cols[1].metric("Dimensions", embeddings.shape[1])
    cols[2].metric("Memory", f"{embeddings.nbytes / 1024:.1f} KB")


def _chunk_payload(chunk: Chunk) -> dict:
    return {
        "text": chunk.text,
        "book": chunk.source,
        "chunk_id": chunk.global_id,
        "start": chunk.start,
        "end": chunk.end,
        "page": chunk.page,
    }


def _canonical_results(results: Sequence[dict]) -> list[dict]:
    return [
        {
            "rank": result.get("rank", position),
            "similarity": float(result["final_score"]),
            "semantic_score": float(result["semantic_score"]),
            "answerability_score": float(result["answerability_score"]),
            "final_score": float(result["final_score"]),
            "boost_reasons": list(result.get("boost_reasons", [])),
            "originating_subquery": result.get("originating_subquery", ""),
            "originating_subqueries": list(result.get("originating_subqueries", [])),
            "coverage_score": int(result.get("coverage_score", 1)),
            "chunk": _chunk_payload(result["chunk"]),
        }
        for position, result in enumerate(results, start=1)
    ]


def _commit_search_state(
    query: str,
    smart_run: dict,
    settings: dict,
    chunks: Sequence[Chunk],
    embeddings: np.ndarray | None,
    sync_generation_input: bool = False,
    origin: str = "chat",
) -> None:
    """Atomically replace the canonical state after every completed search."""
    payload = _canonical_results(smart_run["results"])
    merged_payload = _canonical_results(smart_run["merged_candidates"])
    candidates_by_subquery = [
        {
            "subquery": group["subquery"],
            "intent_type": group["intent_type"],
            "transformed_retrieval_query": group["transformed_retrieval_query"],
            "candidates": _canonical_results(group["candidates"]),
        }
        for group in smart_run["candidates_by_subquery"]
    ]
    st.session_state["original_query"] = smart_run["original_query"]
    st.session_state["decomposed_subqueries"] = list(smart_run["decomposed_subqueries"])
    st.session_state["candidates_by_subquery"] = candidates_by_subquery
    st.session_state["merged_candidates"] = merged_payload
    st.session_state["final_retrieved_results"] = payload
    st.session_state["evidence_coverage_score"] = int(smart_run.get("coverage_score", 1))
    st.session_state["latest_query"] = query
    st.session_state["transformed_query"] = smart_run["transformed_retrieval_query"]
    st.session_state["detected_intent"] = smart_run["intent_type"]
    st.session_state["extracted_subject"] = smart_run["extracted_entity_or_subject"]
    st.session_state["retrieval_candidates"] = merged_payload
    st.session_state["retrieved_results"] = payload
    st.session_state["query_embedding"] = smart_run["query_embedding"]
    st.session_state["search_engine"] = settings["engine"]
    st.session_state["chunking_strategy"] = settings["strategy"]
    st.session_state["chunk_size"] = settings["chunk_size"]
    st.session_state["overlap"] = settings["overlap"]
    st.session_state["retrieval_timestamp"] = time.time()
    st.session_state["retrieval_origin"] = origin
    st.session_state["retrieval_latency_ms"] = smart_run["retrieval_latency"] * 1_000
    st.session_state["embedding_latency"] = smart_run["embedding_latency"]
    st.session_state["retrieval_latency"] = smart_run["retrieval_latency"]
    # Object results remain available only to non-generation visualizations.
    st.session_state["results"] = [
        SearchResult(item["chunk"], item["semantic_score"], item["rank"])
        for item in smart_run["results"]
    ]
    st.session_state["query_vector"] = smart_run["query_embedding"]
    st.session_state["last_query"] = query
    st.session_state["last_engine"] = settings["engine"]
    st.session_state.pop("generation_result", None)
    st.session_state.pop("generation_metrics", None)
    st.session_state.pop("answer_mode", None)
    st.session_state.pop("extractive_latency_ms", None)
    st.session_state.pop("extractive_sentences_used", None)
    st.session_state.pop("extractive_sources_used", None)
    st.session_state.pop("subquery_coverage", None)
    st.session_state.pop("prompt_viewer", None)
    if sync_generation_input:
        st.session_state["generation_query_input"] = query
    for item in payload:
        chunk = item["chunk"]
        preview = " ".join(chunk["text"][:200].splitlines())
        logger.info(
            "Smart result rank=%d chunk=%d source=%s semantic=%.4f answerability=%.4f final=%.4f preview=%r",
            item["rank"], chunk["chunk_id"], chunk["book"], item["semantic_score"],
            item["answerability_score"], item["final_score"], preview,
        )
    if payload:
        previews = {
            chunk.global_id: [float(value) for value in embeddings[index][:10]]
            for index, chunk in enumerate(chunks)
        } if embeddings is not None else {}
        record = create_experiment_record(
            query=query,
            smart_run=smart_run,
            settings=settings,
            embedding_model=DEFAULT_MODEL,
            embedding_dimension=(embeddings.shape[1] if embeddings is not None and embeddings.ndim == 2 else 0),
            knowledge_base_version=st.session_state.get("knowledge_base_version", "unknown"),
            retrieved_results=payload,
            vector_count=len(chunks),
            embedding_previews=previews,
        )
        append_experiment(record)


def render_retrieval(chunks: Sequence[Chunk], embeddings: np.ndarray | None, settings: dict) -> None:
    st.subheader("Test Retrieval")
    st.caption("Search the vector index and inspect the final smart-ranked chunks. No LLM is used in this tab.")
    with st.form("retrieval_search_form", clear_on_submit=False):
        query = st.text_input("Ask a question", key="retrieval_query")
        with st.container(key="retrieval_cta"):
            cta_col, _ = st.columns([1.7, 4.3])
            with cta_col:
                search_submitted = st.form_submit_button(
                    "🔎 Search Knowledge Base",
                    type="primary",
                    disabled=not chunks,
                    use_container_width=True,
                )
    if search_submitted and query.strip():
        try:
            if settings["engine"] == "Brute Force":
                store = BruteForceStore(chunks, embeddings)
            else:
                store = ChromaStore(chunks, embeddings)
            with st.spinner("Searching and smart-ranking the strongest evidence…"):
                smart_run = smart_retrieve(query, store, embed_query, final_k=settings["top_k"])
        except ImportError:
            st.error("ChromaDB is not installed. Run `pip install -r requirements.txt`.")
            return
        except Exception as error:
            st.error(f"Smart Retrieval failed: {error}")
            return
        _commit_search_state(
            query, smart_run, settings, chunks, embeddings,
            sync_generation_input=True, origin="test_retrieval"
        )
        if not smart_run["candidates"]:
            st.error("Smart Retrieval found no candidate chunks for this question.")

    if (
        st.session_state.get("retrieval_timestamp")
        and st.session_state.get("retrieval_origin") == "test_retrieval"
    ):
        final_results = st.session_state.get("retrieved_results", [])
        if not final_results:
            st.warning("No semantic candidates were found. Try a different question or knowledge base.")
            return

        st.markdown("#### Final smart-ranked results")
        retrieval_time_ms = float(st.session_state.get("retrieval_latency_ms", 0.0))
        result_stats = st.columns([1, 1, 3])
        result_stats[0].metric("Retrieval time", f"{retrieval_time_ms:,.1f} ms")
        result_stats[1].metric("Results returned", len(final_results))
        for position, result in enumerate(final_results, start=1):
            chunk = result.get("chunk", {})
            rank = result.get("rank", position)
            reasons = result.get("boost_reasons", []) or ["semantic relevance only"]
            with st.expander(
                f"#{rank} · {chunk.get('book', 'Unknown source')} · Chunk {chunk.get('chunk_id', '—')}",
                expanded=False,
            ):
                score_cols = st.columns(3)
                score_cols[0].metric("Semantic", f"{result.get('semantic_score', 0.0):.4f}")
                score_cols[1].metric("Answerability", f"{result.get('answerability_score', 0.0):.4f}")
                score_cols[2].metric("Final score", f"{result.get('final_score', 0.0):.4f}")
                st.write("**Boost reasons:** " + " · ".join(reasons))
                st.write(chunk.get("text", "No chunk text available."))
                st.caption(
                    f"Page {chunk.get('page', '—')} · Characters "
                    f"{chunk.get('start', '—')}–{chunk.get('end', '—')}"
                )
    elif not chunks:
        render_empty("🔎", "Ready to search", "Add PDFs in Knowledge Base, then ask a question and watch every comparison happen.")


def _project_vectors(embeddings: np.ndarray, query_vector: np.ndarray | None = None):
    """Project vectors to 2D with a sample-fitted PCA, without another dependency."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    sample = matrix if len(matrix) <= 2_000 else matrix[np.linspace(0, len(matrix) - 1, 2_000, dtype=int)]
    center = sample.mean(axis=0)
    _, _, axes = np.linalg.svd(sample - center, full_matrices=False)
    components = axes[:2].T
    if components.shape[1] < 2:
        components = np.pad(components, ((0, 0), (0, 2 - components.shape[1])))
    points = (matrix - center) @ components
    query_point = None if query_vector is None else (np.asarray(query_vector) - center) @ components
    return points, query_point


def render_vector_db(chunks: Sequence[Chunk], embeddings: np.ndarray | None) -> None:
    st.subheader("Vector database")
    results: list[SearchResult] = sorted(
        st.session_state.get("results", []), key=lambda result: result.score, reverse=True
    )
    query_vector = st.session_state.get("query_vector")
    dimensions = embeddings.shape[1] if embeddings is not None and embeddings.ndim == 2 and len(embeddings) else 0
    st.markdown(
        f'<div class="db-stats">'
        f'<div class="db-stat"><div class="db-stat-label">Documents</div><div class="db-stat-value">{len({chunk.source for chunk in chunks}):,}</div></div>'
        f'<div class="db-stat"><div class="db-stat-label">Vectors</div><div class="db-stat-value">{len(chunks):,}</div></div>'
        f'<div class="db-stat"><div class="db-stat-label">Dimensions</div><div class="db-stat-value">{f"{dimensions:,}" if dimensions else "—"}</div></div>'
        f'<div class="db-stat"><div class="db-stat-label">Distance metric</div><div class="db-stat-value">Cosine</div></div>'
        f'</div>', unsafe_allow_html=True,
    )
    st.caption("Chunks are ranked by cosine similarity to the query vector.")
    active_query = st.session_state.get("latest_query") or st.session_state.get("last_query")
    if query_vector is not None and active_query:
        with st.container(border=True):
            st.markdown("**Visualization context**")
            st.markdown(f"**Query:** {active_query}")
            st.markdown("⚪ Stored knowledge-base chunks &nbsp;&nbsp; 🟢 Retrieved Top-K chunks &nbsp;&nbsp; 🔴 Current query vector")
            if len(st.session_state.get("decomposed_subqueries", [])) > 1:
                st.caption("For a multi-part question, the red point is the normalized mean of its sub-query vectors.")
    else:
        st.info(
            "The map currently shows stored knowledge-base chunks only. Run Test Retrieval or Chat "
            "to add a query vector and highlight its Top-K neighbours."
        )

    if embeddings is None or not len(chunks):
        render_empty("🗺️", "The vector map is empty", "Upload PDFs to populate the index and reveal the semantic landscape.")
        return

    with st.spinner("Projecting high-dimensional vectors into 2D…"):
        points, query_point = _project_vectors(embeddings, query_vector)
    retrieved_ids = {result.chunk.id for result in results}
    map_rows = [
        {
            "x": float(point[0]), "y": float(point[1]),
            "type": "Retrieved top-K" if chunk.id in retrieved_ids else "Stored chunk",
            "chunk": chunk.global_id, "chunk_key": str(chunk.global_id), "source": chunk.source, "page": chunk.page,
            "label": f"Chunk {chunk.global_id}",
            "start": chunk.start, "end": chunk.end,
            "preview": chunk.text[:100],
        }
        for point, chunk in zip(points, chunks, strict=True)
    ]
    if query_point is not None:
        map_rows.append({
            "x": float(query_point[0]), "y": float(query_point[1]), "type": "Current query",
            "chunk": "QUERY", "chunk_key": "QUERY", "label": "Query", "source": "Current search", "page": 0, "start": 0, "end": 0,
            "preview": st.session_state.get("last_query", ""),
        })

    st.markdown("#### Two-dimensional vector map")
    st.caption("Hover over any point to inspect its chunk, source, page, character range, and text preview.")
    chart_col, neighbor_col = st.columns([2.2, 1], gap="large")
    with chart_col:
        st.vega_lite_chart(
            map_rows,
            {
                "height": 520,
                "encoding": {
                    "x": {"field": "x", "type": "quantitative", "axis": {"title": "Projection 1", "gridColor": "#25314d"}},
                    "y": {"field": "y", "type": "quantitative", "axis": {"title": "Projection 2", "gridColor": "#25314d"}},
                    "color": {
                        "field": "type", "type": "nominal",
                        "scale": {"domain": ["Stored chunk", "Retrieved top-K", "Current query"], "range": ["#64748b", "#22c55e", "#ef4444"]},
                        "legend": {"orient": "top", "title": None},
                    },
                    "tooltip": [
                        {"field": "type", "title": "Vector"}, {"field": "chunk", "title": "Chunk"},
                        {"field": "source", "title": "Source"}, {"field": "page", "title": "Page"},
                        {"field": "start", "title": "Start character"}, {"field": "end", "title": "End character"},
                        {"field": "preview", "title": "Text"},
                    ],
                },
                "layer": [
                    {
                        "transform": [{"filter": "datum.type === 'Stored chunk'"}],
                        "mark": {"type": "circle", "filled": True, "size": 28, "opacity": 0.62},
                    },
                    {
                        "transform": [{"filter": "datum.type === 'Retrieved top-K'"}],
                        "mark": {"type": "circle", "filled": True, "size": 190, "opacity": 1, "stroke": "#dcfce7", "strokeWidth": 2.5},
                    },
                    {
                        "transform": [{"filter": "datum.type === 'Retrieved top-K'"}],
                        "mark": {"type": "text", "dy": -14, "fontSize": 12, "fontWeight": "bold", "color": "#bbf7d0"},
                        "encoding": {"text": {"field": "label", "type": "nominal"}},
                    },
                    {
                        "transform": [{"filter": "datum.type === 'Current query'"}],
                        "mark": {"type": "circle", "filled": True, "size": 280, "opacity": 1, "stroke": "#fee2e2", "strokeWidth": 3},
                    },
                ],
                "config": {"background": "#0b1424", "view": {"stroke": "#34445f"}, "axis": {"labelColor": "#c8d3e3", "titleColor": "#edf2f8"}, "legend": {"labelColor": "#edf2f8"}},
            },
            use_container_width=True,
        )
    with neighbor_col:
        st.markdown("#### Retrieved neighbours")
        st.caption("Sorted by cosine similarity, highest to lowest.")
        if not results:
            st.info("Run a search in Retrieval to highlight the nearest vectors.")
        for position, result in enumerate(results, start=1):
            st.markdown(
                f"**#{position} · Chunk {result.chunk.global_id}**  \n"
                f"Similarity `{result.score:.4f}`  \n"
                f"{result.chunk.source} · page {result.chunk.page}  \n"
                f"Characters `{result.chunk.start}–{result.chunk.end}`"
            )
            st.progress(max(0.0, min(1.0, result.score)))

    if results:
        st.markdown("#### Query-to-result distance")
        st.caption("Ordered by cosine similarity, highest to lowest.")
        distance_cols = st.columns(min(5, len(results)))
        for index, result in enumerate(results):
            with distance_cols[index % len(distance_cols)]:
                st.markdown(
                    f"<div class='distance-card'>"
                    f"<strong>Query vector</strong><span class='distance-arrow'>↓</span>"
                    f"<span>cosine similarity: <b style='color:#86efac'>{result.score:.4f}</b></span>"
                    f"<span class='distance-arrow'>↓</span><strong>Chunk {result.chunk.global_id}</strong>"
                    f"<span>characters {result.chunk.start}–{result.chunk.end}</span>"
                    f"</div>", unsafe_allow_html=True,
                )


def render_generation(chunks: Sequence[Chunk], embeddings: np.ndarray | None, settings: dict) -> None:
    st.subheader("Chat with your knowledge base")
    st.markdown(
        '<div class="chat-banner"><div class="chat-spark">✨</div><div>'
        '<strong>Ask your documents anything</strong>'
        '<span>One click retrieves the strongest evidence, then uses Gemini or a local extractive fallback.</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    latest_query = st.session_state.get("latest_query", "")
    retrieved_results: list[dict] = st.session_state.get("retrieved_results", [])
    model_name = get_model_name()

    if not get_gemini_api_key():
        st.warning(
            "Gemini is not configured. Auto mode will use the local extractive-answer fallback."
        )
    with st.form("chat_generation_form", clear_on_submit=False):
        generation_mode = st.selectbox(
            "Generation Mode",
            ["Auto", "Gemini", "Extractive only"],
            key="generation_mode",
            help="Auto tries Gemini first and uses extractive evidence only for recognized backend failures.",
        )
        with st.container(key="chat_question"):
            query = st.text_input(
                "Ask a question",
                value=latest_query,
                placeholder="What would you like to know from these documents?",
                key="generation_query_input",
            )
        with st.container(key="chat_cta"):
            cta_col, _ = st.columns([1.7, 4.3])
            with cta_col:
                generation_submitted = st.form_submit_button(
                    "✨ Retrieve & Answer",
                    type="primary",
                    disabled=not chunks or embeddings is None,
                    use_container_width=True,
                )

    if generation_submitted and query.strip():
        try:
            with st.spinner("Decomposing the question and running Smart Retrieval…"):
                if settings["engine"] == "Brute Force":
                    store = BruteForceStore(chunks, embeddings)
                else:
                    store = ChromaStore(chunks, embeddings)
                smart_run = smart_retrieve(query, store, embed_query, final_k=settings["top_k"])
                _commit_search_state(query, smart_run, settings, chunks, embeddings, origin="chat")
                st.session_state.pop("generation_result", None)
                latest_query = st.session_state["latest_query"]
                retrieved_results = st.session_state["retrieved_results"]
                subquestions = st.session_state.get("decomposed_subqueries", [latest_query])
                if not retrieved_results:
                    raise ValueError("Retrieval returned no results. Adjust the query or knowledge base and retry.")
                viewer_context, _ = format_retrieved_context(retrieved_results)
                if not viewer_context.strip():
                    raise ValueError("Retrieval produced empty context. Generation has been stopped.")
                if len(viewer_context) < 100:
                    raise ValueError(
                        f"Retrieved context is only {len(viewer_context)} characters; at least 100 are required."
                    )
                user_prompt = build_grounded_prompt(latest_query, retrieved_results, subquestions)
                st.session_state["prompt_viewer"] = {
                    "query": latest_query,
                    "context": viewer_context,
                    "prompt": f"SYSTEM INSTRUCTION\n\n{SYSTEM_PROMPT}\n\nUSER CONTENT\n\n{user_prompt}",
                }
            with st.spinner("Preparing a grounded answer…"):
                try:
                    active_subquestions = st.session_state.get(
                        "decomposed_subqueries", [st.session_state["latest_query"]]
                    )
                    if generation_mode == "Extractive only":
                        generation = generate_extractive_answer(
                            st.session_state["latest_query"],
                            st.session_state["retrieved_results"],
                            subquestions=active_subquestions,
                        )
                    else:
                        try:
                            generation = generate_answer(
                                st.session_state["latest_query"],
                                st.session_state["retrieved_results"],
                                model_name,
                                active_subquestions,
                            )
                            generation["mode"] = "Gemini"
                        except GenerationBackendError as error:
                            if generation_mode != "Auto" or not is_fallback_eligible_generation_error(error):
                                raise
                            generation = generate_extractive_answer(
                                st.session_state["latest_query"],
                                st.session_state["retrieved_results"],
                                subquestions=active_subquestions,
                            )
                            generation["fallback_reason"] = str(error)
                    st.session_state["generation_result"] = generation
                    latest_experiment_id = st.session_state.get("latest_experiment_id")
                    if latest_experiment_id:
                        update_experiment_generation(latest_experiment_id, generation)
                    st.session_state["answer_mode"] = generation.get("mode", "Gemini")
                    st.session_state["extractive_latency_ms"] = generation.get("extractive_latency_ms")
                    st.session_state["extractive_sentences_used"] = generation.get("sentences_used", [])
                    st.session_state["extractive_sources_used"] = generation.get("sources_used", [])
                    if generation.get("mode") == "Gemini":
                        st.session_state["subquery_coverage"] = [
                            {
                                "sub_question": part["sub_question"],
                                "coverage_status": part["coverage_status"],
                                "supporting_chunks": part["supporting_chunks"],
                                "evidence_summary": part["evidence_summary"],
                                "answer": part["answer"],
                            }
                            for part in generation.get("subquestion_answers", [])
                        ]
                    if generation.get("prompt"):
                        st.session_state["prompt_viewer"]["prompt"] = (
                            f"SYSTEM INSTRUCTION\n\n{SYSTEM_PROMPT}\n\nUSER CONTENT\n\n"
                            f"{generation['prompt']}"
                        )
                    st.session_state["generation_metrics"] = {
                        key: generation.get(key)
                        for key in (
                            "mode", "model", "latency_ms", "extractive_latency_ms",
                            "prompt_tokens", "output_tokens", "sentences_used", "sources_used",
                        )
                    }
                except (GenerationBackendError, ValueError) as error:
                    st.error(str(error))
        except ImportError:
            st.error("ChromaDB is not installed. Run `pip install -r requirements.txt`.")
        except Exception as error:
            st.error(f"Automatic retrieval failed: {error}")

    latest_query = st.session_state.get("latest_query", "")
    retrieved_results = st.session_state.get("retrieved_results", [])
    result = st.session_state.get("generation_result")
    with st.container(key="chat_summary"):
        if result:
            if result.get("mode") == "Extractive":
                evidence_ids = {
                    (sentence["source"], sentence["chunk_id"])
                    for sentence in result.get("sentences_used", [])
                }
            else:
                evidence_ids = {
                    (chunk["source"], chunk["chunk_id"])
                    for part in result.get("subquestion_answers", [])
                    for chunk in part.get("supporting_chunks", [])
                }
            evidence_count = len(evidence_ids) if result.get("subquestion_answers") else len(retrieved_results)
            summary = st.columns(4)
            summary[0].metric("Retrieved chunks", len(retrieved_results))
            summary[1].metric("Evidence chunks used", evidence_count)
            summary[2].metric("Answer mode", result.get("mode", "Gemini"))
            summary[3].metric("Search engine", st.session_state.get("search_engine", settings["engine"]))
        else:
            summary = st.columns(2)
            summary[0].metric("Model", model_name)
            summary[1].metric("Search engine", settings["engine"])
    if result:
        with st.container(key="chat_metrics"):
            stats = st.columns(5)
            stats[0].metric("Embedding", f"{st.session_state.get('embedding_latency', 0) * 1_000:,.1f} ms")
            stats[1].metric("Retrieval", f"{st.session_state.get('retrieval_latency', 0) * 1_000:,.1f} ms")
            stats[2].metric("Answer", f"{result['latency_ms']:,.1f} ms")
            stats[3].metric("Input prompt tokens", result["prompt_tokens"] if result["prompt_tokens"] is not None else "—")
            stats[4].metric("Output tokens", result["output_tokens"] if result["output_tokens"] is not None else "—")

    if result:
        st.markdown("#### Grounded answer")
        if result.get("mode") == "Extractive":
            if result.get("fallback_reason"):
                st.warning(
                    "Gemini generation is unavailable, so the app is showing the strongest "
                    "supporting sentences rather than a generated synthesis."
                )
            else:
                st.warning(
                    "Extractive only mode is showing the strongest supporting sentences "
                    "rather than a generated synthesis."
                )
            st.markdown("**Extractive answer — selected directly from retrieved text**")
        if result.get("subquestion_answers"):
            for index, part in enumerate(result["subquestion_answers"], start=1):
                st.markdown(f"**{index}. {part['sub_question']}**")
                if part["coverage_status"] == "Supported":
                    st.success("Evidence coverage: Supported")
                elif part["coverage_status"] == "Partially supported":
                    st.warning("Evidence coverage: Partially supported")
                else:
                    st.error("Evidence coverage: Unsupported")
                st.markdown(part["answer"])
        else:
            st.success(result["answer"])

    if result and result.get("mode") == "Extractive":
        st.markdown("#### Selected supporting sentences")
        sentences_used = result.get("sentences_used", [])
        if not sentences_used:
            st.info("No sentence met the minimum relevance threshold.")
        for index, sentence in enumerate(sentences_used, start=1):
            st.markdown(
                f"**{index}. {sentence['source']} · Chunk {sentence['chunk_id']}**  \n"
                f"{sentence['text']}"
            )
            st.progress(
                max(0.0, min(1.0, sentence["similarity"])),
                text=f"Sentence similarity: {sentence['similarity']:.4f}",
            )

    if result and result.get("mode") != "Extractive" and result.get("subquestion_answers"):
        st.markdown("#### Evidence inspection")
        for index, part in enumerate(result["subquestion_answers"], start=1):
            supporting = part.get("supporting_chunks", [])
            with st.expander(
                f"{index}. {part['sub_question']} · {part['coverage_status']}", expanded=False
            ):
                st.write(f"**Coverage status:** {part['coverage_status']}")
                st.write(f"**Supporting chunks:** {len(supporting)}")
                st.write(f"**Evidence summary:** {part['evidence_summary']}")
                if not supporting:
                    st.info("No retrieved chunk contained direct or partial answering evidence.")
                for chunk in supporting:
                    st.markdown(
                        f"**{chunk['source']} · Chunk {chunk['chunk_id']}**  \n"
                        f"{chunk['excerpt']}"
                    )
                    st.caption(f"Why this counts as evidence: {chunk['reason']}")

    if retrieved_results:
        with st.expander(f"📚 Retrieved chunks ({len(retrieved_results)})", expanded=False):
            for position, item in enumerate(retrieved_results, start=1):
                chunk = item["chunk"]
                rank = item.get("rank", position)
                st.markdown(
                    f"**#{rank} · Chunk {chunk['chunk_id']} · {chunk['book']} · "
                    f"page {chunk.get('page', '—')}**"
                )
                st.progress(
                    max(0.0, min(1.0, item["similarity"])),
                    text=f"Similarity: {item['similarity']:.4f}",
                )
                st.write(chunk["text"])
                st.caption(
                    f"Characters {chunk['start']}–{chunk['end']} · {len(chunk['text'])} characters"
                )
                if position < len(retrieved_results):
                    st.divider()

    viewer = st.session_state.get("prompt_viewer")
    if viewer and viewer.get("query") == latest_query:
        st.markdown("#### Prompt Viewer")
        with st.expander("1. Retrieved Context – exact chunks selected by retrieval"):
            st.code(viewer["context"], language=None)
        if result and result.get("mode") == "Extractive":
            with st.expander("2. Extractive selection details"):
                st.info("No prompt was sent to Gemini. Sentences were ranked locally and copied verbatim.")
        else:
            with st.expander("2. Prompt Sent to Gemini – full constructed prompt"):
                st.code(viewer["prompt"], language=None)
            with st.expander("3. Gemini Response – answer with citations", expanded=False):
                if result:
                    st.markdown(result["answer"])
                else:
                    st.info("Gemini did not return a response for this request.")
    elif not chunks:
        render_empty("💬", "Upload documents first", "Build the knowledge base, then ask a question directly here.")
    else:
        render_empty("✨", "Ask anything in your knowledge base", "Retrieval and generation will run together when you click the button.")


def _experiment_table_rows(records: Sequence[dict]) -> list[dict]:
    return [
        {
            "Query": record["question"][:90] + ("…" if len(record["question"]) > 90 else ""),
            "Chunk strategy": record["chunk_strategy"],
            "Chunk size": record["chunk_size"],
            "Search engine": record["search_engine"],
            "Top K": record["top_k"],
            "Retrieval time (ms)": round(record["retrieval_latency_ms"], 2),
            "Answer mode": record["answer_mode"],
            "Answer quality": record["answer_quality"] or "Not rated",
        }
        for record in records
    ]


def _experiment_label(record: dict) -> str:
    return f"{record['question'][:70]} · {record['chunk_strategy']} · {record['search_engine']}"


def render_experiments() -> None:
    """Render session-scoped RAG experiment history and quality comparisons."""
    st.subheader("🧪 Experiments")
    history = initialize_experiment_history()
    st.caption("Compare retrieval configurations and add a human quality rating to each run.")
    if not history:
        render_empty("🧪", "No experiments yet", "Run Test Retrieval or Retrieve & Answer to record the first experiment.")
        return

    st.dataframe(
        _experiment_table_rows(history),
        use_container_width=True,
        hide_index=True,
        row_height=30,
        height=min(36 + 30 * len(history), 310),
    )

    review_card = st.container(border=True, key="experiment_review_card")
    with review_card.container(key="experiment_selector"):
        selected_id = st.selectbox(
            "Select an experiment",
            [record["experiment_id"] for record in history],
            format_func=lambda value: _experiment_label(next(record for record in history if record["experiment_id"] == value)),
            key="selected_experiment_id",
        )
    selected = next(record for record in history if record["experiment_id"] == selected_id)

    review_card.markdown("#### Human quality rating")
    answer_options = ["Not rated", "Correct", "Partially correct", "Incorrect"]
    evidence_options = ["Not rated", "Strong", "Partial", "Weak"]
    correct_options = ["Not sure", "Yes", "No"]
    with review_card.form("experiment_rating_form"):
        rating_cols = st.columns([1, 1, 1, .55])
        answer_quality = rating_cols[0].selectbox(
            "Answer quality", answer_options,
            index=answer_options.index(selected["answer_quality"] or "Not rated"),
        )
        evidence_quality = rating_cols[1].selectbox(
            "Evidence quality", evidence_options,
            index=evidence_options.index(selected["evidence_quality"] or "Not rated"),
        )
        current_correct = "Yes" if selected["correct_chunk_found"] is True else "No" if selected["correct_chunk_found"] is False else "Not sure"
        correct_choice = rating_cols[2].selectbox(
            "Correct chunk found", correct_options, index=correct_options.index(current_correct)
        )
        with rating_cols[3]:
            st.markdown("<div style='height:1.35rem'></div>", unsafe_allow_html=True)
            save_rating = st.form_submit_button("Save", type="primary", use_container_width=True)
        if save_rating:
            update_experiment_rating(
                selected_id, answer_quality, evidence_quality,
                True if correct_choice == "Yes" else False if correct_choice == "No" else None,
                selected.get("notes", ""),
            )
            st.rerun()

    with review_card.expander("Selected experiment details", expanded=False):
        st.markdown(f"**{selected['question']}**")
        st.caption(
            f"Original query: {selected['original_query']}  \n"
            f"Transformed query: {selected['transformed_query']}  \n"
            f"Detected intent: {selected['detected_intent']}"
        )
        with st.container(key="experiment_detail_cards"):
            configuration_cards = st.columns(4)
            configuration_cards[0].metric("Chunk strategy", selected["chunk_strategy"])
            configuration_cards[1].metric("Chunk size", selected["chunk_size"])
            configuration_cards[2].metric("Search engine", selected["search_engine"])
            configuration_cards[3].metric("Top K", selected["top_k"])
            result_cards = st.columns(4)
            result_cards[0].metric("Answer mode", selected["answer_mode"])
            result_cards[1].metric("Retrieval", f"{selected['retrieval_latency_ms']:,.1f} ms")
            result_cards[2].metric("Total", f"{selected['total_latency_ms']:,.1f} ms")
            result_cards[3].metric("Answer quality", selected["answer_quality"] or "Not rated")
        if selected["generated_answer"]:
            st.markdown("**Answer**")
            st.write(selected["generated_answer"])
        st.markdown("**Retrieved evidence**")
        for position, result in enumerate(selected["retrieved_results"], start=1):
            chunk = result["chunk"]
            with st.expander(f"#{position} · {chunk.get('book', 'Unknown')} · Chunk {chunk.get('chunk_id', '—')}"):
                st.write(chunk.get("text", ""))
                st.caption(
                    f"Semantic {result.get('semantic_score', 0):.4f} · "
                    f"Answerability {result.get('answerability_score', 0):.4f} · "
                    f"Final {result.get('final_score', 0):.4f}"
                )
