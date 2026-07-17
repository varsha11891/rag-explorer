"""RAG Explorer Streamlit entry point."""
import streamlit as st
from src import components as ui
from src.embedder import DEFAULT_MODEL
from src.experiment_tracker import build_knowledge_base_version, initialize_experiment_history
st.set_page_config(page_title="RAG Explorer", page_icon="🔎", layout="wide")
ui.render_app_shell()

settings = ui.render_sidebar()
uploaded_files = st.session_state.get("pdf_files", [])
initialize_experiment_history()
st.session_state["knowledge_base_version"] = build_knowledge_base_version(
    uploaded_files, settings, DEFAULT_MODEL
)

configuration_signature = (
    tuple((file.name, getattr(file, "size", 0)) for file in uploaded_files),
    settings["strategy"],
    settings["chunk_size"],
    settings["overlap"],
)
configuration_changed = st.session_state.get("knowledge_base_configuration") != configuration_signature
if configuration_changed:
    for key in (
        "latest_query", "original_query", "decomposed_subqueries", "candidates_by_subquery",
        "merged_candidates", "final_retrieved_results", "evidence_coverage_score",
        "transformed_query", "detected_intent",
        "extracted_subject", "retrieval_candidates", "retrieved_results", "retrieval_latency_ms",
        "query_embedding", "search_engine",
        "chunking_strategy", "chunk_size", "overlap", "retrieval_timestamp", "retrieval_origin",
        "results", "query_vector", "generation_result", "generation_metrics",
        "prompt_viewer", "generation_query_input", "subquery_coverage",
        "answer_mode", "extractive_latency_ms", "extractive_sentences_used",
        "extractive_sources_used",
        "knowledge_base_artifacts",
    ):
        st.session_state.pop(key, None)
    st.session_state["knowledge_base_configuration"] = configuration_signature
    if not st.session_state.get("knowledge_base_rebuild_requested"):
        st.session_state["main_navigation_tab"] = "Knowledge Base"

rebuild_progress = None
if configuration_changed and uploaded_files:
    rebuild_progress = st.progress(
        0.0,
        text=f"Chunking in progress… Applying {settings['strategy']} strategy",
    )

artifacts = st.session_state.get("knowledge_base_artifacts")
rebuild_requested = st.session_state.get("knowledge_base_rebuild_requested", False)
rebuilding_existing = bool(
    rebuild_requested and st.session_state.get("knowledge_base_rebuild_existing", False)
)
if st.session_state.get("main_navigation_tab") == "Analytics":
    st.session_state["main_navigation_tab"] = "🧪 Experiments"
knowledge_base_ready = bool(
    not rebuild_requested and artifacts is not None and artifacts[2] and artifacts[3] is not None
)
show_full_navigation = knowledge_base_ready or rebuilding_existing
tab_labels = (
    ["Knowledge Base", "Chunking", "Embeddings", "Test Retrieval", "Chat", "🧪 Experiments", "Vector DB"]
    if show_full_navigation
    else ["Knowledge Base"]
)
tabs = st.tabs(
    tab_labels,
    key="main_navigation_tab",
    on_change="rerun",
)
with tabs[0]:
    documents, all_chunks, chunks, embeddings = ui.render_knowledge_base(
        uploaded_files, settings, rebuild_progress
    )
completed_artifacts = st.session_state.get("knowledge_base_artifacts")
rebuild_completed = bool(
    completed_artifacts is not None and completed_artifacts[2] and completed_artifacts[3] is not None
)
if rebuild_completed and rebuild_requested:
    st.session_state["knowledge_base_rebuild_requested"] = False
    st.session_state["knowledge_base_rebuild_existing"] = False
    st.session_state.pop("rebuild_return_tab", None)

if not show_full_navigation:
    if rebuild_completed:
        st.rerun()
else:
    with tabs[1]:
        ui.render_chunking(all_chunks, settings)
    with tabs[2]:
        ui.render_embeddings(chunks, embeddings)
    with tabs[3]:
        ui.render_retrieval(chunks, embeddings, settings)
    with tabs[4]:
        ui.render_generation(chunks, embeddings, settings)
    with tabs[5]:
        ui.render_experiments()
    with tabs[6]:
        ui.render_vector_db(chunks, embeddings)
