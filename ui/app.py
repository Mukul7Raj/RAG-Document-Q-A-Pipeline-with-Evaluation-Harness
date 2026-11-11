"""Interactive Streamlit Dashboard for RAG Document Q&A and Evaluation Harness."""
import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from config.settings import settings
from src.generation.prompts import PromptStrategy
from src.pipeline.rag_pipeline import RAGPipeline
from evaluation.harness import EvaluationHarness, AblationStudyRunner
from evaluation.benchmark_data import get_default_benchmark_dataset

st.set_page_config(
    page_title="RAG Document Q&A & Evaluation Harness",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #4f46e5;
        margin-bottom: 12px;
    }
    .citation-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 6px;
        padding: 12px;
        margin-top: 8px;
    }
    .guardrail-pass {
        color: #15803d;
        font-weight: 600;
    }
    .guardrail-fail {
        color: #b91c1c;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_pipeline():
    """Initializes and caches the RAG pipeline instance."""
    return RAGPipeline(auto_load_index=True)


pipeline = load_rag_pipeline()

# Sidebar: System Status & Configuration
with st.sidebar:
    st.title("⚙️ RAG Engine")
    stats = pipeline.vector_store.get_stats()
    st.metric("Total Indexed Chunks", stats.get("total_vectors", 0))
    st.metric("Total Documents", stats.get("total_documents", 0))

    st.markdown("---")
    st.subheader("Model Configuration")
    st.caption(f"**Embedding:** `{settings.EMBEDDING_MODEL_NAME}`")
    st.caption(f"**LLM:** `{pipeline.generator.model_name}`")
    st.caption(f"**Vector Store:** `FAISS (IndexFlat{stats.get('index_type', 'IP')})`")

    st.markdown("---")
    st.subheader("Quick Actions")
    if st.button("📥 Load Sample Documents", use_container_width=True):
        sample_dir = Path(settings.SAMPLE_DATA_DIR)
        sample_files = list(sample_dir.glob("*.txt")) + list(sample_dir.glob("*.md"))
        if sample_files:
            res = pipeline.ingest_files(sample_files, persist=True)
            st.success(f"Indexed {res['indexed_chunks']} chunks from {res['total_documents']} files!")
            st.rerun()
        else:
            st.warning("No sample files found in sample_data directory.")

    if st.button("🗑️ Clear Vector Index", use_container_width=True):
        pipeline.vector_store.clear()
        pipeline.save_index()
        st.info("Vector index cleared.")
        st.rerun()

# Main Application Tabs
tab_qa, tab_ingest, tab_eval, tab_docs = st.tabs([
    "💬 Grounded Q&A",
    "📄 Ingest Documents",
    "📊 Evaluation Harness",
    "📚 Document Index"
])

# ----------------- TAB 1: Grounded Q&A -----------------
with tab_qa:
    st.header("Ask Questions with Grounded Source Citations")
    st.caption("Answers are strictly synthesized from retrieved context with hallucination guardrails.")

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        query_input = st.text_input(
            "Enter your query:",
            placeholder="e.g., What failure rate threshold triggers the circuit breaker into an OPEN state?"
        )
    with col_q2:
        strategy_choice = st.selectbox(
            "Prompt Strategy",
            options=[
                PromptStrategy.STRICT_GROUNDED.value,
                PromptStrategy.CHAIN_OF_THOUGHT.value,
                PromptStrategy.CONCISE.value
            ],
            index=0
        )

    with st.expander("🛠️ Advanced Retrieval Parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            top_k_val = st.slider("Top K Chunks", min_value=1, max_value=10, value=4)
        with c2:
            threshold_val = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
        with c3:
            use_mmr_val = st.checkbox("Enable MMR Diversity", value=False)

    # Example question buttons
    st.markdown("**Sample Questions:**")
    eq_cols = st.columns(3)
    sample_queries = [
        "What failure rate triggers the circuit breaker, and for how long?",
        "What restocking fee applies to opened electronics in the refund policy?",
        "What quantum algorithm is used for interstellar communication satellites? (Refusal Test)"
    ]
    for i, sq in enumerate(sample_queries):
        if eq_cols[i].button(sq, key=f"btn_sq_{i}"):
            query_input = sq

    if query_input:
        with st.spinner("Retrieving context and synthesizing grounded answer..."):
            response = pipeline.query(
                question=query_input,
                top_k=top_k_val,
                similarity_threshold=threshold_val,
                prompt_strategy=PromptStrategy(strategy_choice),
                use_mmr=use_mmr_val
            )

        st.markdown("### Answer")
        st.markdown(response.answer)

        # Performance & Economics Banner
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Latency", f"{response.total_latency_ms:.1f} ms", f"Ret: {response.retrieval_latency_ms:.1f}ms")
        m2.metric("Groundedness", f"{response.validation.groundedness_score:.0%}")
        m3.metric("Tokens Used", f"{response.tokens.get('total', 0)}", f"Prompt: {response.tokens.get('prompt', 0)}")
        m4.metric("Estimated Cost", f"${response.cost_usd:.6f}")

        # Guardrails & Validation Box
        with st.container():
            st.markdown("#### 🛡️ Guardrails & Hallucination Diagnostics")
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                if response.validation.is_valid:
                    st.markdown("<p class='guardrail-pass'>✅ Guardrail Status: PASSED</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p class='guardrail-fail'>⚠️ Guardrail Status: FLAGGED</p>", unsafe_allow_html=True)
                st.write(f"- **Hallucination Rate:** `{response.validation.hallucination_score:.1%}`")
                st.write(f"- **Abstention / Refusal:** `{response.validation.is_refusal}`")

            with g_col2:
                st.write(f"- **Valid Citations:** `{response.validation.valid_citations}`")
                st.write(f"- **Invalid Citations:** `{response.validation.invalid_citations}`")
                if response.validation.warning_messages:
                    st.warning("Warnings: " + "; ".join(response.validation.warning_messages))

        # Citations Accordion
        st.markdown("#### 📑 Grounded Source Citations")
        if response.citations:
            for cite in response.citations:
                with st.expander(f"Reference [Doc {cite.doc_index}] — {cite.source} (Page {cite.page_number}) | Score: {cite.relevance_score:.2f}"):
                    st.caption(f"**Chunk ID:** `{cite.chunk_id}`")
                    st.markdown(f"> *{cite.snippet}*")
        else:
            st.info("No documents cited in this answer.")

# ----------------- TAB 2: Ingest Documents -----------------
with tab_ingest:
    st.header("Document Ingestion & Chunking Studio")

    sub_tab1, sub_tab2 = st.tabs(["📁 File Upload", "✍️ Raw Text Input"])

    with sub_tab1:
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, Markdown, Text, Word DOCX)",
            accept_multiple_files=True,
            type=["pdf", "md", "txt", "docx", "json"]
        )

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            chunk_size = st.number_input("Chunk Size (characters)", value=500, step=50, min_value=100)
        with col_c2:
            chunk_overlap = st.number_input("Chunk Overlap (characters)", value=50, step=10, min_value=0)
        with col_c3:
            strategy = st.selectbox("Chunking Strategy", ["recursive", "token", "character", "sentence"])

        if st.button("Process and Index Files", type="primary") and uploaded_files:
            import tempfile
            with st.spinner("Extracting text, chunking, and computing embeddings..."):
                temp_paths = []
                for uf in uploaded_files:
                    tf = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uf.name).suffix)
                    tf.write(uf.read())
                    tf.close()
                    temp_paths.append(Path(tf.name))

                res = pipeline.ingest_files(
                    temp_paths,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    strategy=strategy,
                    persist=True
                )
                st.success(f"Successfully indexed {res['indexed_chunks']} chunks from {len(uploaded_files)} files!")

    with sub_tab2:
        raw_title = st.text_input("Document Name", value="custom_note.txt")
        raw_content = st.text_area("Paste document text here:", height=200)
        if st.button("Index Text"):
            if raw_content.strip():
                with st.spinner("Indexing text..."):
                    c_count = pipeline.ingest_text(
                        text=raw_content,
                        source_name=raw_title,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        strategy=strategy,
                        persist=True
                    )
                    st.success(f"Indexed {c_count} chunks for '{raw_title}'!")
            else:
                st.warning("Please enter text content to index.")

# ----------------- TAB 3: Evaluation Harness -----------------
with tab_eval:
    st.header("RAG Evaluation Harness & Benchmarking")
    st.caption("Measure Answer Accuracy (F1, ROUGE-L), Hallucination Rate, Latency, and Cost across configurations.")

    eval_col1, eval_col2 = st.columns([1, 1])
    with eval_col1:
        if st.button("🚀 Run Standard Benchmark (10 Queries)", type="primary", use_container_width=True):
            harness = EvaluationHarness(pipeline)
            with st.spinner("Executing benchmark across test suite..."):
                summary = harness.run_benchmark(experiment_name="UI_Benchmark_Run")
                st.session_state["latest_summary"] = summary

    with eval_col2:
        if st.button("🔬 Run Full Ablation Study (Chunking & Prompts)", use_container_width=True):
            sample_dir = Path(settings.SAMPLE_DATA_DIR)
            sample_files = list(sample_dir.glob("*.txt")) + list(sample_dir.glob("*.md"))
            if sample_files:
                with st.spinner("Executing ablation sweeps over chunk sizes and prompt styles..."):
                    runner = AblationStudyRunner(sample_files=sample_files)
                    chunk_sums = runner.run_chunking_sweep()
                    prompt_sums = runner.run_prompt_strategy_sweep(pipeline)
                    st.session_state["ablation_results"] = chunk_sums + prompt_sums
                    st.session_state["ablation_matrix"] = runner.generate_comparison_table(chunk_sums + prompt_sums)
            else:
                st.warning("No sample files found for ablation sweep.")

    if "latest_summary" in st.session_state:
        s = st.session_state["latest_summary"]
        st.markdown(f"### Results: {s.experiment_name}")

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Answer Token F1", f"{s.mean_token_f1:.1%}")
        e2.metric("Answer ROUGE-L", f"{s.mean_rouge_l:.1%}")
        e3.metric("Hallucination Rate", f"{s.mean_hallucination_rate:.1%}")
        e4.metric("Groundedness", f"{s.mean_groundedness_score:.1%}")

        e5, e6, e7, e8 = st.columns(4)
        e5.metric("Refusal Accuracy", f"{s.refusal_accuracy:.1%}")
        e6.metric("Retrieval Hit@K", f"{s.hit_at_k:.1%}")
        e7.metric("P50 Latency", f"{s.latency_p50_ms:.1f} ms")
        e8.metric("Cost / 1k Queries", f"${s.cost_per_1000_queries:.4f}")

        st.markdown("#### Query-by-Query Breakdown")
        q_data = [item.to_dict() for item in s.detailed_items]
        st.dataframe(q_data, use_container_width=True)

    if "ablation_matrix" in st.session_state:
        st.markdown("### 🏆 Ablation Study Parameter Sweep Comparison")
        st.markdown(st.session_state["ablation_matrix"])

# ----------------- TAB 4: Document Index -----------------
with tab_docs:
    st.header("Indexed Chunks Explorer")
    chunks = pipeline.vector_store.chunks
    st.write(f"Total chunks in vector store: **{len(chunks)}**")

    if chunks:
        doc_table = [
            {
                "Chunk ID": c.chunk_id,
                "Source": c.metadata.get("source", "unknown"),
                "Page": c.metadata.get("page_number", 1),
                "Tokens": c.token_count,
                "Chars": c.char_count,
                "Snippet": c.text[:120] + "..."
            }
            for c in chunks
        ]
        st.dataframe(doc_table, use_container_width=True)
    else:
        st.info("Vector store is empty. Ingest documents or click 'Load Sample Documents' in the sidebar.")
