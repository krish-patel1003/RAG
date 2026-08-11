"""Streamlit demo UI for the production RAG system.

Talks to the FastAPI service over HTTP (``RAG_API_URL``).  Four tabs:

* **Ask**        -- query box, grounded answer, per-chunk attribution table,
  faithfulness / relevance scores, and the span waterfall for that request.
* **Corpus**     -- index stats, document list, add / delete a document.
* **Traces**     -- recent requests; drill into any trace's spans + chunk events.
* **Quality**    -- low-faithfulness requests over a window (the feedback loop).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Production RAG", page_icon="🔎", layout="wide")


def api_get(path: str, **params):
    return requests.get(f"{API}{path}", params=params, timeout=120).json()


def api_post(path: str, payload: dict):
    return requests.post(f"{API}{path}", json=payload, timeout=180).json()


def api_delete(path: str):
    return requests.delete(f"{API}{path}", timeout=60).json()


# ---- header -----------------------------------------------------------
st.title("🔎 Production RAG")
try:
    health = api_get("/health")
    stats = api_get("/stats")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Backend", health.get("backend", "?"))
    c2.metric("Embedder", health.get("embedder", "?"))
    c3.metric("Active docs", stats.get("active_docs", 0))
    c4.metric("Active chunks", stats.get("active_chunks", 0))
    c5.metric("Index version", stats.get("current_index_version", "?"))
except Exception as exc:  # noqa: BLE001
    st.error(f"Cannot reach API at {API}: {exc}")
    st.stop()

ask_tab, corpus_tab, traces_tab, quality_tab = st.tabs(
    ["Ask", "Corpus", "Traces", "Quality"]
)

# ---- Ask --------------------------------------------------------------
with ask_tab:
    question = st.text_input("Ask a question about the indexed corpus",
                             placeholder="e.g. What is HNSW and why is it used?")
    col_a, col_b = st.columns([1, 3])
    top_k = col_a.slider("top_k", 1, 15, 5)
    judge = col_b.checkbox("Run LLM-as-judge (faithfulness / relevance)", value=True)
    if st.button("Search", type="primary") and question:
        with st.spinner("Retrieving + generating..."):
            res = api_post("/query", {"question": question, "top_k": top_k, "judge": judge})
        st.subheader("Answer")
        st.markdown(res["answer"])
        meta = st.columns(4)
        meta[0].caption(f"mode: **{res.get('mode')}**")
        meta[1].caption(f"index: **{res.get('index_version')}**")
        meta[2].caption(f"latency: **{res.get('duration_ms')} ms**")
        ev = res.get("evaluation")
        if ev:
            meta[3].caption(
                f"faithfulness **{ev.get('faithfulness')}** · relevance **{ev.get('answer_relevance')}**"
            )
            if ev.get("rationale"):
                st.info(f"Judge: {ev['rationale']}")

        st.subheader("Retrieved chunks (attribution)")
        st.caption(
            "Every retrieved chunk with its vector / BM25 / rerank scores — this is "
            "what makes a wrong answer debuggable."
        )
        st.dataframe(
            [
                {
                    "#": s["n"], "doc_id": s["doc_id"], "section": s["section"],
                    "final": s["final_score"], "vector": s["vector_score"],
                    "bm25": s["bm25_score"], "rerank": s["rerank_score"],
                    "preview": s["preview"],
                }
                for s in res["sources"]
            ],
            use_container_width=True, hide_index=True,
        )
        st.caption(f"trace id: `{res['trace_id']}` — open it in the Traces tab")
        st.session_state["last_trace"] = res["trace_id"]

# ---- Corpus -----------------------------------------------------------
with corpus_tab:
    st.subheader("Add / update a document")
    with st.form("add_doc"):
        did = st.text_input("doc_id")
        txt = st.text_area("text", height=150)
        force = st.checkbox("force reindex (ignore content hash)")
        if st.form_submit_button("Index") and did and txt:
            st.json(api_post("/documents", {"doc_id": did, "text": txt, "force": force}))

    st.subheader("Indexed documents")
    docs = api_get("/documents", limit=200).get("documents", [])
    st.dataframe(docs, use_container_width=True, hide_index=True)
    st.subheader("Delete a document")
    del_id = st.text_input("doc_id to delete")
    if st.button("Delete") and del_id:
        st.json(api_delete(f"/documents/{del_id}"))

# ---- Traces -----------------------------------------------------------
with traces_tab:
    st.subheader("Recent requests")
    rows = api_get("/traces", limit=50).get("traces", [])
    st.dataframe(rows, use_container_width=True, hide_index=True)
    tid = st.text_input("Trace id", value=st.session_state.get("last_trace", ""))
    if st.button("Load trace") and tid:
        tr = api_get(f"/traces/{tid}")
        st.caption(f"total {tr.get('duration_ms')} ms · {tr['attributes'].get('index_version')}")
        for span in tr.get("spans", []):
            with st.expander(f"⏱ {span['name']} — {span['duration_ms']} ms"):
                st.json(span["attributes"])
                if span.get("events"):
                    st.markdown("**Chunk attribution events**")
                    st.dataframe(
                        [e for e in span["events"]],
                        use_container_width=True, hide_index=True,
                    )

# ---- Quality ----------------------------------------------------------
with quality_tab:
    st.subheader("Low-faithfulness requests (retrieval → answer feedback loop)")
    thr = st.slider("faithfulness threshold", 0.0, 1.0, 0.7, 0.05)
    days = st.slider("window (days)", 1, 30, 7)
    data = api_get("/traces/quality/low", threshold=thr, days=days)
    st.dataframe(data.get("traces", []), use_container_width=True, hide_index=True)
    st.caption(
        "Drill into any trace above to classify the failure: wrong document "
        "(index/model drift), wrong section (chunking boundary), or context "
        "ignored (a generation problem)."
    )
