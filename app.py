import os
from collections import Counter
from typing import Any, Dict, List, Tuple

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from data_service import dataset_hash, load_documents, load_events
from search_service import (
    apply_event_filters,
    combine_document_results,
    event_searchable_text,
    lexical_score,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
APP_TITLE = "MasterControl Quality Knowledge Navigator"


def make_document_text(item: Dict[str, Any]) -> str:
    return "\n".join([
        f"Document ID: {item['document_id']}", f"Title: {item['title']}",
        f"Document Type: {item['document_type']}", f"Department: {item['department']}",
        f"Business Area: {item['business_area']}", f"Process Area: {item['process_area']}",
        f"Approved Metadata Summary: {item['ai_summary']}",
        f"Search Keywords: {', '.join(item['search_keywords'])}",
        f"Common Questions: {' | '.join(item['common_questions'])}",
        f"Use When: {' | '.join(item['use_when'])}",
    ])


@st.cache_resource(show_spinner="Loading semantic search model...")
def get_embedding_engine():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading reranking model...")
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource(show_spinner="Preparing controlled-document index...")
def build_document_index(data_hash: str):
    del data_hash
    docs = []
    for item in load_documents():
        if item["status"] != "Effective":
            continue
        metadata = dict(item)
        for key in ("search_keywords", "common_questions", "use_when", "related_documents"):
            metadata[key] = " | ".join(metadata[key])
        docs.append(Document(page_content=make_document_text(item), metadata=metadata))
    return Chroma.from_documents(
        docs,
        get_embedding_engine(),
        collection_name="controlled_documents",
    )


@st.cache_resource(show_spinner="Preparing historical-event index...")
def build_event_index(data_hash: str):
    del data_hash
    docs = [
        Document(
            page_content=event_searchable_text(item),
            metadata={"record_id": item["record_id"]},
        )
        for item in load_events()
    ]
    return Chroma.from_documents(
        docs,
        get_embedding_engine(),
        collection_name="historical_events",
    )


def search_documents(query: str, vector_db: Chroma) -> List[Tuple[Document, float]]:
    candidates = vector_db.similarity_search_with_score(query, k=8)
    scores = get_reranker().predict([(query, doc.page_content) for doc, _ in candidates])
    reranked = [(doc, distance, float(score)) for (doc, distance), score in zip(candidates, scores)]
    return combine_document_results(query, reranked)[:3]


def search_events(
    query: str,
    items: List[Dict[str, Any]],
    vector_db: Chroma,
    limit: int = 10,
) -> List[Tuple[Dict[str, Any], float]]:
    if not items:
        return []
    allowed = {item["record_id"]: item for item in items}
    candidates = vector_db.similarity_search_with_score(query, k=min(20, len(load_events())))
    candidates = [(doc, distance) for doc, distance in candidates if doc.metadata["record_id"] in allowed]
    semantic_scores = {}
    if candidates:
        scores = get_reranker().predict([(query, doc.page_content) for doc, _ in candidates])
        semantic_scores = {
            doc.metadata["record_id"]: float(score)
            for (doc, _), score in zip(candidates, scores)
        }
    ranked = []
    for record_id, item in allowed.items():
        lexical = lexical_score(query, event_searchable_text(item))
        semantic = semantic_scores.get(record_id, -10.0)
        ranked.append((item, semantic + lexical * 2.0))
    return sorted(
        ranked,
        key=lambda row: (row[1], row[0]["event_date"]),
        reverse=True,
    )[:limit]


def multiselect_filter(label: str, key: str, events: List[Dict[str, Any]]) -> List[str]:
    options = sorted({str(item[key]) for item in events if item.get(key)})
    return st.multiselect(label, options, placeholder="All")


def render_filters(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    years = [int(item["event_date"][:4]) for item in events]
    with st.expander("Narrow historical-event search", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sites = multiselect_filter("Site", "site", events)
        with c2:
            types = multiselect_filter("Record type", "event_type", events)
        with c3:
            departments = multiselect_filter("Department", "department", events)
        with c4:
            statuses = multiselect_filter("Status", "status", events)
        year_range = st.slider("Event year", min(years), max(years), (min(years), max(years)))
    return {"site": sites, "event_type": types, "department": departments,
            "status": statuses, "year_range": year_range}


def render_event_summary(results: List[Tuple[Dict[str, Any], float]]) -> None:
    st.subheader("Potentially Related Historical Records")
    st.caption("Potential matches—not a recurrence determination. Verify each authoritative source record before use.")
    closed = sum(item["status"] == "Closed" for item, _ in results)
    categories = Counter(item["root_cause_category"] for item, _ in results)
    c1, c2, c3 = st.columns(3)
    c1.metric("Potential matches", len(results))
    c2.metric("Closed records", closed)
    c3.metric("Most common recorded category", categories.most_common(1)[0][0] if categories else "—")


def render_event(item: Dict[str, Any], rank: int) -> None:
    status = item["status"]
    heading = f"{item['event_date']} · {item['record_id']} · {item['title']}"
    with st.expander(heading, expanded=rank == 0):
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Type**  \n{item['event_type']}")
        c2.markdown(f"**Status**  \n{status}")
        c3.markdown(f"**Site / Area**  \n{item['site']} / {item['area']}")
        c4.markdown(f"**Equipment**  \n{item['equipment_id']} · {item['equipment_type']}")
        st.markdown("**Recorded problem statement**")
        st.write(item["problem_statement"])
        left, right = st.columns(2)
        with left:
            st.markdown("**Recorded failure mode**")
            st.write(item["failure_mode"])
            st.markdown("**Confirmed root cause in source metadata**")
            st.write(f"{item['root_cause_category']}: {item['confirmed_root_cause']}")
        with right:
            st.markdown("**Corrective / preventive actions**")
            st.write(item["corrective_action"])
            st.write(item["preventive_action"])
            st.markdown("**Recorded recurrence classification**")
            st.write(item["recurrence_classification"])
        st.caption("Related records: " + ", ".join(item["related_records"]))
        st.link_button(f"Open authoritative record {item['record_id']}", item["url"])


def render_document(doc: Document, primary: bool = False) -> None:
    meta = doc.metadata
    with st.container(border=True):
        if primary:
            st.markdown("**Top controlled-document match**")
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"### {meta['document_id']} · {meta['title']}")
            st.write(meta["ai_summary"])
        with c2:
            st.markdown(f"**{meta['status']}**")
            st.caption(f"{meta['document_type']} · {meta['revision']}")
        st.link_button(f"Open {meta['document_id']} in MasterControl", meta["url"])


def render_page_style() -> None:
    st.markdown("""
    <style>
      .main .block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem;}
      div[data-testid="stButton"] > button {width: 100%; font-weight: 650;}
      div[data-testid="stTextArea"] textarea {font-size: 1.05rem;}
    </style>
    """, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    render_page_style()
    documents = load_documents()
    events = load_events()
    vector_db = build_document_index(dataset_hash(documents))
    event_vector_db = build_event_index(dataset_hash(events))

    st.title(APP_TITLE)
    st.write("Search prior quality events and effective controlled-document metadata using ordinary engineering language.")
    st.info("**Decision-support boundary:** Results support discovery and reflection only. MasterControl remains the validated system of record, and authorized personnel must verify records and make all quality decisions.")

    with st.form("quality-search"):
        query = st.text_area(
            "What are you trying to understand?",
            placeholder="Have we had issues with centrifugal pumps running dry and breaking in the past?",
            height=110,
        )
        filters = render_filters(events)
        submitted = st.form_submit_button("Search quality knowledge", use_container_width=True)

    if submitted:
        if not query.strip():
            st.error("Enter a question before searching.")
        else:
            filtered_events = apply_event_filters(events, filters)
            event_results = search_events(query, filtered_events, event_vector_db, limit=10)
            document_results = search_documents(query, vector_db)
            st.divider()
            if event_results:
                render_event_summary(event_results)
                for rank, (item, _) in enumerate(event_results):
                    render_event(item, rank)
            else:
                st.warning("No potentially related historical records met the search terms and filters. This does not establish that no prior event exists.")
            st.divider()
            st.subheader("Applicable Effective Controlled Documents")
            st.caption("These are metadata matches. Open and follow the effective controlled record in MasterControl.")
            for rank, (doc, _) in enumerate(document_results):
                render_document(doc, primary=rank == 0)

    st.divider()
    st.markdown("""
    **Demonstration and compliance notice**

    This prototype uses fictional metadata. It does not reproduce controlled attachments, determine recurrence,
    assign root cause, assess product impact, or replace investigation and Quality-unit review. Production use
    requires approved intended use, risk assessment, access controls, auditability, validation evidence, change
    control, and verified read-only synchronization with MasterControl.

    **Intended production flow:** MasterControl → permission-filtered read-only metadata → validated search index
    → source-cited potential matches → authoritative MasterControl record.
    """)


if __name__ == "__main__":
    main()
