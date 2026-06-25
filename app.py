import os
import json
import hashlib
from typing import List, Dict, Any, Tuple

import streamlit as st
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

APP_TITLE = "MasterControl Metadata Navigator"
DATA_FILE = "mock_documents.json"


def load_documents() -> List[Dict[str, Any]]:
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def dataset_hash(items: List[Dict[str, Any]]) -> str:
    raw = json.dumps(items, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_searchable_text(item: Dict[str, Any]) -> str:
    return f"""
Document ID: {item["document_id"]}
Title: {item["title"]}
Document Type: {item["document_type"]}
Department: {item["department"]}
Owner: {item["owner"]}
Business Area: {item["business_area"]}
Process Area: {item["process_area"]}
AI Summary: {item["ai_summary"]}
Search Keywords: {", ".join(item["search_keywords"])}
Common Questions: {" | ".join(item["common_questions"])}
Use When: {" | ".join(item["use_when"])}
Related Documents: {", ".join(item["related_documents"])}
""".strip()


@st.cache_resource(show_spinner="Loading semantic search model...")
def get_embedding_engine():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner="Loading reranking model...")
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource(show_spinner="Preparing metadata index...")
def build_vector_db(data_hash: str):
    items = load_documents()
    effective_items = [item for item in items if item["status"] == "Effective"]

    docs = []

    for item in effective_items:
        docs.append(
            Document(
                page_content=make_searchable_text(item),
                metadata={
                    "document_id": item["document_id"],
                    "title": item["title"],
                    "revision": item["revision"],
                    "status": item["status"],
                    "document_type": item["document_type"],
                    "department": item["department"],
                    "owner": item["owner"],
                    "effective_date": item["effective_date"],
                    "business_area": item["business_area"],
                    "process_area": item["process_area"],
                    "ai_summary": item["ai_summary"],
                    "search_keywords": ", ".join(item["search_keywords"]),
                    "common_questions": " | ".join(item["common_questions"]),
                    "use_when": " | ".join(item["use_when"]),
                    "related_documents": ", ".join(item["related_documents"]),
                    "url": item["url"],
                },
            )
        )

    return Chroma.from_documents(docs, get_embedding_engine())


def rerank_results(
    query: str,
    candidate_results: List[Tuple[Document, float]]
) -> List[Tuple[Document, float, float]]:
    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc, _ in candidate_results]
    scores = reranker.predict(pairs)

    ranked = []

    for (doc, distance), score in zip(candidate_results, scores):
        ranked.append((doc, distance, float(score)))

    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked


def get_related_documents(
    selected_doc: Document,
    all_items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    related_ids = [
        value.strip()
        for value in selected_doc.metadata.get("related_documents", "").split(",")
        if value.strip()
    ]

    return [
        item for item in all_items
        if item["document_id"] in related_ids
    ]


def get_reason_terms(query: str, doc: Document) -> List[str]:
    query_words = {
        word.strip().lower()
        for word in query.replace("?", " ").replace(",", " ").replace(".", " ").split()
        if len(word.strip()) > 3
    }

    searchable_text = " ".join([
        doc.metadata.get("title", ""),
        doc.metadata.get("ai_summary", ""),
        doc.metadata.get("search_keywords", ""),
        doc.metadata.get("common_questions", ""),
        doc.metadata.get("use_when", "")
    ]).lower()

    matches = []

    for word in query_words:
        if word in searchable_text:
            matches.append(word.title())

    if matches:
        return sorted(matches)[:6]

    fallback = doc.metadata.get("search_keywords", "")
    return [term.strip() for term in fallback.split(",")[:6] if term.strip()]


def render_matched_topics(topics: List[str]):
    if not topics:
        return

    st.markdown("**Matched Topics**")

    chip_html = "<div class='chip-container'>"
    for topic in topics:
        chip_html += f"<span class='topic-chip'>{topic}</span>"
    chip_html += "</div>"

    st.markdown(chip_html, unsafe_allow_html=True)


def render_recommended_document(doc: Document, query: str):
    meta = doc.metadata
    reason_terms = get_reason_terms(query, doc)

    with st.container(border=True):
        left, right = st.columns([3, 1])

        with left:
            st.markdown("### Recommended Controlled Document")
            st.markdown(f"## {meta['document_id']}")
            st.markdown(f"### {meta['title']}")

        with right:
            st.markdown("**InfoCard Status**")
            st.markdown(f"**{meta['status']}**")

        st.divider()

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("**Type**")
            st.write(meta["document_type"])

        with c2:
            st.markdown("**Revision**")
            st.write(meta["revision"])

        with c3:
            st.markdown("**Department**")
            st.write(meta["department"])

        with c4:
            st.markdown("**Effective Date**")
            st.write(meta["effective_date"])

        st.markdown("**Approved Metadata Summary**")
        st.write(meta["ai_summary"])

        render_matched_topics(reason_terms)

        st.link_button("Open in MasterControl", meta["url"])


def render_supporting_documents(related_docs: List[Dict[str, Any]]):
    if not related_docs:
        return

    st.markdown("### Supporting Documents")

    for item in related_docs:
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 3, 1])

            with c1:
                st.markdown(f"**{item['document_id']}**")
                st.caption(item["document_type"])

            with c2:
                st.markdown(f"**{item['title']}**")
                st.write(item["ai_summary"])

            with c3:
                st.link_button("Open", item["url"])


def render_other_relevant_documents(results: List[Tuple[Document, float, float]]):
    if len(results) <= 1:
        return

    st.markdown("### Other Relevant Documents")

    for doc, _, _ in results[1:]:
        meta = doc.metadata

        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 4, 1])

            with c1:
                st.markdown(f"**{meta['document_id']}**")
                st.caption(meta["document_type"])

            with c2:
                st.markdown(f"**{meta['title']}**")
                st.write(meta["ai_summary"])

            with c3:
                st.link_button("Open", meta["url"])


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
    )

    st.markdown(
        """
        <style>
            .main .block-container {
                max-width: 1150px;
                padding-top: 3rem;
                padding-bottom: 3rem;
            }

            div[data-testid="stSidebar"] {
                display: none;
            }

            div[data-testid="stButton"] > button {
                width: 100%;
                height: 44px;
                font-weight: 600;
            }

            div[data-testid="stTextArea"] textarea {
                font-size: 1.05rem;
            }

            .chip-container {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 0.5rem;
                margin-bottom: 1rem;
            }

            .topic-chip {
                border: 1px solid rgba(250, 250, 250, 0.25);
                border-radius: 999px;
                padding: 0.35rem 0.75rem;
                font-size: 0.9rem;
                font-weight: 500;
                background-color: rgba(250, 250, 250, 0.06);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    items = load_documents()
    data_hash = dataset_hash(items)
    vector_db = build_vector_db(data_hash)

    st.title("MasterControl Metadata Navigator")

    st.markdown(
        """
        AI-assisted semantic discovery for controlled document InfoCards.

        This demonstration assumes MasterControl remains the validated system of record. 
        The navigator indexes only approved metadata fields and routes users back to the official MasterControl record.

        **No controlled document content is stored or generated by this application. Only approved metadata is indexed to improve document discoverability.**
        """
    )

    st.divider()

    query = st.text_area(
        "Search approved document metadata",
        placeholder="Example: I installed a new PLC on a GMP manufacturing system. What document should I review?",
        height=125,
    )

    c1, c2, c3 = st.columns([1, 1, 1])

    with c2:
        search_clicked = st.button("Search")

    if search_clicked:
        if not query.strip():
            st.error("Enter a search request before searching.")
            return

        candidate_results = vector_db.similarity_search_with_score(query, k=8)
        ranked_results = rerank_results(query, candidate_results)
        top_results = ranked_results[:3]

        if not top_results:
            st.error("No matching effective InfoCards were found.")
            return

        st.divider()

        recommended_doc = top_results[0][0]
        render_recommended_document(recommended_doc, query)

        related_docs = get_related_documents(recommended_doc, items)
        render_supporting_documents(related_docs)

        render_other_relevant_documents(top_results)

    st.divider()

    st.markdown(
        """
        **Compliance Notice**

        This application is a metadata routing layer only. It does not store, reproduce, summarize, or replace controlled SOP attachments.
        Users must open and follow the official document in the validated Quality Management System before performing any regulated activity.

        **Intended Production Architecture**

        ```text
                    MasterControl
              Validated System of Record
                         │
                  Read-only Metadata
                         │
                         ▼
           MasterControl Metadata Navigator
                         │
          Semantic Search and Metadata Ranking
                         │
                         ▼
          Recommended Controlled Documents
                         │
                         ▼
              Official MasterControl Record
        ```
        """
    )


if __name__ == "__main__":
    main()