import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

LEGAL_DISCLAIMER = (
    "\n\n---\n*DISCLAIMER: This assistant provides summaries "
    "for informational purposes only and does not constitute legal advice.*"
)


def format_source_citations(source_documents):
    citations = []
    for doc in source_documents:
        source_name = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        citations.append(f"📄 {source_name} (Page {page})")

    return "\n".join(list(set(citations)))


def is_query_in_domain(query: str, domain_keywords=None):
    if domain_keywords is None:
        domain_keywords = [
            "contract", "agreement", "clause",
            "liability", "termination",
            "party", "obligation", "payment"
        ]

    query_lower = query.lower()
    return any(word in query_lower for word in domain_keywords)


def validate_response_grounding(answer, source_documents, embeddings):

    if not source_documents:
        return "No supporting document chunks were retrieved."

    answer_embedding = embeddings.embed_query(answer)

    context_texts = [doc.page_content for doc in source_documents]
    context_embeddings = embeddings.embed_documents(context_texts)

    similarities = cosine_similarity(
        [answer_embedding],
        context_embeddings
    )[0]

    max_similarity = max(similarities)

    if max_similarity < 0.5:
        return "The response may not be strongly grounded in the retrieved document content."

    return None