"""HybridRetriever module for deterministic, offline-safe hybrid search.

Combines BM25 lexical retrieval and TF-IDF cosine similarity via
Reciprocal Rank Fusion (RRF) with min_score threshold filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    """Represents a knowledge base document."""

    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredDocument:
    """Represents a retrieved document with fused and individual retrieval scores."""

    id: str
    title: str
    content: str
    score: float
    bm25_score: float
    tfidf_score: float
    bm25_rank: Optional[int]
    tfidf_rank: Optional[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ScoredDocument to standard dictionary format."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "score": round(self.score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "tfidf_score": round(self.tfidf_score, 6),
            "bm25_rank": self.bm25_rank,
            "tfidf_rank": self.tfidf_rank,
            "metadata": self.metadata,
        }


# Built-in knowledge base covering 9 core AI engineering topics
BUILTIN_DOCUMENTS: List[Document] = [
    Document(
        id="DOC-HYBRID-SEARCH",
        title="Hybrid Search in Modern Information Retrieval",
        content=(
            "Hybrid search combines dense semantic retrieval (vector embeddings) with sparse lexical "
            "retrieval (BM25 or keyword matching). Lexical search excels at exact keyword matching, "
            "technical identifiers, rare terms, and acronyms, while semantic search captures contextual "
            "meaning, synonyms, and conversational intent. Querying both index structures and fusing candidate "
            "lists overcomes the vocabulary mismatch problem while retaining precision for specific technical queries."
        ),
        metadata={"category": "retrieval", "topic": "hybrid_search"},
    ),
    Document(
        id="DOC-RRF",
        title="Reciprocal Rank Fusion (RRF) for Ranking Aggregation",
        content=(
            "Reciprocal Rank Fusion (RRF) is an unsupervised ranking fusion algorithm that merges ranked lists "
            "from multiple search systems, such as BM25 and dense or TF-IDF retrievers. For each document d, "
            "its score is defined as RRF(d) = sum(1 / (k + rank_m(d))), where k is a smoothing constant typically "
            "set to 60, and rank_m(d) is the 1-based rank position of the document in the m-th retriever. "
            "RRF eliminates the need for score calibration across disparate scoring distributions and provides "
            "monotonic consensus ranking."
        ),
        metadata={"category": "retrieval", "topic": "rrf"},
    ),
    Document(
        id="DOC-LANGGRAPH",
        title="LangGraph State Graphs and Cyclic Agent Workflows",
        content=(
            "LangGraph models agentic workflows as StateGraph objects consisting of nodes (functions representing "
            "tools, LLM calls, or decision logic) and edges (control flow transitions). Unlike linear DAG pipelines, "
            "LangGraph natively supports cyclic computation graphs, enabling reflection, iterative refinement, "
            "human-in-the-loop inspection, and bounded retry loops. State is passed between nodes via typed state schemas, "
            "with conditional edges routing execution based on intermediate outputs or verification checks."
        ),
        metadata={"category": "agents", "topic": "langgraph"},
    ),
    Document(
        id="DOC-GUARDRAILS",
        title="Guardrails and Safety Boundaries in LLM Systems",
        content=(
            "Guardrails in AI applications establish input moderation, output validation, and safety boundaries. "
            "Input guardrails detect prompt injection attacks, jailbreaks, toxic language, and out-of-scope requests "
            "before costly model inference. Output guardrails inspect model responses for hallucination, ungrounded claims, "
            "privacy leaks (PII), and policy violations. When output fails grounding checks against source context, "
            "guardrail systems trigger self-correction or return clean abstentions."
        ),
        metadata={"category": "safety", "topic": "guardrails"},
    ),
    Document(
        id="DOC-EVAL-METRICS",
        title="Evaluation Metrics for Retrieval-Augmented Generation",
        content=(
            "Comprehensive evaluation of RAG systems requires assessing both the retrieval and generation components. "
            "Retrieval metrics include Context Precision, Context Recall, and Mean Reciprocal Rank (MRR), measuring "
            "whether relevant chunks are retrieved and ranked highly. Generation metrics include Groundedness (or Faithfulness), "
            "measuring if all statements in the answer are supported by retrieved context, and Answer Relevance, evaluating "
            "whether the answer directly addresses the user query."
        ),
        metadata={"category": "evaluation", "topic": "metrics"},
    ),
    Document(
        id="DOC-FASTAPI-DEPLOY",
        title="FastAPI Architecture and Microservice Deployment",
        content=(
            "FastAPI provides a high-performance Python framework for building microservices with asynchronous endpoints "
            "(async def), automatic Pydantic v2 data validation, and OpenAPI documentation. In production AI deployments, "
            "FastAPI services are executed using Uvicorn or Gunicorn ASGI workers. Standard microservice architecture includes "
            "dedicated endpoints for /health status probes, live /metrics telemetry (tracking latency, query volume, "
            "and abstention rates), and graceful exception handling."
        ),
        metadata={"category": "deployment", "topic": "fastapi"},
    ),
    Document(
        id="DOC-DOCKER",
        title="Docker Containerization for Production AI Services",
        content=(
            "Containerizing AI microservices with Docker ensures reproducible environments and seamless deployment "
            "across cloud clusters. Best practices include using lightweight base images such as python:3.11-slim or "
            "python:3.12-slim, multi-stage builds to separate build dependencies from runtime binaries, creating a non-root "
            "system user for security isolation, configuring HEALTHCHECK directives, and pinning dependencies in requirements.txt "
            "to prevent unexpected drift."
        ),
        metadata={"category": "deployment", "topic": "docker"},
    ),
    Document(
        id="DOC-AGENTIC-RAG",
        title="Agentic RAG Patterns: Routing, Synthesis, and Self-Correction",
        content=(
            "Agentic RAG enhances conventional naive RAG through dynamic query routing, multi-step document retrieval, "
            "tool calling, and self-correction loops. Instead of blindly passing raw retrieved snippets to an LLM, "
            "the agent first classifies user intent to filter out-of-scope queries. It then performs hybrid retrieval, "
            "synthesizes answers with explicit citations, and executes a verification node to check citation grounding "
            "before returning the final response to the user."
        ),
        metadata={"category": "agents", "topic": "agentic_rag"},
    ),
    Document(
        id="DOC-GROUNDING-VERIFICATION",
        title="Citation Grounding and Hallucination Detection",
        content=(
            "Citation grounding verification ensures that every claim in an agent's response is directly attributable "
            "to cited source documents. The verification node compares tokens, n-grams, and semantic overlap between "
            "the generated answer and the retrieved reference passages. If citation overlap falls below the verification "
            "threshold or if hallucinations are detected, the agent routes back to the generation node with correction "
            "feedback, bounded by a maximum attempt limit."
        ),
        metadata={"category": "evaluation", "topic": "grounding"},
    ),
]


class HybridRetriever:
    """Deterministic, offline-safe hybrid retriever combining BM25 and TF-IDF via RRF."""

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        k: int = 60,
        min_score: float = 0.01,
    ) -> None:
        """Initialize the retriever with knowledge base documents.

        Args:
            documents: Optional list of documents. Defaults to BUILTIN_DOCUMENTS.
            k: Smoothing constant for Reciprocal Rank Fusion. Defaults to 60.
            min_score: Minimum fused score threshold for a document to be returned.
        """
        self.documents: List[Document] = documents if documents is not None else list(BUILTIN_DOCUMENTS)
        self.k: int = k
        self.min_score: float = min_score

        # Prepare corpus texts (title + content)
        self._corpus_texts = [f"{doc.title} {doc.content}" for doc in self.documents]

        # BM25 Setup
        self._tokenized_corpus = [self._tokenize(text) for text in self._corpus_texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        # TF-IDF Setup
        self._tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(self._corpus_texts)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple regex word tokenizer converting to lowercase alphanumeric tokens."""
        return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: Optional[float] = None,
    ) -> List[ScoredDocument]:
        """Retrieve documents using BM25 and TF-IDF combined via Reciprocal Rank Fusion.

        Args:
            query: The user search query string.
            top_k: Maximum number of documents to return. Defaults to 3.
            min_score: Optional threshold override. If None, uses self.min_score.

        Returns:
            List of ScoredDocument objects passing the min_score threshold, sorted by score.
        """
        threshold = min_score if min_score is not None else self.min_score
        clean_query = query.strip()
        if not clean_query:
            return []

        tokenized_query = self._tokenize(clean_query)
        if not tokenized_query:
            return []

        num_docs = len(self.documents)

        # 1. BM25 Lexical Retrieval
        bm25_scores = np.array(self._bm25.get_scores(tokenized_query), dtype=float)

        # Determine BM25 1-based ranks for matching documents
        bm25_ranks: Dict[int, int] = {}
        bm25_matching_indices = np.where(bm25_scores > 0.0)[0]
        if len(bm25_matching_indices) > 0:
            # Sort matching indices descending by score
            sorted_bm25_idx = bm25_matching_indices[np.argsort(-bm25_scores[bm25_matching_indices])]
            for rank_0, idx in enumerate(sorted_bm25_idx):
                bm25_ranks[int(idx)] = rank_0 + 1

        # 2. TF-IDF Cosine Similarity Retrieval
        query_tfidf = self._tfidf_vectorizer.transform([clean_query])
        tfidf_scores = cosine_similarity(query_tfidf, self._tfidf_matrix).flatten()

        # Determine TF-IDF 1-based ranks for matching documents
        tfidf_ranks: Dict[int, int] = {}
        tfidf_matching_indices = np.where(tfidf_scores > 0.0)[0]
        if len(tfidf_matching_indices) > 0:
            sorted_tfidf_idx = tfidf_matching_indices[np.argsort(-tfidf_scores[tfidf_matching_indices])]
            for rank_0, idx in enumerate(sorted_tfidf_idx):
                tfidf_ranks[int(idx)] = rank_0 + 1

        # 3. Reciprocal Rank Fusion (RRF)
        scored_candidates: List[ScoredDocument] = []
        for i, doc in enumerate(self.documents):
            b_rank = bm25_ranks.get(i)
            t_rank = tfidf_ranks.get(i)

            rrf_score = 0.0
            if b_rank is not None:
                rrf_score += 1.0 / (self.k + b_rank)
            if t_rank is not None:
                rrf_score += 1.0 / (self.k + t_rank)

            b_score = float(bm25_scores[i])
            t_score = float(tfidf_scores[i])

            if rrf_score >= threshold:
                scored_candidates.append(
                    ScoredDocument(
                        id=doc.id,
                        title=doc.title,
                        content=doc.content,
                        score=rrf_score,
                        bm25_score=b_score,
                        tfidf_score=t_score,
                        bm25_rank=b_rank,
                        tfidf_rank=t_rank,
                        metadata=doc.metadata,
                    )
                )

        # Sort candidates descending by RRF score; break ties by raw scores sum
        scored_candidates.sort(
            key=lambda d: (d.score, d.bm25_score + d.tfidf_score),
            reverse=True,
        )

        return scored_candidates[:top_k]
