"""Comprehensive automated test suite for Agentic RAG microservice.

Directly imports and validates:
1. retriever (HybridRetriever ranking, thresholding, and built-in corpus)
2. agent (LangGraph StateGraph intent routing, cited synthesis, abstention, and grounding retry)
3. api (FastAPI TestClient testing GET /health, POST /query, and live GET /metrics telemetry)
4. eval_suite (benchmark threshold evaluation >= 0.90 across all key metrics)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from retriever import BUILTIN_DOCUMENTS, Document, HybridRetriever, ScoredDocument
from agent import (
    AgentState,
    RAGAgent,
    classify_intent,
    generate_answer,
    route_after_intent,
    route_after_verification,
    verify_grounding,
)
from api import app, metrics_tracker
from eval_suite import BENCHMARK_DATASET, run_benchmark


# ==========================================
# 1. Knowledge Base & Requirements Tests
# ==========================================
def test_requirements_file_exists():
    """Verify requirements.txt exists and contains all required dependencies."""
    req_path = os.path.join(ROOT_DIR, "requirements.txt")
    assert os.path.exists(req_path), f"requirements.txt not found at {req_path}"
    with open(req_path, "r", encoding="utf-8") as f:
        content = f.read()

    expected_pkgs = [
        "fastapi",
        "uvicorn",
        "pydantic>=2.0.0",
        "langgraph>=0.2.0",
        "langchain-core>=0.3.0",
        "rank-bm25",
        "scikit-learn",
        "httpx",
        "pytest>=8.0.0",
    ]
    for pkg in expected_pkgs:
        assert pkg in content, f"Missing {pkg} in requirements.txt"


def test_builtin_documents_corpus():
    """Verify built-in knowledge base contains at least 8 documents covering all required topics."""
    assert len(BUILTIN_DOCUMENTS) >= 8
    doc_ids = {doc.id for doc in BUILTIN_DOCUMENTS}
    expected_ids = {
        "DOC-HYBRID-SEARCH",
        "DOC-RRF",
        "DOC-LANGGRAPH",
        "DOC-GUARDRAILS",
        "DOC-EVAL-METRICS",
        "DOC-FASTAPI-DEPLOY",
        "DOC-DOCKER",
        "DOC-AGENTIC-RAG",
        "DOC-GROUNDING-VERIFICATION",
    }
    assert expected_ids.issubset(doc_ids)

    # Verify key topics appear across document corpus
    corpus_text = " ".join(
        f"{doc.title} {doc.content}".lower() for doc in BUILTIN_DOCUMENTS
    )
    required_keywords = [
        "hybrid search",
        "reciprocal rank fusion",
        "langgraph",
        "state graphs",
        "guardrails",
        "evaluation metrics",
        "fastapi",
        "docker",
    ]
    for kw in required_keywords:
        assert kw in corpus_text, f"Keyword '{kw}' missing from knowledge base corpus"


# ==========================================
# 2. HybridRetriever Ranking & Thresholding
# ==========================================
def test_hybrid_retriever_ranking():
    """Verify HybridRetriever correctly scores and ranks documents via BM25 + TF-IDF RRF."""
    retriever = HybridRetriever(k=60, min_score=0.01)

    # In-scope query for RRF algorithm
    results = retriever.retrieve("What is the formula and smoothing constant k in Reciprocal Rank Fusion RRF?", top_k=3)
    assert len(results) > 0
    top_doc = results[0]
    assert top_doc.id == "DOC-RRF"
    assert top_doc.score > 0.01
    assert top_doc.bm25_rank is not None or top_doc.tfidf_rank is not None
    # Maximum theoretical RRF score with k=60 is 2 / (60 + 1) = 2/61 ~= 0.03278
    assert top_doc.score <= (1.0 / 61.0) + (1.0 / 61.0) + 1e-9


def test_hybrid_retriever_min_score_threshold_filtering():
    """Verify min_score threshold filters out irrelevant queries and weak candidates."""
    retriever = HybridRetriever(k=60, min_score=0.01)

    # Completely unrelated query
    unrelated_res = retriever.retrieve("xylophone quantum avocado sandwich 998877")
    assert len(unrelated_res) == 0

    # Very high threshold that no document can reach
    high_thresh_res = retriever.retrieve("Reciprocal Rank Fusion", min_score=0.50)
    assert len(high_thresh_res) == 0

    # Empty and whitespace queries
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   \t  \n ") == []


# ==========================================
# 3. LangGraph Agent Routing & Node Tests
# ==========================================
def test_agent_routing_in_scope():
    """Verify in-scope query routes through retrieve -> grounded -> answered."""
    agent = RAGAgent()
    query = "How does LangGraph model cyclic agent workflows and StateGraph nodes?"

    state = agent.run(query)

    assert state["intent"] == "knowledge_query"
    assert state["abstained"] is False
    assert state["grounded"] is True
    assert len(state["citations"]) > 0
    assert "DOC-LANGGRAPH" in state["citations"]
    assert "DOC-LANGGRAPH" in state["answer"]
    assert any("classify_intent" in step for step in state["trace"])
    assert any("retrieve_docs" in step for step in state["trace"])
    assert any("generate_answer" in step for step in state["trace"])
    assert any("verify_grounding" in step for step in state["trace"])


def test_agent_routing_out_of_scope():
    """Verify out-of-scope query cleanly abstains with abstained=True and no citations."""
    agent = RAGAgent()
    query = "What is the best recipe for baking chocolate brownies?"

    state = agent.run(query)

    assert state["intent"] == "out_of_scope"
    assert state["abstained"] is True
    assert state["grounded"] is True
    assert state["citations"] == []
    assert "outside the scope" in state["answer"].lower()
    assert any("cleanly abstained" in step for step in state["trace"])


def test_grounding_verifier_retry_loop_bounded():
    """Verify grounding verifier loops back once if ungrounded, bounded by max_attempts=2."""
    # Attempt 1: ungrounded -> route back to generate_answer
    state_attempt_1: AgentState = {"grounded": False, "attempts": 1}
    assert route_after_verification(state_attempt_1) == "generate_answer"

    # Attempt 2: ungrounded -> terminate at END (bounded by max_attempts=2)
    state_attempt_2: AgentState = {"grounded": False, "attempts": 2}
    assert route_after_verification(state_attempt_2) == "__end__"

    # Grounded -> terminate at END
    state_grounded: AgentState = {"grounded": True, "attempts": 1}
    assert route_after_verification(state_grounded) == "__end__"


def test_verify_grounding_overlap_logic():
    """Verify grounding check passes with high overlap and fails on hallucination."""
    doc = {
        "id": "DOC-GUARDRAILS",
        "title": "Guardrails in LLM Systems",
        "content": "Input guardrails detect prompt injection attacks, jailbreaks, and toxic language.",
    }

    # Pass case: response derived from cited doc
    pass_state: AgentState = {
        "answer": "According to Guardrails in LLM Systems [DOC-GUARDRAILS], input guardrails detect prompt injection attacks and jailbreaks.",
        "citations": ["DOC-GUARDRAILS"],
        "retrieved_docs": [doc],
        "abstained": False,
        "trace": [],
    }
    assert verify_grounding(pass_state)["grounded"] is True

    # Hallucinated citation failure
    fail_state_cit: AgentState = {
        "answer": "Claim without source.",
        "citations": ["NON_EXISTENT_DOC"],
        "retrieved_docs": [doc],
        "abstained": False,
        "trace": [],
    }
    assert verify_grounding(fail_state_cit)["grounded"] is False

    # Zero token overlap failure
    fail_state_overlap: AgentState = {
        "answer": "Planetary orbits around celestial galaxies produce interstellar gravitational waves.",
        "citations": ["DOC-GUARDRAILS"],
        "retrieved_docs": [doc],
        "abstained": False,
        "trace": [],
    }
    assert verify_grounding(fail_state_overlap)["grounded"] is False


# ==========================================
# 4. FastAPI Endpoints & In-Memory Telemetry
# ==========================================
@pytest.fixture(autouse=True)
def reset_telemetry():
    """Reset live telemetry counters before each test."""
    metrics_tracker.reset()


def test_fastapi_health_endpoint():
    """Verify GET /health returns ok status."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "rag-agent-eval-service"


def test_fastapi_query_and_metrics_telemetry():
    """Verify POST /query returns required fields and updates GET /metrics live counters."""
    client = TestClient(app)

    # Initial metrics check
    initial_metrics = client.get("/metrics").json()
    assert initial_metrics["total_queries"] == 0
    assert initial_metrics["abstained_queries"] == 0
    assert initial_metrics["grounded_passes"] == 0
    assert initial_metrics["avg_latency_ms"] == 0.0

    # 1. In-scope query
    query_payload_1 = {"query": "Explain Reciprocal Rank Fusion RRF ranking aggregation."}
    resp_1 = client.post("/query", json=query_payload_1)
    assert resp_1.status_code == 200
    data_1 = resp_1.json()
    assert "answer" in data_1
    assert "DOC-RRF" in data_1["citations"]
    assert data_1["grounded"] is True
    assert data_1["abstained"] is False
    assert data_1["attempts"] >= 1
    assert data_1["latency_ms"] > 0
    assert len(data_1["trace"]) > 0

    # 2. Out-of-scope query
    query_payload_2 = {"query": "What is the weather forecast and temperature in Tokyo tomorrow?"}
    resp_2 = client.post("/query", json=query_payload_2)
    assert resp_2.status_code == 200
    data_2 = resp_2.json()
    assert data_2["abstained"] is True
    assert data_2["citations"] == []
    assert data_2["grounded"] is True

    # 3. Check updated live metrics
    updated_metrics = client.get("/metrics").json()
    assert updated_metrics["total_queries"] == 2
    assert updated_metrics["abstained_queries"] == 1
    assert updated_metrics["grounded_passes"] == 2
    assert updated_metrics["avg_latency_ms"] > 0.0


def test_fastapi_query_validation():
    """Verify POST /query rejects empty payload."""
    client = TestClient(app)
    resp = client.post("/query", json={"query": ""})
    assert resp.status_code in [400, 422]


# ==========================================
# 5. Evaluation Suite Benchmark Tests
# ==========================================
def test_eval_suite_benchmark_thresholds():
    """Verify benchmark metrics pass >= 0.90 threshold across all key dimensions."""
    report = run_benchmark(target_threshold=0.90)

    # Assert Retrieval Hit Rate@3 >= 0.90
    assert report.hit_rate_at_3 >= 0.90, f"Hit Rate@3 too low: {report.hit_rate_at_3:.2%}"

    # Assert Grounding Pass Rate >= 0.90
    assert (
        report.grounding_pass_rate >= 0.90
    ), f"Grounding Pass Rate too low: {report.grounding_pass_rate:.2%}"

    # Assert Abstention Accuracy >= 0.90
    assert (
        report.abstention_accuracy >= 0.90
    ), f"Abstention Accuracy too low: {report.abstention_accuracy:.2%}"

    # Assert overall pass
    assert report.passed_all_thresholds is True
    assert report.avg_latency_ms > 0
