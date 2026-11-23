"""FastAPI application for Agentic RAG and Evaluation microservice.

Provides endpoints:
- POST /query: Executes Agentic RAG workflow and returns synthesis with metrics and trace
- GET /metrics: Returns live in-memory counters for total queries, abstentions, grounding passes, and avg latency
- GET /health: Health probe endpoint
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from agent import RAGAgent


# ==========================================
# In-Memory Metrics Tracker
# ==========================================
class MetricsTracker:
    """Thread-safe live in-memory counters for evaluation & operational metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_queries: int = 0
        self.abstained_queries: int = 0
        self.grounded_passes: int = 0
        self.total_latency_ms: float = 0.0

    def record_query(self, latency_ms: float, abstained: bool, grounded: bool) -> None:
        """Record the metrics of an executed query in a thread-safe manner."""
        with self._lock:
            self.total_queries += 1
            self.total_latency_ms += latency_ms
            if abstained:
                self.abstained_queries += 1
            if grounded:
                self.grounded_passes += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Compute current live aggregate metrics."""
        with self._lock:
            avg_latency = (
                self.total_latency_ms / self.total_queries
                if self.total_queries > 0
                else 0.0
            )
            return {
                "total_queries": self.total_queries,
                "abstained_queries": self.abstained_queries,
                "grounded_passes": self.grounded_passes,
                "avg_latency_ms": round(avg_latency, 2),
            }

    def reset(self) -> None:
        """Reset all in-memory metric counters to zero."""
        with self._lock:
            self.total_queries = 0
            self.abstained_queries = 0
            self.grounded_passes = 0
            self.total_latency_ms = 0.0


# ==========================================
# Pydantic Schemas
# ==========================================
class QueryRequest(BaseModel):
    """Incoming query payload."""

    query: str = Field(
        ...,
        description="The natural language question to process.",
        min_length=1,
        examples=["How does Reciprocal Rank Fusion combine BM25 and TF-IDF scores?"],
    )


class QueryResponse(BaseModel):
    """Response payload for a processed query."""

    answer: str = Field(..., description="The synthesized answer or abstention notice.")
    citations: List[str] = Field(
        default_factory=list,
        description="Document identifiers cited in the synthesized answer.",
    )
    grounded: bool = Field(
        ...,
        description="Whether the answer passed grounding and citation verification.",
    )
    abstained: bool = Field(
        ...,
        description="True if the agent abstained due to out-of-scope query or insufficient context.",
    )
    attempts: int = Field(
        ...,
        description="Number of synthesis attempts performed by the agent (bounded by max_attempts=2).",
    )
    latency_ms: float = Field(
        ...,
        description="Total query processing latency in milliseconds.",
    )
    trace: List[str] = Field(
        default_factory=list,
        description="Step-by-step execution trace across StateGraph nodes.",
    )


class MetricsResponse(BaseModel):
    """Response schema for live operational metrics."""

    total_queries: int = Field(..., description="Total queries handled since service start.")
    abstained_queries: int = Field(
        ..., description="Total queries where agent abstained cleanly."
    )
    grounded_passes: int = Field(
        ..., description="Total queries passing citation grounding verification."
    )
    avg_latency_ms: float = Field(
        ..., description="Average processing latency per query in milliseconds."
    )


class HealthResponse(BaseModel):
    """Response schema for service health status."""

    status: str = Field("ok", description="Current health status of the service.")
    service: str = Field("rag-agent-eval-service", description="Service identifier.")


# ==========================================
# FastAPI App Initialization
# ==========================================
app = FastAPI(
    title="Agentic RAG & Evaluation Microservice",
    description=(
        "Production-grade microservice providing deterministic hybrid search, "
        "LangGraph StateGraph agent execution, grounding verification, and live telemetry."
    ),
    version="1.0.0",
)

metrics_tracker = MetricsTracker()
agent = RAGAgent()


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check probe",
)
async def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="ok", service="rag-agent-eval-service")


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["System"],
    summary="Live evaluation metrics",
)
async def get_metrics() -> MetricsResponse:
    """Return live in-memory telemetry counters."""
    metrics_data = metrics_tracker.get_metrics()
    return MetricsResponse(**metrics_data)


@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["RAG Agent"],
    summary="Execute Agentic RAG workflow",
)
async def process_query(request: QueryRequest) -> QueryResponse:
    """Process a user query through the LangGraph StateGraph.

    Steps executed:
    1. classify_intent (distinguishes 'knowledge_query' vs 'out_of_scope')
    2. retrieve_docs (HybridRetriever BM25 + TF-IDF RRF)
    3. generate_answer (synthesizes cited response or clean abstention)
    4. verify_grounding (checks citation overlap, loops back if ungrounded, max 2 attempts)
    """
    clean_query = request.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    t0 = time.perf_counter()
    try:
        result = agent.run(clean_query)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing agent workflow: {str(exc)}",
        ) from exc

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    answer = result.get("answer", "")
    citations = result.get("citations", [])
    grounded = bool(result.get("grounded", False))
    abstained = bool(result.get("abstained", False))
    attempts = int(result.get("attempts", 1))
    trace = result.get("trace", [])

    # Update in-memory telemetry
    metrics_tracker.record_query(
        latency_ms=latency_ms,
        abstained=abstained,
        grounded=grounded,
    )

    return QueryResponse(
        answer=answer,
        citations=citations,
        grounded=grounded,
        abstained=abstained,
        attempts=attempts,
        latency_ms=latency_ms,
        trace=trace,
    )
