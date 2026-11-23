"""Evaluation and benchmark suite for Agentic RAG microservice.

Measures:
1. Retrieval Hit Rate@3 on in-scope domain queries.
2. Grounding Pass Rate on in-scope synthesized answers.
3. Abstention Accuracy on out-of-scope queries.
4. Latency performance across all benchmark queries.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent import RAGAgent
from retriever import HybridRetriever


@dataclass
class BenchmarkItem:
    """Represents a benchmark test case."""

    query: str
    expected_doc_id: Optional[str] = None
    expected_abstained: bool = False
    category: str = "in_scope"


# Curated benchmark dataset
BENCHMARK_DATASET: List[BenchmarkItem] = [
    # In-Scope AI Engineering Queries
    BenchmarkItem(
        query="How does hybrid search combine dense semantic retrieval and sparse lexical BM25?",
        expected_doc_id="DOC-HYBRID-SEARCH",
        expected_abstained=False,
        category="in_scope",
    ),
    BenchmarkItem(
        query="What is the formula and smoothing constant k used in Reciprocal Rank Fusion RRF?",
        expected_doc_id="DOC-RRF",
        expected_abstained=False,
        category="in_scope",
    ),
    BenchmarkItem(
        query="How does LangGraph model cyclic agent workflows and StateGraph nodes?",
        expected_doc_id="DOC-LANGGRAPH",
        expected_abstained=False,
        category="in_scope",
    ),
    BenchmarkItem(
        query="How do input and output guardrails detect prompt injection and hallucinations?",
        expected_doc_id="DOC-GUARDRAILS",
        expected_abstained=False,
        category="in_scope",
    ),
    BenchmarkItem(
        query="What are evaluation metrics for RAG such as Context Precision and Faithfulness?",
        expected_doc_id="DOC-EVAL-METRICS",
        expected_abstained=False,
        category="in_scope",
    ),
    BenchmarkItem(
        query="How is FastAPI deployed with Uvicorn ASGI workers and telemetry endpoints?",
        expected_doc_id="DOC-FASTAPI-DEPLOY",
        expected_abstained=False,
        category="in_scope",
    ),
    BenchmarkItem(
        query="What are Docker containerization best practices for production AI services?",
        expected_doc_id="DOC-DOCKER",
        expected_abstained=False,
        category="in_scope",
    ),
    BenchmarkItem(
        query="How does agentic RAG implement dynamic routing, multi-step retrieval, and self-correction?",
        expected_doc_id="DOC-AGENTIC-RAG",
        expected_abstained=False,
        category="in_scope",
    ),
    BenchmarkItem(
        query="How does citation grounding verification calculate token overlap to prevent hallucinations?",
        expected_doc_id="DOC-GROUNDING-VERIFICATION",
        expected_abstained=False,
        category="in_scope",
    ),
    BenchmarkItem(
        query="Explain reciprocal rank fusion score aggregation across disparate retriever ranks.",
        expected_doc_id="DOC-RRF",
        expected_abstained=False,
        category="in_scope",
    ),
    # Out-of-Scope Queries
    BenchmarkItem(
        query="What is the best recipe for baking chocolate fudge brownies?",
        expected_doc_id=None,
        expected_abstained=True,
        category="out_of_scope",
    ),
    BenchmarkItem(
        query="Who won the FIFA World Cup championship soccer final?",
        expected_doc_id=None,
        expected_abstained=True,
        category="out_of_scope",
    ),
    BenchmarkItem(
        query="What is the weather forecast and temperature in Tokyo tomorrow?",
        expected_doc_id=None,
        expected_abstained=True,
        category="out_of_scope",
    ),
    BenchmarkItem(
        query="Can you tell me a funny joke about elephants and mice?",
        expected_doc_id=None,
        expected_abstained=True,
        category="out_of_scope",
    ),
    BenchmarkItem(
        query="What are the complete song lyrics to Bohemian Rhapsody?",
        expected_doc_id=None,
        expected_abstained=True,
        category="out_of_scope",
    ),
    BenchmarkItem(
        query="What is the capital city of Australia and its population?",
        expected_doc_id=None,
        expected_abstained=True,
        category="out_of_scope",
    ),
]


@dataclass
class ItemEvaluationResult:
    """Evaluation result for an individual benchmark item."""

    query: str
    category: str
    expected_doc_id: Optional[str]
    retrieved_doc_ids: List[str]
    hit_at_3: Optional[bool]
    grounded: bool
    abstained: bool
    expected_abstained: bool
    abstention_correct: Optional[bool]
    latency_ms: float
    attempts: int


@dataclass
class EvalReport:
    """Aggregate benchmark evaluation report."""

    total_in_scope: int
    total_out_of_scope: int
    hit_rate_at_3: float
    grounding_pass_rate: float
    abstention_accuracy: float
    avg_latency_ms: float
    p95_latency_ms: float
    passed_all_thresholds: bool
    target_threshold: float = 0.90
    item_results: List[ItemEvaluationResult] = field(default_factory=list)


def run_benchmark(
    dataset: Optional[List[BenchmarkItem]] = None,
    target_threshold: float = 0.90,
) -> EvalReport:
    """Execute evaluation benchmark over dataset and compute aggregate metrics.

    Args:
        dataset: List of BenchmarkItem objects (defaults to BENCHMARK_DATASET).
        target_threshold: Minimum acceptable threshold for all metrics (default 0.90).

    Returns:
        EvalReport containing individual and aggregate benchmark metrics.
    """
    items = dataset if dataset is not None else BENCHMARK_DATASET
    retriever = HybridRetriever()
    agent = RAGAgent(retriever=retriever)

    item_results: List[ItemEvaluationResult] = []
    latencies: List[float] = []

    in_scope_hits = 0
    in_scope_grounded_passes = 0
    total_in_scope = 0

    out_of_scope_correct = 0
    total_out_of_scope = 0

    for item in items:
        t0 = time.perf_counter()
        agent_result = agent.run(item.query)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        retrieved_ids = [doc["id"] for doc in agent_result.get("retrieved_docs", [])]
        grounded = bool(agent_result.get("grounded", False))
        abstained = bool(agent_result.get("abstained", False))
        attempts = int(agent_result.get("attempts", 1))

        if item.category == "in_scope":
            total_in_scope += 1
            # Check Hit@3
            hit = item.expected_doc_id in retrieved_ids[:3] if item.expected_doc_id else False
            if hit:
                in_scope_hits += 1

            # Check grounding pass (must be grounded and not abstained for in-scope)
            grounding_passed = grounded and not abstained
            if grounding_passed:
                in_scope_grounded_passes += 1

            item_results.append(
                ItemEvaluationResult(
                    query=item.query,
                    category=item.category,
                    expected_doc_id=item.expected_doc_id,
                    retrieved_doc_ids=retrieved_ids[:3],
                    hit_at_3=hit,
                    grounded=grounded,
                    abstained=abstained,
                    expected_abstained=False,
                    abstention_correct=None,
                    latency_ms=latency_ms,
                    attempts=attempts,
                )
            )
        else:
            total_out_of_scope += 1
            abstention_correct = abstained == item.expected_abstained
            if abstention_correct:
                out_of_scope_correct += 1

            item_results.append(
                ItemEvaluationResult(
                    query=item.query,
                    category=item.category,
                    expected_doc_id=None,
                    retrieved_doc_ids=retrieved_ids[:3],
                    hit_at_3=None,
                    grounded=grounded,
                    abstained=abstained,
                    expected_abstained=True,
                    abstention_correct=abstention_correct,
                    latency_ms=latency_ms,
                    attempts=attempts,
                )
            )

    hit_rate = (in_scope_hits / total_in_scope) if total_in_scope > 0 else 0.0
    grounding_rate = (
        (in_scope_grounded_passes / total_in_scope) if total_in_scope > 0 else 0.0
    )
    abstention_acc = (
        (out_of_scope_correct / total_out_of_scope) if total_out_of_scope > 0 else 0.0
    )
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_latencies = sorted(latencies)
    p95_idx = int(len(sorted_latencies) * 0.95)
    p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)] if sorted_latencies else 0.0

    passed_all = (
        hit_rate >= target_threshold
        and grounding_rate >= target_threshold
        and abstention_acc >= target_threshold
    )

    return EvalReport(
        total_in_scope=total_in_scope,
        total_out_of_scope=total_out_of_scope,
        hit_rate_at_3=round(hit_rate, 4),
        grounding_pass_rate=round(grounding_rate, 4),
        abstention_accuracy=round(abstention_acc, 4),
        avg_latency_ms=round(avg_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
        passed_all_thresholds=passed_all,
        target_threshold=target_threshold,
        item_results=item_results,
    )


def print_summary_table(report: EvalReport) -> None:
    """Print clean ASCII summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("  AGENTIC RAG BENCHMARK & EVALUATION SUITE SUMMARY")
    print("=" * 80)

    # Aggregate Metrics Table
    print(f"{'Metric':<30} | {'Target':<10} | {'Achieved':<12} | {'Status':<8}")
    print("-" * 30 + "-+-" + "-" * 10 + "-+-" + "-" * 12 + "-+-" + "-" * 8)

    metrics = [
        ("Retrieval Hit Rate@3", f">= {report.target_threshold:.0%}", f"{report.hit_rate_at_3:.1%}", "PASS" if report.hit_rate_at_3 >= report.target_threshold else "FAIL"),
        ("Grounding Pass Rate", f">= {report.target_threshold:.0%}", f"{report.grounding_pass_rate:.1%}", "PASS" if report.grounding_pass_rate >= report.target_threshold else "FAIL"),
        ("Abstention Accuracy", f">= {report.target_threshold:.0%}", f"{report.abstention_accuracy:.1%}", "PASS" if report.abstention_accuracy >= report.target_threshold else "FAIL"),
    ]

    for name, target, achieved, status_str in metrics:
        print(f"{name:<30} | {target:<10} | {achieved:<12} | {status_str:<8}")

    print("-" * 80)
    print(f"Total Test Queries: {report.total_in_scope + report.total_out_of_scope} "
          f"({report.total_in_scope} in-scope, {report.total_out_of_scope} out-of-scope)")
    print(f"Latency Profile: Avg = {report.avg_latency_ms:.2f} ms | P95 = {report.p95_latency_ms:.2f} ms")
    print(f"Overall Benchmark Status: {'PASSED ALL THRESHOLDS' if report.passed_all_thresholds else 'FAILED'}")
    print("=" * 80)

    # Detailed Item Breakdown Table
    print("\nDetailed Query Execution Breakdown:")
    print(f"{'Category':<13} | {'Hit@3':<6} | {'Grounded':<9} | {'Abst.':<6} | {'Latency':<9} | Query")
    print("-" * 13 + "-+-" + "-" * 6 + "-+-" + "-" * 9 + "-+-" + "-" * 6 + "-+-" + "-" * 9 + "-+-" + "-" * 30)

    for item in report.item_results:
        hit_str = "YES" if item.hit_at_3 else ("NO" if item.hit_at_3 is False else "N/A")
        grounded_str = "YES" if item.grounded else "NO"
        abst_str = "YES" if item.abstained else "NO"
        lat_str = f"{item.latency_ms:.1f}ms"
        q_preview = item.query[:45] + ("..." if len(item.query) > 45 else "")
        print(f"{item.category:<13} | {hit_str:<6} | {grounded_str:<9} | {abst_str:<6} | {lat_str:<9} | {q_preview}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    report = run_benchmark()
    print_summary_table(report)
    sys.exit(0 if report.passed_all_thresholds else 1)
