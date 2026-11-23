"""Agentic RAG implementation using LangGraph StateGraph.

Implements nodes for:
- classify_intent: distinguishes 'knowledge_query' vs 'out_of_scope'
- retrieve_docs: retrieves documents via HybridRetriever
- generate_answer: synthesizes a cited answer or cleanly abstains
- verify_grounding: verifies citation grounding and routes back to generate_answer
  once if grounding fails (bounded by max_attempts=2).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, TypedDict

from langgraph.graph import END, START, StateGraph
from retriever import HybridRetriever, ScoredDocument


class AgentState(TypedDict, total=False):
    """LangGraph agent state schema."""

    query: str
    intent: str  # 'knowledge_query' or 'out_of_scope'
    retrieved_docs: List[Dict[str, Any]]
    answer: str
    citations: List[str]
    grounded: bool
    abstained: bool
    attempts: int
    trace: List[str]


# Technical domain vocabulary keywords for intent classification
TECHNICAL_DOMAIN_KEYWORDS: Set[str] = {
    "hybrid",
    "search",
    "rrf",
    "reciprocal",
    "rank",
    "fusion",
    "retrieval",
    "retriever",
    "bm25",
    "tfidf",
    "tf-idf",
    "vector",
    "embedding",
    "langgraph",
    "graph",
    "stategraph",
    "cyclic",
    "dag",
    "edge",
    "node",
    "guardrail",
    "guardrails",
    "moderation",
    "injection",
    "jailbreak",
    "hallucination",
    "eval",
    "evaluation",
    "metric",
    "metrics",
    "precision",
    "recall",
    "mrr",
    "grounded",
    "groundedness",
    "faithfulness",
    "relevance",
    "fastapi",
    "uvicorn",
    "gunicorn",
    "pydantic",
    "endpoint",
    "endpoints",
    "docker",
    "dockerfile",
    "container",
    "containerization",
    "multistage",
    "rag",
    "agent",
    "agentic",
    "citation",
    "citations",
    "microservice",
    "llm",
    "ai",
}

OUT_OF_SCOPE_INDICATORS: Set[str] = {
    "recipe",
    "cook",
    "bake",
    "cake",
    "pizza",
    "weather",
    "forecast",
    "movie",
    "song",
    "lyrics",
    "football",
    "soccer",
    "basketball",
    "joke",
    "funny",
    "president",
    "capital of",
    "horoscope",
    "dating",
    "restaurant",
}


def _tokenize(text: str) -> List[str]:
    """Tokenize string into lowercase alphanumeric words."""
    return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())


def classify_intent(state: AgentState) -> Dict[str, Any]:
    """Classify user query as 'knowledge_query' or 'out_of_scope'."""
    query = state.get("query", "").strip()
    trace = list(state.get("trace", []))

    tokens = set(_tokenize(query))

    # Check for explicit out-of-scope triggers
    has_out_of_scope = any(indicator in query.lower() for indicator in OUT_OF_SCOPE_INDICATORS)

    # Check for domain-specific technical keyword matches
    overlap_keywords = tokens.intersection(TECHNICAL_DOMAIN_KEYWORDS)

    if has_out_of_scope and len(overlap_keywords) == 0:
        intent = "out_of_scope"
    elif len(overlap_keywords) > 0:
        intent = "knowledge_query"
    elif len(tokens) == 0:
        intent = "out_of_scope"
    else:
        # Default heuristic: check if any word stems match technical domain
        is_tech = any(
            any(k in token or token in k for k in TECHNICAL_DOMAIN_KEYWORDS if len(token) > 3)
            for token in tokens
        )
        intent = "knowledge_query" if is_tech else "out_of_scope"

    trace.append(f"classify_intent: classified query as '{intent}'")
    return {"intent": intent, "trace": trace}


def route_after_intent(state: AgentState) -> str:
    """Conditional router routing to retrieve_docs or generate_answer."""
    if state.get("intent") == "knowledge_query":
        return "retrieve_docs"
    return "generate_answer"


def create_retrieve_docs_node(retriever: HybridRetriever):
    """Factory to create retrieve_docs node bound to a HybridRetriever instance."""

    def retrieve_docs(state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
        trace = list(state.get("trace", []))

        scored_docs: List[ScoredDocument] = retriever.retrieve(query)
        docs_data = [doc.to_dict() for doc in scored_docs]

        trace.append(
            f"retrieve_docs: retrieved {len(docs_data)} document(s) passing min_score threshold {retriever.min_score}"
        )
        return {
            "retrieved_docs": docs_data,
            "trace": trace,
        }

    return retrieve_docs


def generate_answer(state: AgentState) -> Dict[str, Any]:
    """Synthesize cited answer from retrieved docs or cleanly abstain."""
    attempts = state.get("attempts", 0) + 1
    intent = state.get("intent", "out_of_scope")
    retrieved_docs = state.get("retrieved_docs", [])
    trace = list(state.get("trace", []))

    # Case 1: Out of scope query
    if intent == "out_of_scope":
        answer = (
            "I am an AI engineering assistant specialized in hybrid search, RRF, "
            "LangGraph, guardrails, evaluation metrics, FastAPI, and Docker. "
            "Your query is outside the scope of my knowledge base."
        )
        trace.append(f"generate_answer: cleanly abstained (out_of_scope, attempt {attempts})")
        return {
            "attempts": attempts,
            "answer": answer,
            "citations": [],
            "abstained": True,
            "trace": trace,
        }

    # Case 2: No documents passed retrieval threshold
    if not retrieved_docs:
        answer = (
            "I could not find sufficient grounded information in the knowledge base "
            "to answer this query with confidence."
        )
        trace.append(
            f"generate_answer: cleanly abstained (no documents passed threshold, attempt {attempts})"
        )
        return {
            "attempts": attempts,
            "answer": answer,
            "citations": [],
            "abstained": True,
            "trace": trace,
        }

    # Case 3: Grounded synthesis from retrieved documents
    query = state.get("query", "")
    query_tokens = set(_tokenize(query))

    synthesized_sections: List[str] = []
    citations: List[str] = []

    for doc in retrieved_docs:
        doc_id = doc["id"]
        doc_title = doc["title"]
        content = doc["content"]

        # Split content into sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]

        if attempts > 1:
            # On retry attempt, select only the most direct sentences to ensure strict grounding
            scored_sentences = []
            for s in sentences:
                s_tokens = set(_tokenize(s))
                score = len(s_tokens.intersection(query_tokens))
                scored_sentences.append((score, s))
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            chosen = [s for _, s in scored_sentences[:2]]
        else:
            chosen = sentences[:2]

        if chosen:
            citations.append(doc_id)
            passage = " ".join(chosen)
            synthesized_sections.append(f"According to {doc_title} [{doc_id}], {passage}")

    answer = "\n\n".join(synthesized_sections)
    trace.append(
        f"generate_answer: synthesized answer citing {citations} (attempt {attempts})"
    )

    return {
        "attempts": attempts,
        "answer": answer,
        "citations": citations,
        "abstained": False,
        "trace": trace,
    }


def verify_grounding(state: AgentState) -> Dict[str, Any]:
    """Verify citation overlap between answer and cited documents.

    Marks grounded=True if citations are valid and content overlap is sufficient,
    or if cleanly abstained.
    """
    trace = list(state.get("trace", []))
    abstained = state.get("abstained", False)

    if abstained:
        trace.append("verify_grounding: query was abstained; grounding verified as valid.")
        return {"grounded": True, "trace": trace}

    citations = state.get("citations", [])
    retrieved_docs = state.get("retrieved_docs", [])
    answer = state.get("answer", "")

    if not citations or not retrieved_docs:
        trace.append("verify_grounding: failed (missing citations or retrieved docs).")
        return {"grounded": False, "trace": trace}

    retrieved_doc_ids = {doc["id"] for doc in retrieved_docs}
    for cit in citations:
        if cit not in retrieved_doc_ids:
            trace.append(f"verify_grounding: failed (citation {cit} not in retrieved docs).")
            return {"grounded": False, "trace": trace}

    # Verify text overlap between answer and cited document contents
    cited_docs_text = " ".join(
        doc["content"] for doc in retrieved_docs if doc["id"] in citations
    )
    doc_tokens = set(_tokenize(cited_docs_text))
    answer_tokens = _tokenize(answer)

    if not answer_tokens:
        trace.append("verify_grounding: failed (empty answer tokens).")
        return {"grounded": False, "trace": trace}

    # Check proportion of answer tokens that originate from cited texts
    overlap_count = sum(1 for token in answer_tokens if token in doc_tokens)
    overlap_ratio = overlap_count / len(answer_tokens)

    # Minimum grounding threshold: at least 40% of non-punctuation tokens grounded in citations
    grounded = overlap_ratio >= 0.40

    trace.append(
        f"verify_grounding: grounded={grounded} "
        f"(overlap_ratio={overlap_ratio:.2f}, citations={len(citations)}, attempt={state.get('attempts')})"
    )

    return {"grounded": grounded, "trace": trace}


def route_after_verification(state: AgentState) -> str:
    """Route back to generate_answer once if grounding fails, bounded by max_attempts=2."""
    grounded = state.get("grounded", False)
    attempts = state.get("attempts", 1)

    if not grounded and attempts < 2:
        return "generate_answer"
    return END


def create_agent_graph(retriever: Optional[HybridRetriever] = None):
    """Build and compile the LangGraph StateGraph agent."""
    if retriever is None:
        retriever = HybridRetriever()

    retrieve_docs_node = create_retrieve_docs_node(retriever)

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_docs", retrieve_docs_node)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("verify_grounding", verify_grounding)

    # Add edges
    workflow.add_edge(START, "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "retrieve_docs": "retrieve_docs",
            "generate_answer": "generate_answer",
        },
    )
    workflow.add_edge("retrieve_docs", "generate_answer")
    workflow.add_edge("generate_answer", "verify_grounding")
    workflow.add_conditional_edges(
        "verify_grounding",
        route_after_verification,
        {
            "generate_answer": "generate_answer",
            END: END,
        },
    )

    return workflow.compile()


class RAGAgent:
    """Wrapper class around the compiled LangGraph StateGraph."""

    def __init__(self, retriever: Optional[HybridRetriever] = None) -> None:
        self.retriever = retriever if retriever is not None else HybridRetriever()
        self.app = create_agent_graph(self.retriever)

    def run(self, query: str) -> AgentState:
        """Execute the LangGraph workflow for a given query."""
        initial_state: AgentState = {
            "query": query,
            "intent": "",
            "retrieved_docs": [],
            "answer": "",
            "citations": [],
            "grounded": False,
            "abstained": False,
            "attempts": 0,
            "trace": [],
        }
        final_state = self.app.invoke(initial_state)
        return final_state
