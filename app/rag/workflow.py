# app/rag/workflow.py

import logging
import re
from typing import Literal

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from app.core.config import get_settings
from app.rag.state import AgentState
from app.rag.vectorstore import get_retriever


logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None


# ---------------------------------------------------------------------
# LLM AND WEB SEARCH CLIENTS
# ---------------------------------------------------------------------

def llm():
    """Create and reuse the Groq chat model."""
    global _llm

    if _llm is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing from .env")

        _llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )

    return _llm


def web_search_tool():
    """Create and reuse the Tavily search client."""
    global _web_search

    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing from .env")

        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )

    return _web_search


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def add_trace(state: AgentState, message: str):
    """Append a workflow event without modifying the existing state."""
    return [*state.get("trace", []), message]


def _text(response) -> str:
    """Safely convert an LLM response to plain text."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])

        return "\n".join(parts).strip()

    return str(content).strip()


def _normalise_question(question: str) -> str:
    """Normalize a question for reliable routing."""
    return re.sub(r"\s+", " ", question.lower()).strip()


def _is_document_question(question: str) -> bool:
    """
    Detect questions that explicitly ask about documents or private records.
    These must be answered from the private knowledge base, not public web search.
    """
    q = _normalise_question(question)

    document_phrases = (
        "from docs",
        "from the docs",
        "from document",
        "from the document",
        "from documents",
        "from the documents",
        "from uploaded file",
        "from the uploaded file",
        "from uploaded document",
        "from the uploaded document",
        "in the document",
        "in the docs",
        "in the file",
        "in the uploaded file",
        "according to the document",
        "according to the docs",
        "according to the file",
        "based on the document",
        "based on the docs",
        "based on the file",
        "company knowledge base",
        "private knowledge base",
        "internal document",
        "internal documents",
    )

    return any(phrase in q for phrase in document_phrases)


def _requires_private_evidence(question: str) -> bool:
    """
    Identify questions about private records that should not fall back
    to public web search if the knowledge base lacks sufficient evidence.
    """
    q = _normalise_question(question)

    private_terms = (
        "candidate",
        "applicant",
        "employee",
        "staff member",
        "resume",
        "résumé",
        "salary",
        "employee id",
        "personal details",
        "uploaded document",
        "uploaded file",
        "internal record",
        "internal records",
        "from docs",
        "from the docs",
        "from document",
        "from the document",
        "from documents",
        "from the documents",
        "from the file",
        "from the uploaded file",
        "from the uploaded document",
    )

    return _is_document_question(q) or any(term in q for term in private_terms)


# ---------------------------------------------------------------------
# QUESTION ROUTING
# ---------------------------------------------------------------------

def route_question(state: AgentState):
    """
    Route greetings to a direct response and knowledge-seeking questions
    to the private knowledge base.

    Uses plain-text output instead of Groq JSON mode.
    """
    question = state["question"]
    q = _normalise_question(question)

    # Explicit document questions must always use private KB retrieval.
    if _is_document_question(q):
        route = "kb"

    # Candidate information is likely to belong to an uploaded/private record.
    elif "candidate" in q or "applicant" in q:
        route = "kb"

    else:
        response = llm().invoke(
            f"""
You are the router for an enterprise IT support assistant.

Choose exactly one route:

KB
Use KB for questions requiring company knowledge, internal documents,
employee or candidate records, IT policies, VPN, password resets, MFA,
laptop setup, software access, security, email, devices, troubleshooting,
or other factual information.

DIRECT
Use DIRECT only for greetings, thanks, farewells, or casual conversation
that does not require factual information from company documents.

When uncertain, choose KB.

Question: {question}

Return exactly KB or DIRECT. Do not explain your choice.
"""
        )

        decision = _text(response).upper()
        route = "direct" if decision.startswith("DIRECT") else "kb"

    logger.info("Question routed to: %s", route.upper())

    return {
        "source_used": route,
        "trace": add_trace(state, f"Router -> {route.upper()}"),
    }


def route_after_router(
    state: AgentState,
) -> Literal["retrieve_kb", "direct_answer"]:
    return (
        "retrieve_kb"
        if state.get("source_used") == "kb"
        else "direct_answer"
    )


# ---------------------------------------------------------------------
# PRIVATE KNOWLEDGE BASE RETRIEVAL
# ---------------------------------------------------------------------

def retrieve_kb(state: AgentState):
    """Retrieve the most relevant chunks from Pinecone."""
    query = state.get("current_query") or state["question"]

    docs = get_retriever().invoke(query)

    logger.info(
        "Private KB retrieval returned %d chunks for query: %s",
        len(docs),
        query,
    )

    return {
        "kb_docs": docs,
        "trace": add_trace(
            state,
            f"Private KB retrieval -> {len(docs)} chunks",
        ),
    }


# ---------------------------------------------------------------------
# PRIVATE KB EVIDENCE GRADING
# ---------------------------------------------------------------------

def grade_kb(state: AgentState):
    """
    Grade whether retrieved chunks directly answer the user's question.

    Uses plain-text GOOD/WEAK output to avoid Groq JSON validation errors.
    """
    docs = state.get("kb_docs", [])

    if not docs:
        logger.warning("No private KB documents were retrieved.")

        return {
            "kb_grade": "weak",
            "trace": add_trace(
                state,
                "KB evidence grade -> WEAK (no chunks retrieved)",
            ),
        }

    context = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"Content:\n{doc.page_content}"
        )
        for doc in docs
    )

    response = llm().invoke(
        f"""
You are a strict evidence evaluator for an enterprise RAG system.

Determine whether the supplied private knowledge base evidence contains
enough relevant information to answer the user's exact question.

Mark GOOD only when:
- The evidence directly addresses the question.
- The required answer is explicitly stated or clearly supported.
- You do not need to invent missing facts.

Mark WEAK when:
- The evidence is empty, irrelevant, ambiguous, or incomplete.
- The requested name, date, value, or other fact cannot be found.
- The answer would require guessing.

Question:
{state["question"]}

Retrieved private knowledge base evidence:
{context}

Return exactly one word: GOOD or WEAK.
Do not provide an explanation.
"""
    )

    result = _text(response).upper()
    grade = "good" if result.startswith("GOOD") else "weak"

    logger.info("Private KB evidence grade: %s", grade.upper())

    return {
        "kb_grade": grade,
        "trace": add_trace(
            state,
            f"KB evidence grade -> {grade.upper()}",
        ),
    }


def after_kb(
    state: AgentState,
) -> Literal["generate_from_kb", "search_web", "insufficient"]:
    """Choose the next step based on evidence quality and question type."""
    if state.get("kb_grade") == "good":
        return "generate_from_kb"

    # Never search the public web for private candidate/employee records
    # or questions explicitly asking about uploaded documents.
    if _requires_private_evidence(state["question"]):
        return "insufficient"

    return "search_web"


# ---------------------------------------------------------------------
# WEB SEARCH FALLBACK
# ---------------------------------------------------------------------

def search_web(state: AgentState):
    """Search the public web only when private evidence is insufficient."""
    query = state.get("current_query") or state["question"]

    result = web_search_tool().invoke({"query": query})

    lines = []
    citations = []

    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + str(result["answer"]))

        for item in result.get("results", []):
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")

            lines.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )

            if url:
                citations.append(
                    {
                        "title": title or url,
                        "url": url,
                        "type": "web",
                    }
                )

    else:
        lines.append(str(result))

    web_results = "\n\n".join(lines)

    logger.info(
        "Tavily search completed; result text length: %d",
        len(web_results),
    )

    return {
        "web_results": web_results,
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(
            state,
            "Web fallback -> Tavily search",
        ),
    }


def grade_web(state: AgentState):
    """Check whether web evidence is relevant enough to answer."""
    web_results = state.get("web_results", "").strip()

    if not web_results:
        return {
            "web_grade": "weak",
            "trace": add_trace(
                state,
                "Web evidence grade -> WEAK (no results)",
            ),
        }

    response = llm().invoke(
        f"""
You are evaluating evidence for an enterprise IT support assistant.

Mark GOOD only if the supplied web evidence is directly relevant and
sufficient to answer the question.

Mark WEAK if the evidence is empty, unrelated, contradictory, or insufficient.

Question:
{state["question"]}

Web evidence:
{web_results}

Return exactly one word: GOOD or WEAK.
Do not explain your choice.
"""
    )

    result = _text(response).upper()
    grade = "good" if result.startswith("GOOD") else "weak"

    logger.info("Web evidence grade: %s", grade.upper())

    return {
        "web_grade": grade,
        "trace": add_trace(
            state,
            f"Web evidence grade -> {grade.upper()}",
        ),
    }


def after_web(
    state: AgentState,
) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state.get("web_grade") == "good":
        return "generate_from_web"

    if state.get("retry_count", 0) < settings.max_retries:
        return "rewrite_query"

    return "insufficient"


# ---------------------------------------------------------------------
# QUERY REWRITING
# ---------------------------------------------------------------------

def rewrite_query(state: AgentState):
    """Rewrite a query for a single additional retrieval/search attempt."""
    response = llm().invoke(
        f"""
Rewrite the following question into a concise search query.

Preserve its original intent.
Add useful technical keywords only where appropriate.
Do not answer the question.
Return only the rewritten query.

Question:
{state["question"]}
"""
    )

    rewritten = _text(response)

    if not rewritten:
        rewritten = state["question"]

    return {
        "current_query": rewritten,
        "retry_count": state.get("retry_count", 0) + 1,
        "trace": add_trace(
            state,
            f"Query rewrite -> {rewritten}",
        ),
    }


# ---------------------------------------------------------------------
# ANSWER GENERATION
# ---------------------------------------------------------------------

def generate_from_kb(state: AgentState):
    """
    Generate an answer strictly grounded in retrieved private KB chunks.
    Preserve source metadata for the API response.
    """
    docs = state.get("kb_docs", [])

    context = "\n\n".join(
        (
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n"
            f"{doc.page_content}"
        )
        for doc in docs
    )

    response = llm().invoke(
        f"""
You are an enterprise IT support copilot.

Answer the user's question using ONLY the private knowledge base evidence
provided below.

Rules:
1. Do not use outside knowledge to fill gaps.
2. Do not invent names, dates, values, policies, or procedures.
3. For questions asking for a specific name or fact, return it only if
   the evidence explicitly supports it.
4. If the evidence does not contain the answer, clearly state that the
   available excerpts do not specify it.
5. If the evidence provides steps, present them in a clear numbered list.
6. Be concise and distinguish documented facts from uncertainty.
7. Do not claim that a source contains information it does not contain.

User question:
{state["question"]}

Private knowledge base evidence:
{context}

Answer:
"""
    )

    answer = _text(response)

    citations = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "Private KB")

        if source not in seen:
            seen.add(source)
            citations.append(
                {
                    "title": str(source).replace("\\", "/").split("/")[-1],
                    "url": "",
                    "type": "private_kb",
                }
            )

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(
            state,
            "Answer generation -> PRIVATE KB",
        ),
    }


def generate_from_web(state: AgentState):
    """Generate an answer using only the external web evidence."""
    response = llm().invoke(
        f"""
You are an enterprise IT support copilot.

The private company knowledge base did not provide sufficient evidence.
Answer using ONLY the external web evidence below.

Rules:
1. Do not invent facts not supported by the supplied evidence.
2. Clearly identify the response as external web information.
3. For company-managed systems, advise the user to validate changes
   with their internal IT team when appropriate.
4. If the evidence does not answer the question, say so.

Question:
{state["question"]}

External web evidence:
{state.get("web_results", "")}

Answer:
"""
    )

    return {
        "answer": _text(response),
        "source_used": "web_search",
        "trace": add_trace(
            state,
            "Answer generation -> WEB SEARCH",
        ),
    }


def direct_answer(state: AgentState):
    """Handle greetings and other non-retrieval conversation."""
    response = llm().invoke(
        f"""
Respond briefly and naturally to this casual message.
Do not claim to have searched documents or retrieved evidence.

Message:
{state["question"]}
"""
    )

    return {
        "answer": _text(response),
        "source_used": "direct",
        "trace": add_trace(
            state,
            "Direct response -> no retrieval",
        ),
    }


def insufficient(state: AgentState):
    """Return a safe response when reliable evidence is unavailable."""
    if _requires_private_evidence(state["question"]):
        answer = (
            "I couldn't find enough reliable evidence in the retrieved "
            "company documents to answer that question. I won't guess or "
            "use public web search for private records. Please check the "
            "source document or provide the relevant document."
        )
    else:
        answer = (
            "I couldn't find enough reliable evidence in the company "
            "knowledge base or external search to answer confidently. "
            "Please contact the IT help desk or provide more details."
        )

    return {
        "answer": answer,
        "source_used": "insufficient_evidence",
        "trace": add_trace(
            state,
            "Stopped -> insufficient reliable evidence",
        ),
    }


# ---------------------------------------------------------------------
# LANGGRAPH WORKFLOW
# ---------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    nodes = {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }

    for name, function in nodes.items():
        graph.add_node(name, function)

    graph.add_edge(START, "route_question")

    graph.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("retrieve_kb", "grade_kb")

    graph.add_conditional_edges(
        "grade_kb",
        after_kb,
        {
            "generate_from_kb": "generate_from_kb",
            "search_web": "search_web",
            "insufficient": "insufficient",
        },
    )

    graph.add_edge("search_web", "grade_web")

    graph.add_conditional_edges(
        "grade_web",
        after_web,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )

    # One retry returns to private KB retrieval with the rewritten query.
    graph.add_edge("rewrite_query", "retrieve_kb")

    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)

    return graph.compile()


agent_graph = build_graph()


def ask(question: str):
    """Run the complete agentic RAG workflow for a user question."""
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }

    return agent_graph.invoke(initial)