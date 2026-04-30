import datetime
import os
import time

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from mcp import StdioServerParameters
from vertexai.preview import rag


def now() -> dict:
    """Returns the current date and time."""
    return {
        "status": "success",
        "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def rag_corpus_status() -> dict:
    """Reports whether RAG_CORPUS is configured."""
    rag_corpus = os.getenv("RAG_CORPUS")
    if rag_corpus:
        return {"status": "success", "rag_corpus": rag_corpus}
    return {
        "status": "error",
        "message": (
            "RAG_CORPUS is not set. Configure it as "
            "'projects/<PROJECT_ID>/locations/<LOCATION>/ragCorpora/<CORPUS_ID>'."
        ),
    }


airbnb_mcp = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
        ),
    )
)

tools = [now, rag_corpus_status, airbnb_mcp]
rag_corpus = os.getenv("RAG_CORPUS")
RAG_MIN_INTERVAL_SECONDS = 2.0
_last_rag_call_ts = 0.0
_rag_query_cache = {}
_last_rag_error_ts = 0.0
RAG_ERROR_COOLDOWN_SECONDS = 30.0

if rag_corpus:
    def retrieve_destination_knowledge(query: str) -> dict:
        """Best-effort destination retrieval from Vertex RAG corpus."""
        global _last_rag_call_ts, _last_rag_error_ts

        normalized_query = query.strip().lower()
        cached = _rag_query_cache.get(normalized_query)
        if cached:
            return {
                "status": "success",
                "contexts": cached,
                "count": len(cached),
                "source": "cache",
            }

        # Avoid hammering the backend when it is already failing.
        if (time.time() - _last_rag_error_ts) < RAG_ERROR_COOLDOWN_SECONDS:
            return {
                "status": "error",
                "message": (
                    "RAG retrieval is in cooldown after a backend precondition/rate"
                    " failure. Continue with Airbnb options and ask the user to retry"
                    " shortly."
                ),
                "error_type": "RagCooldownActive",
                "error": "Recent backend precondition/rate failure",
            }

        now_ts = time.time()
        elapsed = now_ts - _last_rag_call_ts
        if elapsed < RAG_MIN_INTERVAL_SECONDS:
            time.sleep(RAG_MIN_INTERVAL_SECONDS - elapsed)

        try:
            _last_rag_call_ts = time.time()
            response = rag.retrieval_query(
                text=query,
                rag_resources=[rag.RagResource(rag_corpus=rag_corpus)],
                similarity_top_k=1,
                vector_distance_threshold=0.7,
            )
        except Exception as exc:
            error_text = str(exc).lower()
            if "quota" in error_text or "url_rejected" in error_text:
                _last_rag_error_ts = time.time()
            return {
                "status": "error",
                "message": (
                    "RAG retrieval is temporarily unavailable. Continue with Airbnb"
                    " options and clearly mention this limitation."
                ),
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }

        contexts = []
        retrieved_contexts = getattr(response, "contexts", None)
        context_items = (
            getattr(retrieved_contexts, "contexts", []) if retrieved_contexts else []
        )
        for item in context_items:
            text = getattr(item, "text", None) or str(item)
            if text:
                contexts.append(text)

        contexts = contexts[:3]
        if contexts:
            _rag_query_cache[normalized_query] = contexts

        return {
            "status": "success",
            "contexts": contexts,
            "count": len(contexts),
            "source": "rag",
        }

    tools.append(retrieve_destination_knowledge)


root_agent = Agent(
    name="travel_rag_mcp_grounded",
    model="gemini-2.5-flash",
    instruction=(
        "You are a grounded travel assistant.\n"
        "You have tools for current time, Airbnb accommodation search, and"
        " destination RAG retrieval.\n\n"
        "Behavior rules:\n"
        "1. For destination guidance (where to stay, safety, transport,"
        " seasonality, practical local advice), use"
        " `retrieve_destination_knowledge` first when available.\n"
        "2. For concrete accommodation options, use the Airbnb MCP tool.\n"
        "3. If the user asks for time/date, use `now`.\n"
        "4. If retrieval returns an error, continue without failing. Provide a"
        " best-effort destination recommendation from your general knowledge,"
        " clearly label it as 'not grounded in live corpus right now', then"
        " continue with Airbnb planning questions.\n"
        "5. If RAG is unavailable, call `rag_corpus_status`, explain setup"
        " briefly, and continue helping with available tools.\n"
        "6. Structure final responses as:\n"
        "   - Grounded destination insight\n"
        "   - Recommended stays\n"
        "   - Assumptions or missing info\n"
        "7. Never fully refuse destination advice unless user asks for legal,"
        " medical, or safety-critical guarantees; instead provide best-effort"
        " guidance with confidence caveats.\n"
    ),
    tools=tools,
)
