import datetime
from functools import lru_cache
import random
import time

from google.adk.agents import Agent
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.genai import types
from mcp import StdioServerParameters

from .local_rag import build_local_retriever_from_env


def now() -> dict:
    """Returns the current date and time."""
    return {
        "status": "success",
        "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@lru_cache(maxsize=1)
def _get_retriever():
    return build_local_retriever_from_env()


def retrieve_local_destination_knowledge(query: str) -> dict:
    """
    Retrieve travel insights from local docs using FAISS + sentence-transformers.
    """
    try:
        retriever = _get_retriever()
        chunks = retriever.search(query=query, top_k=3)
    except Exception as exc:
        return {
            "status": "error",
            "message": (
                "Local RAG retrieval is unavailable. Ensure dependencies are installed "
                "and LOCAL_RAG_PATHS points to readable markdown content."
            ),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }

    if not chunks:
        return {"status": "success", "count": 0, "results": []}

    results = [
        {"source": chunk.source, "text": chunk.text[:1200]}
        for chunk in chunks
    ]
    return {"status": "success", "count": len(results), "results": results}


def handle_model_error_for_rate_limits(context, llm_request, exc, **kwargs):
    """
    Convert Gemini 429 errors into a graceful assistant response.
    """
    error_text = str(exc)
    normalized = error_text.lower()
    if "429" not in normalized and "resource_exhausted" not in normalized:
        return None

    # Gentle jitter so immediate retries from users are less synchronized.
    time.sleep(0.4 + random.random() * 0.6)

    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    text=(
                        "I hit a temporary model rate limit (429 RESOURCE_EXHAUSTED). "
                        "Please retry in about 10-20 seconds.\n\n"
                        "In the meantime, I can still proceed with a shorter response "
                        "or collect missing trip details (dates, guests, budget) to "
                        "minimize token usage on the next attempt."
                    )
                )
            ],
        ),
        error_code="RESOURCE_EXHAUSTED",
        error_message=error_text,
        turn_complete=True,
    )


airbnb_mcp = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
        ),
    )
)


root_agent = Agent(
    name="travel_local_rag_mcp",
    model="gemini-2.5-flash",
    instruction=(
        "You are a grounded travel assistant with local RAG and Airbnb tools.\n"
        "Use `retrieve_local_destination_knowledge` for destination context "
        "(neighborhoods, transport, seasonality, practical tips).\n"
        "Use Airbnb MCP for concrete accommodation options.\n"
        "Use `now` when date/time context is needed.\n\n"
        "Airbnb tool geo-disambiguation rules:\n"
        "1. Always search with explicit city + country (example: 'Lisbon, Portugal').\n"
        "2. If results look like the wrong place (non-EUR pricing, wrong coordinates,"
        " wrong country clues), retry up to 2 times with stricter location strings:\n"
        "   - 'Lisbon, Portugal'\n"
        "   - 'Chiado, Lisbon, Portugal' or 'Alfama, Lisbon, Portugal'\n"
        "3. Never stop after one ambiguous attempt.\n"
        "4. If listing-level results remain ambiguous after retries, still provide:\n"
        "   - 2-3 neighborhood recommendations\n"
        "   - 3 actionable Airbnb search URLs for Lisbon neighborhoods\n"
        "   - a short note that direct listing extraction was ambiguous.\n\n"
        "If local retrieval fails, continue with best-effort advice and clearly "
        "state it is not grounded.\n"
        "Structure responses with:\n"
        "- Destination insight\n"
        "- Recommended stays\n"
        "- Assumptions / missing info\n"
        "In `Recommended stays`, include at least 2 options. If Airbnb tool returns"
        " URLs, include them inline.\n"
        "Keep each section concise so all sections fit in one response.\n"
    ),
    generate_content_config=types.GenerateContentConfig(max_output_tokens=1200),
    on_model_error_callback=handle_model_error_for_rate_limits,
    tools=[now, retrieve_local_destination_knowledge, airbnb_mcp],
)
