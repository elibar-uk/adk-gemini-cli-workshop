import datetime
import os

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

if rag_corpus:
    def retrieve_destination_knowledge(query: str) -> dict:
        """Best-effort destination retrieval from Vertex RAG corpus."""
        try:
            response = rag.retrieval_query(
                text=query,
                rag_resources=[rag.RagResource(rag_corpus=rag_corpus)],
                similarity_top_k=3,
                vector_distance_threshold=0.7,
            )
        except Exception as exc:
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

        return {
            "status": "success",
            "contexts": contexts[:3],
            "count": len(contexts),
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
        "4. If retrieval returns an error, continue without failing and state"
        " that destination grounding is temporarily unavailable.\n"
        "5. If RAG is unavailable, call `rag_corpus_status`, explain setup"
        " briefly, and continue helping with available tools.\n"
        "6. Structure final responses as:\n"
        "   - Grounded destination insight\n"
        "   - Recommended stays\n"
        "   - Assumptions or missing info\n"
    ),
    tools=tools,
)
