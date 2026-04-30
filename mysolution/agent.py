import datetime
import os

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from mcp import StdioServerParameters
from vertexai.preview import rag


def now() -> dict:
    """Returns the current date and time."""
    return {
        "status": "success",
        "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def rag_corpus_status() -> dict:
    """Reports whether RAG corpus is configured."""
    rag_corpus = os.getenv("RAG_CORPUS")
    if rag_corpus:
        return {"status": "success", "rag_corpus": rag_corpus}
    return {
        "status": "error",
        "message": (
            "RAG_CORPUS is not set. Add it to your environment/.env as "
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

rag_corpus = os.getenv("RAG_CORPUS")
tools = [now, rag_corpus_status, airbnb_mcp]

if rag_corpus:
    retrieve_destination_knowledge = VertexAiRagRetrieval(
        name="retrieve_destination_knowledge",
        description=(
            "Retrieve destination travel context (best areas, safety, local transport,"
            " seasonality, and practical travel tips) from the configured RAG corpus."
        ),
        rag_resources=[rag.RagResource(rag_corpus=rag_corpus)],
        similarity_top_k=3,
        vector_distance_threshold=0.7,
    )
    tools.append(retrieve_destination_knowledge)


root_agent = Agent(
    name="travel_grounded_advisor",
    model="gemini-2.5-flash",
    instruction=(
        "You are a grounded travel assistant.\n"
        "You have tools for: current time, Airbnb accommodation search, and optional"
        " destination RAG retrieval.\n\n"
        "Behavior rules:\n"
        "1. For destination advice (where to stay, neighborhood safety, transport,"
        " seasonality, practical local tips), use `retrieve_destination_knowledge`"
        " first when available.\n"
        "2. For accommodation options, use the Airbnb MCP tool.\n"
        "3. If the user asks time/date, use `now`.\n"
        "4. If RAG is not available, call `rag_corpus_status`, explain the missing"
        " setup briefly, and continue helping with Airbnb/time tools.\n"
        "5. In final answers, separate sections:\n"
        "   - Grounded destination insight\n"
        "   - Recommended stays\n"
        "   - Assumptions or missing info\n"
    ),
    tools=tools,
)
