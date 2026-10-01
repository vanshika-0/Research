import asyncio
import os
import certifi
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import json
import operator
import uuid

from langgraph.types import Command, interrupt
from langgraph.graph import StateGraph, START, END
#paralle agent calling 
from langgraph.constants import Send

from typing import TypedDict, Annotated, Any

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)

from langchain_mcp_adapters.client import MultiServerMCPClient

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# =========================================================
# LLM
# =========================================================

HUGGINGFACE_TOKEN = os.getenv(
    "HUGGINGFACEHUB_ACCESS_TOKEN"
)

if not HUGGINGFACE_TOKEN:
    raise ValueError(
        "HUGGINGFACEHUB_ACCESS_TOKEN is missing from .env"
    )


llm = HuggingFaceEndpoint(
    # This is configurable because not every Hugging Face model is
    # available through the hosted inference provider.  The NVIDIA
    # NVFP4 model is primarily intended for local/deployed inference.
    repo_id=os.getenv(
        "HF_MODEL_ID",
        "meta-llama/Llama-3.1-8B-Instruct"
    ),
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACE_TOKEN,
    max_new_tokens=1024,
    temperature=0.7,
)

model = ChatHuggingFace(
    llm=llm
)


# =========================================================
# DATABASE
# =========================================================

def get_database_url():

    database_url = (
        os.getenv("MONGODB_URI")
        or os.getenv("DATABASE_URL")
    )

    if not database_url:
        raise ValueError(
            "MONGODB_URI is missing. Add it to .env"
        )

    return database_url


# =========================================================
# MCP CLIENT
# =========================================================

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")


if not TAVILY_API_KEY:
    raise ValueError(
        "TAVILY_API_KEY is missing from .env"
    )

if not NEWS_API_KEY:
    raise ValueError(
        "NEWS_API_KEY is missing from .env"
    )

 
client = MultiServerMCPClient(
    {
        "research": {
            # Run the MCP server as a local subprocess over stdio.
            "transport": "stdio",
            "command": os.sys.executable,
            "args": [
                os.path.join(
                    os.path.dirname(__file__),
                    "MCP_servers.py"
                )
            ],
            "env": {
                "TAVILY_API_KEY": TAVILY_API_KEY,
                "NEWS_API_KEY": NEWS_API_KEY,
                "YOUTUBE_API_KEY": YOUTUBE_API_KEY,
            },
        }
    }
)


# =========================================================
# MCP TOOL INITIALIZATION
# =========================================================

web_search_tool = None
paper_search_tool = None
news_search_tool = None
youtube_search_tool = None


async def initialize_research_tools():

    global web_search_tool
    global paper_search_tool
    global news_search_tool
    global youtube_search_tool

    if (
        web_search_tool is not None
        and paper_search_tool is not None
        and news_search_tool is not None
        and youtube_search_tool is not None
    ):
        return

    tools = await client.get_tools(
        server_name="research"
    )

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    web_search_tool = tools_by_name.get(
        "web_search"
    )

    paper_search_tool = tools_by_name.get(
        "paper_search"
    )

    news_search_tool = tools_by_name.get(
        "news_search"
    )
    youtube_search_tool = tools_by_name.get("youtube_search")

    missing = [
        name
        for name, tool in {
            "web_search": web_search_tool,
            "paper_search": paper_search_tool,
            "news_search": news_search_tool,
            "youtube_search": youtube_search_tool
        }.items()
        if tool is None
    ]

    if missing:
        raise RuntimeError(
            "Research MCP tools not found: "
            + ", ".join(missing)
        )


# =========================================================
# MCP SEARCH FUNCTIONS   -- function calling tools 
# =========================================================

async def web_mcp_search(
    query: str,
    limit: int = 5
):

    await initialize_research_tools()

    return await web_search_tool.ainvoke(
        {
            "query": query,
            "max_results": limit
        }
    )


async def paper_mcp_search(
    query: str,
    limit: int = 5
):

    await initialize_research_tools()

    return await paper_search_tool.ainvoke(
        {
            "query": query,
            "limit": limit
        }
    )


async def news_mcp_search(
    query: str,
    limit: int = 5
):

    await initialize_research_tools()

    return await news_search_tool.ainvoke(
        {
            "query": query,
            "limit": limit
        }
    )

async def youtube_mcp_search(query:str , limit:int=5):
    await initialize_research_tools()
    return await youtube_search_tool.ainvoke(
        {
            "query":query,
            "limit": limit
        }
    )


# =========================================================
# ASYNC HELPER   -- ??
# =========================================================

def run_async(coro):

    try:
        asyncio.get_running_loop()

    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(
        max_workers=1
    ) as executor:

        return executor.submit(
            asyncio.run,
            coro
        ).result()


# =========================================================
# STATE
# =========================================================

class ResearchState(TypedDict):

    messages: Annotated[
        list[AnyMessage],
        operator.add
    ]

    user_query: str

    # Supervisor
    guardrail_allowed: bool
    guardrail_reason: str

    selected_agents: list[str]

    research_constraints: dict[str, Any]

    supervisor_reasoning: str

    # Research results
    web_results: str
    paper_results: str
    news_results: str
    youtube_results: str

    # Analysis
    analysis_results: str

    # HITL
    draft_report: str
    approval_request: str
    approved: bool
    human_feedback: str

    # Final
    final_response: str

    llm_calls: Annotated[int, operator.add]


# =========================================================
# AGENTS
# =========================================================

KNOWN_AGENTS = {
    "web_research_agent",
    "paper_research_agent",
    "news_research_agent",
    "youtube_research_agent",
    "analysis_agent",
    "report_agent",
}


AGENT_ORDER = [
    "web_research_agent",
    "paper_research_agent",
    "news_research_agent",
    "youtube_research_agent",
    "analysis_agent",
    "report_agent",
    
]

####agr urani reseacrrh papers pr research krni hui toh islie 
def empty_constraints():

    return {
        "topic": "",
        "time_range": "",
        "source_preference": "",
        "special_requirements": [],
    }


def _prompt_text(value, max_chars: int = 8000):
    """Serialize and cap tool output before sending it to the LLM."""

    if isinstance(value, (dict, list)):
        text = json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    else:
        text = str(value or "")

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n\n[Additional source content truncated.]"
    )


# =========================================================
# LLM HELPERS
# =========================================================

def _llm_text(
    system_prompt: str,
    user_prompt: str
):

    response = model.invoke(
        [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=user_prompt
            ),
        ]
    )

    return str(response.content)



#llm se aae str response ko json mai convert krega 
def _json_from_llm(text: str):

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Model did not return JSON."
        )

    return json.loads(
        text[start:end + 1]
    )


# =========================================================
# SUPERVISOR
# =========================================================

def supervisor_agent(
    state: ResearchState
):


    query = state["user_query"]

    llm_calls = state.get(
        "llm_calls",
        0
    )

    # -----------------------------------------------------
    # GUARDRAIL
    # -----------------------------------------------------

    guardrail_prompt = f"""
Determine whether the following request is suitable
for an AI research assistant.

Allow requests asking for:
- research
- factual information
- academic information
- technology
- science
- business
- current events
- literature
- comparisons
- reports
- general knowledge

Block harmful or clearly inappropriate requests.
Do not block a normal research topic merely because it could be
more specific. If the request names a subject or asks for research,
allow it.

Return strict JSON:

{{
    "allowed": true,
    "reason": ""
}}

User request:
{query}
"""

    try:

        raw = _llm_text(
            "You are a research input guardrail. "
            "Return strict JSON only.",
            guardrail_prompt
        )

        result = _json_from_llm(raw)

        allowed = bool(
            result.get(
                "allowed",
                True
            )
        )

        reason = str(
            result.get(
                "reason",
                ""
            )
        ).strip()

        llm_calls += 1

    except Exception as exc:

        print(
            f"Guardrail fallback used: "
            f"{type(exc).__name__}: {exc}"
        )

        allowed = True

        reason = (
            "Guardrail fallback allowed the request."
        )

    if not allowed:

        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "research_constraints":
                empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [
                AIMessage(
                    content=reason
                )
            ],
            "llm_calls": llm_calls,
        }

    # -----------------------------------------------------
    # SUPERVISOR
    # -----------------------------------------------------

    supervisor_prompt = f"""
You are the supervisor of a multi-agent
AI research system.

Choose the specialist agents required
for the user's research request.

Available agents:

- web_research_agent:
  searches websites and online sources

- paper_research_agent:
  searches academic and research papers

- news_research_agent:
  searches recent news and reports

- youtube_research_agent:
  searches relevant YouTube videos and transcripts

- analysis_agent:
  analyzes and combines collected information

- report_agent:
  creates the final research response

Always include analysis_agent and report_agent.

Return strict JSON:

{{
    "selected_agents": [
        "web_research_agent",
        "paper_research_agent",
        "news_research_agent",
        "analysis_agent",
        "report_agent"
    ],
    "research_constraints": {{
        "topic": "",
        "time_range": "",
        "source_preference": "",
        "special_requirements": []
    }},
    "reasoning": ""
}}

User request:

{query}
"""

    try:
        print("supervisor agent called : ")
        raw = _llm_text(
            "You route research work to specialist agents. "
            "Return strict JSON only.",
            supervisor_prompt
        )

        parsed = _json_from_llm(raw)

        requested = parsed.get(
            "selected_agents",
            []
        )

        selected_agents = [
            agent
            for agent in AGENT_ORDER
            if agent in requested
        ]

        if "analysis_agent" not in selected_agents:
            selected_agents.append(
                "analysis_agent"
            )

        if "report_agent" not in selected_agents:
            selected_agents.append(
                "report_agent"
            )

        constraints = empty_constraints()

        parsed_constraints = parsed.get(
            "research_constraints",
            {}
        )

        if isinstance(
            parsed_constraints,
            dict
        ):
            constraints.update(
                parsed_constraints
            )

        reasoning = str(
            parsed.get(
                "reasoning",
                ""
            )
        ).strip()

        llm_calls += 1

    except Exception as exc:

        print(
            f"Supervisor fallback used: "
            f"{type(exc).__name__}: {exc}"
        )

        selected_agents = AGENT_ORDER.copy()

        constraints = empty_constraints()

        reasoning = (
            "Supervisor parsing failed. "
            "Full research workflow selected."
        )

    return {

        "guardrail_allowed": True,

        "guardrail_reason": reason,

        "selected_agents": selected_agents,

        "research_constraints": constraints,

        "supervisor_reasoning": reasoning,

        "messages": [
            AIMessage(
                content="Research plan created."
            )
        ],

        "llm_calls": llm_calls,
    }


# =========================================================
# GUARDRAIL BLOCK
# =========================================================

def guardrail_blocked_agent(
    state: ResearchState
):

    reason = (
        state.get("final_response")
        or state.get("guardrail_reason")
        or "Request blocked."
    )

    return {
        "final_response": reason,
        "messages": [
            AIMessage(
                content=reason
            )
        ],
    }


# =========================================================
# WEB RESEARCH AGENT
# =========================================================

def web_research_agent(
    state: ResearchState
):
    print("web search agent called : ")
    query = state["user_query"]

    try:

        result = run_async(
            web_mcp_search(
                query,
                limit=5
            )
        )
        result = _prompt_text(result, 8000)

    except Exception as exc:

        print(
            f"Web search failed: "
            f"{type(exc).__name__}: {exc}"
        )

        result = (
            "Web research unavailable."
        )

    return {

        "web_results": result,

        "messages": [
            AIMessage(
                content="Web research completed."
            )
        ],

        "llm_calls": 1
    }


# =========================================================
# PAPER RESEARCH AGENT
# =========================================================

def paper_research_agent(
    state: ResearchState
):
    print("web search agent called : ")
    query = state["user_query"]

    try:

        result = run_async(
            paper_mcp_search(
                query,
                limit=5
            )
        )
        result = _prompt_text(result, 8000)

    except Exception as exc:

        print(
            f"Paper search failed: "
            f"{type(exc).__name__}: {exc}"
        )

        result = (
            "Paper research unavailable."
        )

    return {

        "paper_results": result,

        "messages": [
            AIMessage(
                content="Paper research completed."
            )
        ],

         "llm_calls": 1
    }


# =========================================================
# NEWS RESEARCH AGENT
# =========================================================

def news_research_agent(
    state: ResearchState
):
    print("news_research_agent called : ")
    query = state["user_query"]

    try:

        result = run_async(
            news_mcp_search(
                query,
                limit=5
            )
        )
        result = _prompt_text(result, 8000)

    except Exception as exc:

        print(
            f"News search failed: "
            f"{type(exc).__name__}: {exc}"
        )

        result = (
            "News research unavailable."
        )

    return {

        "news_results": result,

        "messages": [
            AIMessage(
                content="News research completed."
            )
        ],

        "llm_calls":
            state.get("llm_calls", 0) + 1,
    }


def youtube_research_agent(state: ResearchState):

    print("youtube_research_agent called")

    query = state["user_query"]

    try:
        result = run_async(
            youtube_mcp_search(query, limit=5)
        )

        result = _prompt_text(result, 8000)

    except Exception as exc:

        print(
            f"YouTube search failed: "
            f"{type(exc).__name__}: {exc}"
        )

        result = "YouTube research unavailable."

    return {
        "youtube_results": result,

        "messages": [
            AIMessage(
                content="YouTube research completed."
            )
        ],
 "llm_calls": 1
    }




# =========================================================
# RESEARCH FAN-IN
# =========================================================

def research_complete_agent(state: ResearchState):

    print("All parallel research agents completed.")

    return {
        "messages": [
            AIMessage(
                content="All research sources collected."
            )
        ]
    }




# =========================================================
# ANALYSIS AGENT
# =========================================================

def analysis_agent(
    state: ResearchState
):
   
    prompt = f"""
Analyze the following research information.

USER QUESTION:
{state["user_query"]}

WEB SOURCES:
#site s aae hue output ko chota kr deta hai -->bec tht is very huge data 
{_prompt_text(state.get("web_results", ""), 8000)}

ACADEMIC PAPERS:
{_prompt_text(state.get("paper_results", ""), 8000)}

NEWS SOURCES:
{_prompt_text(state.get("news_results", ""), 8000)}

YOUTUBE VIDEOS:
{_prompt_text(
    state.get("youtube_results", ""),
    8000
)}

Tasks:

1. Identify the most relevant information.
2. Extract important findings.
3. Compare information from different sources.
4. Identify agreements or contradictions.
5. Do not invent facts.
6. Preserve source names and URLs.
7. Clearly distinguish information that is
   supported by sources from information that
   is unavailable.

Create structured research notes.
"""

    try:
        print("analysis agent called : ")
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an AI research "
                        "analysis expert."
                    )
                ),
                HumanMessage(
                    content=prompt
                ),
            ]
        )

        result = response.content

    except Exception as exc:

        print(
            f"Analysis failed: "
            f"{type(exc).__name__}: {exc}"
        )

        result = (
            "Research analysis unavailable."
        )

    return {

        "analysis_results": result,

        "messages": [
            AIMessage(
                content="Research analysis completed."
            )
        ],

        "llm_calls": 1,
    }


# =========================================================
# REPORT AGENT
# =========================================================

def report_agent(
    state: ResearchState
):

    prompt = f"""
Create a clear final research response.

USER QUESTION:
{state["user_query"]}

WEB RESEARCH:
{_prompt_text(state.get("web_results", ""), 10000)}

ACADEMIC PAPERS:
{_prompt_text(state.get("paper_results", ""), 10000)}

NEWS:
{_prompt_text(state.get("news_results", ""), 10000)}

YOUTUBE RESEARCH:
{_prompt_text(
    state.get("youtube_results", ""),
    10000
)}

ANALYSIS:
{_prompt_text(state.get("analysis_results", ""), 10000)}

Create these sections:

1. Research Summary

2. Key Findings

3. Detailed Analysis

4. Academic Research

5. Recent News / Reports

6. Sources

For every important source include:

- Source name
- Title
- Short useful summary
- Original URL

IMPORTANT:

- Do not invent sources.
- Do not invent URLs.
- Use only information provided by
  the research agents.
- Preserve original source links.
- Clearly mention when information
  is unavailable.
- Keep the answer readable.
"""

    try:
        print("report agent called : ")
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a professional "
                        "AI research assistant."
                    )
                ),
                HumanMessage(
                    content=prompt
                ),
            ]
        )

    except Exception as exc:
        print(
            f"Report generation failed: "
            f"{type(exc).__name__}: {exc}"
        )

        response = AIMessage(
            content=(
                "Research report generation "
                "is temporarily unavailable."
            )
        )

    return {

        "draft_report": response.content,

        "approval_request": (
            "Please review the generated "
            "research report. Approve it or "
            "provide revision feedback."
        ),

        "messages": [
            response
        ],

        "llm_calls": 1,
    }


# =========================================================
# HUMAN APPROVAL
# =========================================================

def human_approval_agent(
    state: ResearchState
):


    review = interrupt(
        {
            "question":
                "Do you approve this research report?",

            "draft_report":
                state.get(
                    "draft_report",
                    ""
                ),

            "approval_request":
                state.get(
                    "approval_request",
                    ""
                ),

            "selected_agents":
                state.get(
                    "selected_agents",
                    []
                ),

            "supervisor_reasoning":
                state.get(
                    "supervisor_reasoning",
                    ""
                ),

            "expected_response": {
                "approved": True,
                "feedback":
                    "Optional revision feedback",
            },
        }
    )

    approved = bool(
        review.get(
            "approved",
            False
        )
    )

    feedback = str(
        review.get(
            "feedback",
            ""
        )
    ).strip()

    return {

        "approved": approved,

        "human_feedback": feedback,

        "messages": [
            AIMessage(
                content=(
                    "Human review completed."
                )
            )
        ],
    }


# =========================================================
# FINAL AGENT
# =========================================================

def final_agent(
    state: ResearchState
):


    if state.get("approved"):

        review_instruction = (
            "The user approved the draft. "
            "Preserve its factual content "
            "while polishing the presentation."
        )

    else:

        review_instruction = f"""
The user requested a revision.

Apply this feedback:

{state.get("human_feedback", "")}

Do not remove factual source information.
"""

    prompt = f"""
Create the final research response.

USER QUESTION:
{state["user_query"]}

REVIEW:
{review_instruction}

DRAFT REPORT:
{_prompt_text(state.get("draft_report", ""), 11000)}

WEB SOURCES:
{_prompt_text(state.get("web_results", ""), 10000)}

PAPER SOURCES:
{_prompt_text(state.get("paper_results", ""), 10000)}

NEWS SOURCES:
{_prompt_text(state.get("news_results", ""), 10000)}

ANALYSIS:
{_prompt_text(state.get("analysis_results", ""), 10000)}

YOUTUBE_SOURCES:
{_prompt_text(state.get("youtube_results", ""), 10000)}

Rules:

- Preserve source names.
- Preserve original URLs.
- Do not invent information.
- Do not invent citations.
- Do not remove important source details.
- Clearly state when information is unavailable.
- Make the answer concise but useful.
- The user should be able to click/read
  the original sources.
- And the summary should be detailed 

Return the polished final research response.
"""

    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a professional "
                    "AI research assistant."
                )
            ),
            HumanMessage(
                content=prompt
            ),
        ]
    )

    return {

        "final_response":
            response.content,

        "messages": [
            response
        ],

        "llm_calls":
            state.get(
                "llm_calls",
                0
            ) + 1,
    }


# =========================================================
# ROUTING
# =========================================================

ROUTE_MAP = {

    "guardrail_blocked":
        "guardrail_blocked",

    "web_research_agent":
        "web_research_agent",

    "paper_research_agent":
        "paper_research_agent",

    "news_research_agent":
        "news_research_agent",

    "youtube_research_agent":
    "youtube_research_agent",


    "research_complete":
        "research_complete",

    "analysis_agent":
        "analysis_agent",

    "report_agent":
        "report_agent",
}



##parallel agents working 
def route_from_supervisor(state: ResearchState):

    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = state.get("selected_agents", [])

    research_agents = [
        agent
        for agent in [
            "web_research_agent",
            "paper_research_agent",
            "news_research_agent",
            "youtube_research_agent",
        ]
        if agent in selected
    ]

    # If supervisor selected no research agent,
    # run all research agents.
    if not research_agents:
        research_agents = [
            "web_research_agent",
            "paper_research_agent",
            "news_research_agent",
            "youtube_research_agent",
        ]

    return [
        #send 
        Send(agent, state)
        for agent in research_agents
    ]



# =========================================================
# BUILD GRAPH
# =========================================================

graph = StateGraph(
    ResearchState
)


graph.add_node(
    "supervisor",
    supervisor_agent
)

graph.add_node(
    "guardrail_blocked",
    guardrail_blocked_agent
)

graph.add_node(
    "web_research_agent",
    web_research_agent
)

graph.add_node(
    "paper_research_agent",
    paper_research_agent
)

graph.add_node(
    "news_research_agent",
    news_research_agent
)

graph.add_node(
    "youtube_research_agent",
    youtube_research_agent
)

graph.add_node(
    "research_complete",
    research_complete_agent
)


graph.add_node(
    "analysis_agent",
    analysis_agent
)

graph.add_node(
    "report_agent",
    report_agent
)

graph.add_node(
    "human_approval",
    human_approval_agent
)

graph.add_node(
    "final_agent",
    final_agent
)


graph.add_edge(
    START,
    "supervisor"
)


graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    ROUTE_MAP
)


# =========================================================
# PARALLEL RESEARCH → FAN-IN
# =========================================================

graph.add_edge(
    "web_research_agent",
    "research_complete"
)

graph.add_edge(
    "paper_research_agent",
    "research_complete"
)

graph.add_edge(
    "news_research_agent",
    "research_complete"
)

graph.add_edge(
    "youtube_research_agent",
    "research_complete"
)

graph.add_edge(
    "research_complete",
    "analysis_agent"
)

graph.add_edge(
    "analysis_agent",
    "report_agent"
)


graph.add_edge(
    "report_agent",
    "human_approval"
)


graph.add_edge(
    "human_approval",
    "final_agent"
)


graph.add_edge(
    "final_agent",
    END
)


graph.add_edge(
    "guardrail_blocked",
    END
)


# =========================================================
# MONGODB CHECKPOINTER
# =========================================================

research_graph = None
memory_checkpointer = MemorySaver()


def get_research_graph():

    global research_graph

    if research_graph is None:

        try:

            mongodb_uri = get_database_url()

            mongodb_client = MongoClient(
                mongodb_uri,
                connect=False,
                serverSelectionTimeoutMS=int(
                    os.getenv(
                        "MONGODB_SERVER_SELECTION_TIMEOUT_MS",
                        "5000"
                    )
                ),
            )

            checkpointer = MongoDBSaver(
                client=mongodb_client,
                db_name=os.getenv(
                    "MONGODB_DB_NAME",
                    "research_agent_db"
                ),
            )

            research_graph = graph.compile(
                checkpointer=checkpointer
            )

        except PyMongoError as error:

            if (
                os.getenv(
                    "MONGODB_REQUIRED",
                    "false"
                ).lower()
                == "true"
            ):

                raise RuntimeError(
                    "MongoDB connection failed."
                ) from error

            print(
                "MongoDB unavailable. "
                "Using in-memory graph."
            )

            # A checkpointer is still required for interrupt/resume even
            # when MongoDB is unavailable.
            research_graph = graph.compile(
                checkpointer=memory_checkpointer
            )

    return research_graph


# =========================================================
# INTERRUPT SERIALIZATION
# =========================================================

def _interrupt_payload(
    result: dict[str, Any]
):

    interrupts = result.get(
        "__interrupt__",
        []
    )

    if not interrupts:
        return None

    first = interrupts[0]

    payload = getattr(
        first,
        "value",
        first
    )

    return (
        payload
        if isinstance(payload, dict)
        else {"value": payload}
    )


# =========================================================
# SERIALIZE RESULT
# =========================================================

def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
):

    messages = result.get(
        "messages",
        []
    )

    last_message = (
        messages[-1].content
        if messages
        else ""
    )

    answer = (
        result.get("final_response")
        or last_message
    )

    interrupt_payload = (
        _interrupt_payload(result)
    )

    if interrupt_payload:

        answer = (
            interrupt_payload.get(
                "draft_report"
            )
            or result.get(
                "draft_report",
                ""
            )
        )

    return {

        "thread_id":
            thread_id,

        "answer":
            answer,

        "requires_approval":
            interrupt_payload is not None,

        "approval_request": (
            interrupt_payload.get(
                "approval_request",
                ""
            )
            if interrupt_payload
            else result.get(
                "approval_request",
                ""
            )
        ),

        "web_results":
            result.get(
                "web_results",
                ""
            ),

        "paper_results":
            result.get(
                "paper_results",
                ""
            ),

        "news_results":
            result.get(
                "news_results",
                ""
            ),

        "analysis_results":
            result.get(
                "analysis_results",
                ""
            ),

        "draft_report": (
            interrupt_payload.get(
                "draft_report",
                ""
            )
            if interrupt_payload
            else result.get(
                "draft_report",
                ""
            )
        ),

        "selected_agents":
            result.get(
                "selected_agents",
                []
            ),

        "research_constraints":
            result.get(
                "research_constraints",
                {}
            ),

        "supervisor_reasoning":
            result.get(
                "supervisor_reasoning",
                ""
            ),

        "guardrail_allowed":
            result.get(
                "guardrail_allowed",
                True
            ),

        "guardrail_reason":
            result.get(
                "guardrail_reason",
                ""
            ),

        "approved":
            result.get(
                "approved"
            ),

        "human_feedback":
            result.get(
                "human_feedback",
                ""
            ),

        "llm_calls":
            result.get(
                "llm_calls",
                0
            ),
    }


# =========================================================
# FASTAPI - START RESEARCH
# =========================================================

def run_research_agent(
    user_input: str,
    thread_id: str | None = None
):

    if not thread_id:

        thread_id = (
            f"user_{uuid.uuid4().hex}"
        )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    research_graph = (
        get_research_graph()
    )

    result = research_graph.invoke(
        {

            "messages": [
                HumanMessage(
                    content=user_input
                )
            ],

            "user_query":
                user_input,

            "guardrail_allowed":
                True,

            "guardrail_reason":
                "",

            "selected_agents":
                [],

            "research_constraints":
                empty_constraints(),

            "supervisor_reasoning":
                "",

            "web_results":
                "",

            "paper_results":
                "",

            "news_results":
                "",
            
            "youtube_results": "",

            "analysis_results":
                "",

            "draft_report":
                "",

            "approval_request":
                "",

            "approved":
                False,

            "human_feedback":
                "",

            "final_response":
                "",

            "llm_calls":
                0,
        },

        config=config,
    )

    return _serialize_result(
        result,
        thread_id
    )


# =========================================================
# FASTAPI - RESUME AFTER HITL
# =========================================================

def resume_research_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):

    if not thread_id:

        raise ValueError(
            "thread_id is required."
        )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    research_graph = (
        get_research_graph()
    )

    checkpoint = research_graph.get_state(config)

    if not checkpoint or "user_query" not in checkpoint.values:
        raise ValueError(
            "Research thread was not found or has expired. "
            "Start a new research request before approving."
        )

    result = research_graph.invoke(
        Command(
            resume={
                "approved":
                    approved,

                "feedback":
                    feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(
        result,
        thread_id
    )
