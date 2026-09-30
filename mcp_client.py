# ==========================================
# Research MCP Client
# ==========================================

import os
import asyncio
import certifi

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint
)

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

load_dotenv()


# ==========================================
# LLM
# ==========================================

HUGGINGFACE_TOKEN = os.getenv(
    "HUGGINGFACEHUB_ACCESS_TOKEN"
)

if not HUGGINGFACE_TOKEN:
    raise ValueError(
        "HUGGINGFACEHUB_ACCESS_TOKEN is missing from .env"
    )


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACE_TOKEN,
)

model = ChatHuggingFace(
    llm=llm
)


# ==========================================
# API KEYS
# ==========================================

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError(
        "TAVILY_API_KEY is missing from .env"
    )

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not NEWS_API_KEY:
    raise ValueError(
        "NEWS_API_KEY is missing from .env"
    )



# yha client ko server se connect krre hai --> yha sare toolleke unko func m cll krre hai 

# ==========================================
# MCP CLIENT
# ==========================================

client = MultiServerMCPClient(
    {

        # ----------------------------------
        # Custom Research MCP Server
        # ----------------------------------

        "research": {
            "transport": "streamable_http",
            "url": os.getenv(
                "MCP_SERVER_URL",
                "http://127.0.0.1:8001/mcp"
            ),
        },

    }
)


# ==========================================
# TEST ALL TOOLS
# ==========================================

async def get_all_tools():

    server_configs = {

        "research": {
            "transport": "streamable_http",
            "url": os.getenv(
                "MCP_SERVER_URL",
                "http://127.0.0.1:8001/mcp"
            ),
        }

    }

    all_tools = []

    for server_name, config in server_configs.items():

        try:

            server_client = MultiServerMCPClient(
                {
                    server_name: config
                }
            )

            tools = await server_client.get_tools()

            all_tools.append(tools)

            print(f"\n{server_name} tools:")

            for tool in tools:
                print(" -", tool.name)

        except Exception as error:

            print(
                f"{server_name} discovery failed: "
                f"{type(error).__name__}"
            )

    return all_tools


# ==========================================
# RESEARCH TOOLS
# ==========================================

web_search_tool = None
paper_search_tool = None
news_search_tool = None


async def initialize_research_tools():

    global web_search_tool
    global paper_search_tool
    global news_search_tool

    if (
        web_search_tool is not None
        and paper_search_tool is not None
        and news_search_tool is not None
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

    print("web search tool" , web_search_tool)

    paper_search_tool = tools_by_name.get(
        "paper_search"
    )
    print("web search tool" , paper_search_tool)
    

    news_search_tool = tools_by_name.get(
        "news_search"
    )

    print("web search tool" , news_search_tool)
    

    missing_tools = [
        name
        for name, tool in {
            "web_search": web_search_tool,
            "paper_search": paper_search_tool,
            "news_search": news_search_tool,
        }.items()
        if tool is None
    ]

    if missing_tools:

        raise RuntimeError(
            "Research MCP tools not found: "
            + ", ".join(missing_tools)
        )


# ==========================================
# WEB SEARCH
# ==========================================

async def web_mcp_search(
    query: str,
    limit: int = 5
):

    await initialize_research_tools()

    result = await web_search_tool.ainvoke(
        {
            "query": query,
            "limit": limit
        }
    )

    return result


# ==========================================
# PAPER SEARCH
# ==========================================

async def paper_mcp_search(
    query: str,
    limit: int = 5
):

    await initialize_research_tools()

    result = await paper_search_tool.ainvoke(
        {
            "query": query,
            "limit": limit
        }
    )

    return result


# ==========================================
# NEWS SEARCH
# ==========================================

async def news_mcp_search(
    query: str,
    limit: int = 5
):

    await initialize_research_tools()

    result = await news_search_tool.ainvoke(
        {
            "query": query,
            "limit": limit
        }
    )

    return result


# ==========================================
# TEST
# ==========================================

async def test_research_tools():

    print("\nWEB RESULTS")
    print(
        await web_mcp_search(
            "latest developments in artificial intelligence"
        )
    )

    print("\nPAPER RESULTS")
    print(
        await paper_mcp_search(
            "large language models"
        )
    )

    print("\nNEWS RESULTS")
    print(
        await news_mcp_search(
            "artificial intelligence"
        )
    )


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    asyncio.run(
        test_research_tools()
    )
