

import asyncio
from mcp_client import get_all_tools
from backend import run_research_agent

# ans=None

# To run asnych functions 
if __name__ == "__main__":
    # q= "give me latest new about AI"
    asyncio.run(run_research_agent("topic is langchain" ))
    # asyncio.run(tavily_mcp_search(q))
    

