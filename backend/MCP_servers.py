from mcp.server.fastmcp import FastMCP
import requests
import os
import sys
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("Researchmcpserver")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()
print("news api key" , NEWS_API_KEY)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()

from tavily import TavilyClient


try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    # YouTube search remains available without transcript support.
    YouTubeTranscriptApi = None









# --------------------------------------------------
# 1. WEB SEARCH
# --------------------------------------------------

@mcp.tool()
def web_search(query: str, max_results: int = 10):

    if not TAVILY_API_KEY:
        return {
            "error": "TAVILY_API_KEY is missing."
        }

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)

        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            # Full raw pages can exceed the LLM context window.
            # The regular Tavily content is sufficient for research prompts.
            include_raw_content=False
        )
        print("tavily response", response)

        results = []

        for item in response.get("results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "raw_content": item.get("raw_content")
            })

        return {
            "query": query,
            "answer": response.get("answer"),
            "results": results
        }

    except Exception as e:
        return {
            "error": f"Web search failed: {type(e).__name__}"
        }


# --------------------------------------------------
# 2. PAPER SEARCH - OPENALEX
# --------------------------------------------------

@mcp.tool()

def paper_search(query: str, limit: int = 5):

    url = "https://api.openalex.org/works"

    params = {
        "search": query,
        "per-page": limit,
        "select": (
            "id,"
            "display_name,"
            "publication_year,"
            "authorships,"
            "abstract_inverted_index,"
            "doi,"
            "primary_location,"
            "cited_by_count"
        )
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        if not response.ok:
            return (
                "Paper search unavailable: "
                f"HTTP {response.status_code} - "
                f"{response.text[:300]}"
            )

        data = response.json()

    except requests.exceptions.RequestException as error:
        status = getattr(error.response, "status_code", "unknown")
        return []

    except Exception as error:
        return []

    results = []

    for paper in data.get("results", []):

        title = paper.get("display_name") or "Unknown title"

        year = paper.get("publication_year") or "Unknown year"

        # Authors
        authors = paper.get("authorships", [])

        author_names = ", ".join(
            author.get("author", {}).get("display_name", "")
            for author in authors
            if author.get("author")
        )

        # DOI
        doi = paper.get("doi") or ""

        # Paper URL
        primary_location = paper.get("primary_location") or {}

        paper_url = (
            primary_location.get("landing_page_url")
            or doi
            or ""
        )

        # Open access PDF
        pdf_url = ""

        if primary_location.get("is_oa"):
            pdf_url = (
                primary_location
                .get("pdf_url")
                or ""
            )

        # Citation count
        citation_count = paper.get("cited_by_count", 0)

        results.append({
            "title": title,
            "authors": author_names,
            "year": year,
            "paper_url": paper_url,
            "pdf_url": pdf_url,
            "citation_count": citation_count,
        })

    if not results:
        return []

    return results


# --------------------------------------------------
# 3. NEWS SEARCH
# --------------------------------------------------

@mcp.tool()
def news_search(query: str, limit: int = 3):


    print("news api key" , NEWS_API_KEY)

    if not NEWS_API_KEY:
        return "News search unavailable: NEWS_API_KEY is missing."


    print("news api key" , NEWS_API_KEY)

    url = "https://newsapi.org/v2/everything"

   #taki api ko btaya ja ske ki kitna or kya chahiye
    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": limit
    }

    headers = {
        "X-Api-Key": NEWS_API_KEY
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=30
        )

        if not response.ok:
            return (
                "News search unavailable: "
                f"HTTP {response.status_code} - "
                f"{response.text[:300]}"
            )

        data = response.json()

    except requests.exceptions.RequestException as error:
        status = getattr(error.response, "status_code", "unknown")
        return (
            "News search unavailable: "
            f"{type(error).__name__} (HTTP {status})."
        )

    except Exception as error:
        return f"News search unavailable: {type(error).__name__}."

    results = []

    for i, article in enumerate(
        data.get("articles", []), 1
    ):

        source = (
            article.get("source", {}).get("name")
            or "Unknown source"
        )

        title = (
            article.get("title")
            or "Unknown title"
        )

        description = (
            article.get("description")
            or "No description available"
        )

        url = article.get("url") or ""

        published_at = (
            article.get("publishedAt")
            or "Unknown date"
        )

        if len(description) > 300:
            description = (
                description[:300]
                .rsplit(" ", 1)[0]
                + "..."
            )

        results.append(
            f"{i}. **{title}**\n"
            f"Source: {source}\n"
            f"Published: {published_at}\n"
            f"URL: {url}\n"
            f"Summary: {description}"
        )

    if not results:
        return "No relevant news articles found."

    return "\n\n".join(results)





@mcp.tool()
def youtube_search(query: str, limit: int = 5):

    # 1. Search YouTube
    url = "https://www.googleapis.com/youtube/v3/search"

    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": limit,
        "order": "relevance",
        "key": YOUTUBE_API_KEY,
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    videos = response.json().get("items", [])

    results = []

    # 2. Get transcript of each video
    for video in videos:

        video_id = video["id"]["videoId"]

        title = video["snippet"]["title"]
        channel = video["snippet"]["channelTitle"]

        try:

            if YouTubeTranscriptApi is None:
                transcript_text = ""

            else:
                transcript = YouTubeTranscriptApi().fetch(
                    video_id
                )

                transcript_text = " ".join(
                    snippet.text
                    for snippet in transcript
                )

        except Exception as exc:

            print(
                f"Transcript unavailable for "
                f"{video_id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

            transcript_text = ""

        results.append({
            "title": title,
            "channel": channel,
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "transcript": transcript_text
        })

    return results


# --------------------------------------------------
# RUN MCP SERVER
# --------------------------------------------------

if __name__ == "__main__":
    mcp.run()
