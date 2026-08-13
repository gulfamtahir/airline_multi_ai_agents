from tavily import TavilyClient
import os
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


@tool
def tavily_search(query: str) -> str:
    """Search the web for hotel recommendations, prices, and reviews. Use this
    whenever the user's request involves hotels or accommodation."""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query=query,
        max_results=5
    )

    results = []

    for i, r in enumerate(response["results"], 1):
        title   = r.get("title", "Unknown")
        url     = r.get("url", "")
        snippet = r.get("content", "").strip()
        # Keep only the first 300 characters to avoid wall-of-text
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        results.append(f"{i}. **{title}**\n   {url}\n   {snippet}")

    return "\n\n".join(results)