# tool_wrapper.py
from ddgs import DDGS
from typing import List, Dict, Any

class SearchTool:
    """
    A wrapper for DuckDuckGo Search to be used by the LLM-as-a-Judge.
    """
    def __init__(self, max_results=3):
        self.ddgs = DDGS()
        self.max_results = max_results
        print("DuckDuckGo SearchTool initialized.")

    def search(self, query: str) -> str:
        """
        Performs a search and returns a formatted string of results.
        """
        print(f"--- SearchTool running query: {query} ---")
        try:
            results = self.ddgs.text(query, max_results=self.max_results)
            if not results:
                return "No results found."
            
            # Format results into a simple string for the LLM
            formatted_results = "\n".join(
                [f"Snippet {i+1}: {r['body']}" for i, r in enumerate(results)]
            )
            return formatted_results
        except Exception as e:
            print(f"Error during DuckDuckGo search: {e}")
            return f"Error: Could not perform search. {e}"