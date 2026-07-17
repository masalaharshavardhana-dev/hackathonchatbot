from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import merge_agent_results
from tools.product_tools import format_search_response, search_products


logger = logging.getLogger(__name__)


PRODUCT_AGENT_PROMPT = """
You are the Product Assistant.
"""


def _get_user_query(state: dict[str, Any]) -> str:
    # Get the latest human message
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message.content or ""
    return ""


def product_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Intelligent Product Agent node in LangGraph.
    - Inspects the latest user message
    - Calls search_products(query)
    - Formats the response
    - Stores result inside state["agent_results"]["product_agent"]
    - Updates state["final_response"]
    """
    user_query = _get_user_query(state)
    logger.info(f"Product Agent received query: {user_query}")

    # Call the search master function tool using invoke
    search_result = search_products.invoke({"user_query": user_query})

    # Add selected_products and original_query for supervisor_summary compatibility
    if isinstance(search_result, dict):
        if "products" in search_result:
            search_result["selected_products"] = search_result["products"]
        search_result["original_query"] = user_query

    # Format response
    response_text = format_search_response(search_result)

    # Merge results into state["agent_results"]["product"]
    new_agent_results = merge_agent_results(
        state.get("agent_results", {}),
        "product",
        search_result,
    )

    return {
        "messages": [
            SystemMessage(content=PRODUCT_AGENT_PROMPT),
            HumanMessage(content=response_text),
        ],
        "agent_results": new_agent_results,
        "final_response": response_text,
        "current_step": state["current_step"] + 1,
    }
