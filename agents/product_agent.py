from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import merge_agent_results
from tools.product_tools import (
    format_search_response,
    search_products,
    get_total_product_count,
    get_category_statistics,
    get_subcategory_statistics,
    get_inventory_summary,
)


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


def _detect_analytics_intent(user_query: str, goal: str) -> str | None:
    query_norm = (user_query or "").strip().lower()
    goal_norm = (goal or "").strip().lower()

    # Check subcategories list / stats
    if (
        "retrieve all subcategories" in goal_norm
        or "retrieve subcategories" in goal_norm
        or "get subcategories" in goal_norm
        or "subcategory statistics" in goal_norm
        or "list subcategories" in goal_norm
        or "show subcategories" in goal_norm
        or "show me all subcategories" in query_norm
        or "list all subcategories" in query_norm
        or "subcategory statistics" in query_norm
        or "available subcategories" in query_norm
        or "subcategories are available" in query_norm
        or "what subcategories" in query_norm
        or query_norm in ["subcategories", "all subcategories", "show subcategories", "list subcategories", "what subcategories"]
    ):
        return "get_subcategory_statistics"

    # Check categories list / stats
    if (
        "retrieve all product categories" in goal_norm
        or "retrieve product categories" in goal_norm
        or "get categories" in goal_norm
        or "category statistics" in goal_norm
        or "list categories" in goal_norm
        or "show categories" in goal_norm
        or "show me all categories" in query_norm
        or "list all categories" in query_norm
        or "category statistics" in query_norm
        or "available categories" in query_norm
        or "categories are available" in query_norm
        or "what categories" in query_norm
        or query_norm in ["categories", "all categories", "show categories", "list categories", "what categories"]
    ):
        return "get_category_statistics"

    # Check inventory summary
    if (
        "inventory summary" in goal_norm
        or "inventory statistics" in goal_norm
        or "inventory summary" in query_norm
        or "inventory statistics" in query_norm
        or query_norm in ["inventory", "inventory summary", "inventory statistics"]
    ):
        return "get_inventory_summary"

    # Check total product count
    if (
        "total product count" in goal_norm
        or "product count" in goal_norm
        or "count products" in goal_norm
        or "how many products" in goal_norm
        or "total products" in goal_norm
        or "total product count" in query_norm
        or "how many products" in query_norm
        or "number of products" in query_norm
        or "product count" in query_norm
        or "total products" in query_norm
        or "total items" in query_norm
    ):
        return "get_total_product_count"

    # Fallback checks with indicators to catch variations (excluding general search terms like show, list, get, all)
    has_indicator = any(ind in query_norm or ind in goal_norm for ind in ["how many", "total", "count", "statistics", "summary", "exist"])
    if has_indicator:
        if "subcategory" in query_norm or "subcategories" in query_norm or "subcategory" in goal_norm or "subcategories" in goal_norm:
            return "get_subcategory_statistics"
        if "category" in query_norm or "categories" in query_norm or "category" in goal_norm or "categories" in goal_norm:
            return "get_category_statistics"
        if "inventory" in query_norm or "inventory" in goal_norm:
            return "get_inventory_summary"
        if "product" in query_norm or "products" in query_norm or "item" in query_norm or "items" in query_norm or "product" in goal_norm or "products" in goal_norm:
            return "get_total_product_count"

    return None


def _format_analytics_response(intent: str, result: dict[str, Any]) -> str:
    if not result or not result.get("success"):
        return result.get("message") or "I couldn't retrieve the requested inventory information right now."

    if intent == "get_total_product_count":
        return f"Total products available: {result.get('total_products', 0)}"

    if intent == "get_inventory_summary":
        return (
            "Inventory Summary\n\n"
            f"• Total Categories: {result.get('total_categories', 0)}\n"
            f"• Total Subcategories: {result.get('total_subcategories', 0)}\n"
            f"• Total Products: {result.get('total_products', 0)}"
        )

    if intent == "get_subcategory_statistics":
        subcategories = result.get("subcategories") or []
        if not subcategories:
            return "I couldn't find any subcategory statistics right now."
        lines = ["Subcategory Statistics", ""]
        for item in subcategories:
            lines.append(f"• {item.get('subcategory', 'Unknown')} – {item.get('product_count', 0)} products")
        lines.append("")
        lines.append(f"Total Products: {result.get('total_products', 0)}")
        return "\n".join(lines)

    # get_category_statistics is the default / categories list
    categories = result.get("categories") or []
    if not categories:
        return "I couldn't find any category statistics right now."

    lines = ["Category Statistics", ""]
    for item in categories:
        lines.append(f"• {item.get('category', 'Unknown')} – {item.get('product_count', 0)} products")
    lines.append("")
    lines.append(f"Total Products: {result.get('total_products', 0)}")
    return "\n".join(lines)


def product_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Intelligent Product Agent node in LangGraph.
    - Inspects the latest user message
    - Checks for analytics / category statistics intent
    - Calls appropriate tool or fallback to search_products(query)
    - Formats the response
    - Stores result inside state["agent_results"]["product"]
    - Updates state["final_response"]
    """
    user_query = _get_user_query(state)
    logger.info(f"Product Agent received query: {user_query}")

    # Extract goal from current task in execution plan
    current_step = state.get("current_step", 0)
    execution_plan = state.get("execution_plan", [])
    current_task = execution_plan[current_step] if current_step < len(execution_plan) else {}
    goal = current_task.get("goal", "")

    # Detect analytics intent
    analytics_intent = _detect_analytics_intent(user_query, goal)

    if analytics_intent:
        logger.info(f"Analytics intent detected: {analytics_intent}")
        if analytics_intent == "get_category_statistics":
            tool_result = get_category_statistics.invoke({})
        elif analytics_intent == "get_subcategory_statistics":
            tool_result = get_subcategory_statistics.invoke({})
        elif analytics_intent == "get_inventory_summary":
            tool_result = get_inventory_summary.invoke({})
        else:
            tool_result = get_total_product_count.invoke({})

        search_result = {
            "success": tool_result.get("success", False),
            "result_type": "analytics",
            "selected_tool": analytics_intent,
            "analytics_result": tool_result,
            "original_query": user_query,
        }
        response_text = _format_analytics_response(analytics_intent, tool_result)
    else:
        # Fallback to general search
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
