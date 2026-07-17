from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from graph.state import merge_agent_results
from tools.supplier_tools import get_suppliers_by_ids, get_supplier_details, search_suppliers


SUPPLIER_PROMPT = """
You are a Supplier Assistant for a wholesale marketplace.

Classify the request before selecting a tool.
Use get_suppliers_by_ids when supplier recommendations come from product results.
Use search_suppliers when the user asks to search, compare, or recommend suppliers.
Use get_supplier_details only for one specific supplier.
Do not search suppliers by business name through a product tool.
"""


DETAIL_KEYWORDS = ("details", "detail", "contact", "address", "phone", "website", "gst", "pan", "profile")


def _get_user_query(state: dict[str, Any]) -> str:
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            return message.content or ""
    return ""


def _extract_seller_ids(product_result: dict[str, Any]) -> list[str]:
    seller_ids: list[str] = []
    seen: set[str] = set()

    def add_seller_id(value: Any):
        if not value:
            return
        seller_id = str(value).strip()
        if not seller_id or seller_id in seen:
            return
        seen.add(seller_id)
        seller_ids.append(seller_id)

    if not isinstance(product_result, dict):
        return seller_ids

    product_search = product_result.get("product_search")
    if isinstance(product_search, dict):
        for product in product_search.get("products", []):
            if isinstance(product, dict):
                add_seller_id(product.get("seller_id"))

    for product in product_result.get("products", []):
        if isinstance(product, dict):
            add_seller_id(product.get("seller_id"))

    result_payload = product_result.get("result")
    if isinstance(result_payload, dict):
        for product in result_payload.get("products", []):
            if isinstance(product, dict):
                add_seller_id(product.get("seller_id"))

    return seller_ids


def _sort_by_hint(user_query: str) -> str:
    normalized = user_query.lower()
    if "review" in normalized or "reviews" in normalized:
        return "reviews"
    if "experience" in normalized or "experienced" in normalized:
        return "experience"
    if "verified" in normalized or "trusted" in normalized or "authentic" in normalized:
        return "verified"
    return "rating"


def _detect_supplier_intent(user_query: str, seller_ids: list[str]) -> dict[str, str]:
    normalized = (user_query or "").strip().lower()
    if seller_ids:
        return {"intent": "supplier_recommendation", "tool": "get_suppliers_by_ids", "reason": "Supplier results can be derived from product matches."}

    if any(keyword in normalized for keyword in DETAIL_KEYWORDS) and re.search(r"\b(supplier|seller|vendor)\b", normalized):
        return {"intent": "supplier_details", "tool": "get_supplier_details", "reason": "The user asked for one specific supplier's details."}

    return {"intent": "supplier_search", "tool": "search_suppliers", "reason": "The user asked to search or compare suppliers."}


def _parse_search_filters(user_query: str) -> dict[str, Any]:
    normalized = user_query.lower()
    filters: dict[str, Any] = {}
    for field in ("city", "state", "country"):
        match = re.search(rf"\b{field}\s+([a-z][a-z\s]+)", normalized)
        if match:
            filters[field] = match.group(1).strip().split(" ")[0]
    if "verified" in normalized:
        filters["verification_status"] = "verified"
    if "onboarding" in normalized or "completed" in normalized:
        filters["onboarding_completed"] = True
    return filters


def supplier_agent(state):
    current_task = state.get("execution_plan", [])[state.get("current_step", 0)]
    user_query = _get_user_query(state)
    product_result = state.get("agent_results", {}).get("product", {})
    seller_ids = _extract_seller_ids(product_result)
    sort_by = _sort_by_hint(user_query)
    intent_result = _detect_supplier_intent(user_query, seller_ids)

    last_message = state.get("messages", [])[-1] if state.get("messages") else None
    if isinstance(last_message, ToolMessage):
        tool_result = last_message.content
        return {
            "messages": [HumanMessage(content="Supplier task completed.")],
            "agent_results": merge_agent_results(
                state.get("agent_results", {}),
                "supplier",
                {
                    "task": current_task.get("goal", "Supplier lookup"),
                    "result": tool_result,
                    "seller_ids": seller_ids,
                    "detected_intent": intent_result["intent"],
                    "selected_tool": intent_result["tool"],
                    "reason": intent_result["reason"],
                },
            ),
            "current_step": state["current_step"] + 1,
        }

    if intent_result["tool"] == "get_suppliers_by_ids":
        supplier_result = get_suppliers_by_ids.invoke({"seller_ids": seller_ids, "sort_by": sort_by})
        return {
            "messages": [SystemMessage(content=SUPPLIER_PROMPT), HumanMessage(content="Using supplier IDs from product results.")],
            "agent_results": merge_agent_results(
                state.get("agent_results", {}),
                "supplier",
                {
                    "task": current_task.get("goal", "Supplier lookup"),
                    "seller_ids": seller_ids,
                    "supplier_result": supplier_result,
                    "detected_intent": intent_result["intent"],
                    "selected_tool": intent_result["tool"],
                    "reason": intent_result["reason"],
                },
            ),
            "current_step": state["current_step"] + 1,
        }

    if intent_result["tool"] == "get_supplier_details":
        supplier_match = re.search(r"\b(?:supplier|seller|vendor)\s+([a-z0-9_-]+)\b", user_query.lower())
        supplier_id = supplier_match.group(1) if supplier_match else ""
        supplier_result = get_supplier_details.invoke({"supplier_id": supplier_id}) if supplier_id else {"success": False, "message": "Supplier ID is required.", "supplier": None}
    else:
        supplier_result = search_suppliers.invoke(_parse_search_filters(user_query) | {"sort_by": sort_by})

    return {
        "messages": [HumanMessage(content="Supplier search completed.")],
        "agent_results": merge_agent_results(
            state.get("agent_results", {}),
            "supplier",
            {
                "task": current_task.get("goal", "Supplier lookup"),
                "supplier_result": supplier_result,
                "detected_intent": intent_result["intent"],
                "selected_tool": intent_result["tool"],
                "reason": intent_result["reason"],
            },
        ),
        "current_step": state["current_step"] + 1,
    }
