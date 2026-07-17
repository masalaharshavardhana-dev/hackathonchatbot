from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage

from graph.state import merge_agent_results
from tools.order_tools import _get_order_details_impl, _search_orders_impl


ORDER_PROMPT = """
You are an Order Assistant for a B2B marketplace.

Rules:
- Never ask the user for a buyer ID.
- Always use the authenticated buyer ID from backend state.
- Use search_orders for order history, recent orders, delivered orders, pending orders, and cancelled orders.
- Use get_order_details for a specific order number, tracking request, or invoice/order summary request.
- Always keep the search scoped to the current buyer only.
"""


STATUS_KEYWORDS = {
    "delivered": "Delivered",
    "pending": "Pending",
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
    "processing": "Processing",
    "shipped": "Shipped",
    "paid": "Paid",
    "unpaid": "Unpaid",
    "refunded": "Refunded",
}


def _get_user_query(state: dict[str, Any]) -> str:
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            return message.content or ""
    return ""


def _extract_order_number(user_query: str) -> str | None:
    match = re.search(r"\b(?:ORD\d+|\d{4,})\b", user_query.upper())
    if match:
        return match.group(0)
    return None


def _extract_limit(user_query: str) -> int:
    match = re.search(r"\b(?:last|recent|latest)\s+(\d+)\b", user_query.lower())
    if match:
        try:
            return max(1, min(int(match.group(1)), 50))
        except (TypeError, ValueError):
            return 10
    return 10


def _extract_status(user_query: str) -> str | None:
    normalized = user_query.lower()
    for keyword, status in STATUS_KEYWORDS.items():
        if keyword in normalized:
            return status
    return None


def _extract_sort_by(user_query: str) -> str:
    normalized = user_query.lower()
    if "oldest" in normalized:
        return "oldest"
    if "highest" in normalized or "amount high" in normalized or "expensive" in normalized:
        return "amount_high"
    if "lowest" in normalized or "amount low" in normalized or "cheapest" in normalized:
        return "amount_low"
    return "recent"


def _should_fetch_order_details(user_query: str, order_number: str | None) -> bool:
    if not order_number:
        return False
    normalized = user_query.lower()
    return any(
        keyword in normalized
        for keyword in [
            "track",
            "tracking",
            "details",
            "detail",
            "summary",
            "invoice",
            "shipping",
            "delivery",
        ]
    )


def orders_agent(state):
    current_task = state.get("execution_plan", [])[state.get("current_step", 0)]
    user_query = _get_user_query(state)
    buyer_id = state.get("buyer_id")

    if not buyer_id:
        response = {
            "success": False,
            "message": "Unauthorized access. Buyer ID not found in state.",
            "orders": [],
        }
        return {
            "messages": [HumanMessage(content="I couldn't access your order history right now.")],
            "agent_results": merge_agent_results(
                state.get("agent_results", {}),
                "orders",
                response,
            ),
            "current_step": state["current_step"] + 1,
        }

    order_number = _extract_order_number(user_query)
    if _should_fetch_order_details(user_query, order_number):
        order_result = _get_order_details_impl(buyer_id=buyer_id, order_number=order_number or "")
    else:
        order_result = _search_orders_impl(
            buyer_id=buyer_id,
            status=_extract_status(user_query),
            limit=_extract_limit(user_query),
            sort_by=_extract_sort_by(user_query),
        )

    return {
        "messages": [HumanMessage(content="Order search completed.")],
        "agent_results": merge_agent_results(
            state.get("agent_results", {}),
            "orders",
            {
                "task": current_task.get("goal", "Order lookup"),
                "original_query": user_query,
                "buyer_id_present": bool(buyer_id),
                "order_number": order_number,
                "result": order_result,
            },
        ),
        "current_step": state["current_step"] + 1,
    }