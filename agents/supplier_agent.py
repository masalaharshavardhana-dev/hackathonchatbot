from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from graph.state import merge_agent_results
from llm import get_llm_with_tools
from tools.supplier_tools import get_suppliers_by_ids, get_supplier_details


tools = [get_suppliers_by_ids, get_supplier_details]


SUPPLIER_PROMPT = """
You are a Supplier Assistant for a wholesale marketplace.

Rules:
- Do not search the sellers table independently by business name.
- Reuse the Product Agent output from shared state when supplier recommendations are needed.
- Extract seller IDs from the product results and use get_suppliers_by_ids.
- Use get_supplier_details only when the user explicitly asks for detailed information about a specific supplier.
- Always prefer the seller IDs already returned by the Product Agent.
"""


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


def supplier_agent(state):
    current_task = state.get("execution_plan", [])[state.get("current_step", 0)]
    user_query = _get_user_query(state)
    product_result = state.get("agent_results", {}).get("product", {})
    seller_ids = _extract_seller_ids(product_result)
    sort_by = _sort_by_hint(user_query)

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
                },
            ),
            "current_step": state["current_step"] + 1,
        }

    if seller_ids:
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
                },
            ),
            "current_step": state["current_step"] + 1,
        }

    llm = get_llm_with_tools(tools)
    response = llm.invoke(
        [
            SystemMessage(content=SUPPLIER_PROMPT),
            HumanMessage(content=f"User request: {user_query}\nCurrent task: {current_task.get('goal', '')}\nProduct results: {product_result}"),
        ]
    )

    return {"messages": [response]}