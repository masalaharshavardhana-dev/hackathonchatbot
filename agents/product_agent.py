from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from graph.state import merge_agent_results
from tools.product_tools import _search_products_impl, _search_subcategory_impl


logger = logging.getLogger(__name__)


PRODUCT_PROMPT = """
You are a Product Assistant for a wholesale marketplace.

Instructions:
- Understand natural language product requests.
- Extract the important product or category keywords from the user's message.
- Ignore conversational words like I want, Show me, Find, Need, Looking for, Search.
- Try subcategory lookup first.
- If subcategory lookup fails, fall back to product-name search.
- Never stop after a failed subcategory search.
- Always try another search strategy before saying that nothing was found.
- Return customer-facing product results only.
"""


STOP_PHRASES = {
    "i want",
    "i need",
    "show me",
    "show",
    "find",
    "need",
    "looking for",
    "search",
    "give me",
    "get me",
    "do you have",
    "can you get",
    "can you show",
    "please",
}


def _get_user_query(state: dict[str, Any]) -> str:
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            return message.content or ""
    return ""


def _clean_user_query(user_query: str) -> str:
    normalized = user_query.strip().lower()
    for phrase in STOP_PHRASES:
        normalized = normalized.replace(phrase, " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _extract_product_keyword(user_query: str) -> str:
    cleaned = _clean_user_query(user_query)
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        return user_query.strip().lower()

    filler_words = {
        "product",
        "products",
        "items",
        "item",
        "available",
        "for",
        "in",
        "the",
        "a",
        "an",
        "of",
        "to",
        "with",
        "and",
        "best",
        "premium",
        "quality",
        "fresh",
        "organic",
        "pure",
        "wholesale",
    }

    significant_tokens = [token for token in tokens if token not in filler_words]
    if not significant_tokens:
        significant_tokens = tokens

    if len(significant_tokens) >= 3:
        return " ".join(significant_tokens[-2:])

    return " ".join(significant_tokens)


def _extract_filters(user_query: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    normalized = user_query.lower()

    max_price_match = re.search(
        r"(?:below|under|upto|up to|less than|max(?:imum)?(?: price)?)[^\d₹]*₹?\s*(\d+(?:\.\d+)?)",
        normalized,
    )
    if max_price_match:
        filters["max_price"] = float(max_price_match.group(1))

    min_price_match = re.search(
        r"(?:above|over|more than|min(?:imum)?)[^\d₹]*₹?\s*(\d+(?:\.\d+)?)",
        normalized,
    )
    if min_price_match:
        filters["min_price"] = float(min_price_match.group(1))

    rating_match = re.search(r"rating[^\d]*(\d+(?:\.\d+)?)", normalized)
    if rating_match:
        filters["min_rating"] = float(rating_match.group(1))

    moq_match = re.search(r"moq[^\d]*(\d+)", normalized)
    if moq_match:
        filters["max_moq"] = int(moq_match.group(1))

    delivery_match = re.search(r"(?:delivery|deliver(?:y)? days?)[^\d]*(\d+)", normalized)
    if delivery_match:
        filters["max_delivery_days"] = int(delivery_match.group(1))

    if "featured" in normalized:
        filters["featured"] = True

    if "out of stock" in normalized:
        filters["in_stock"] = False
    elif any(keyword in normalized for keyword in ["in stock", "available", "availability"]):
        filters["in_stock"] = True

    return filters


def _pick_best_products(product_result: dict[str, Any]) -> list[dict[str, Any]]:
    products = product_result.get("products") if isinstance(product_result, dict) else None
    if isinstance(products, list):
        return [product for product in products if isinstance(product, dict)]
    return []


def _format_no_results(keyword: str) -> str:
    return f"No products found for '{keyword}'."


def product_agent(state):
    current_task = state.get("execution_plan", [])[state.get("current_step", 0)]
    user_query = _get_user_query(state)
    extracted_keyword = _extract_product_keyword(user_query)
    filters = _extract_filters(user_query)

    logger.debug(
        "product_agent start user_query=%s extracted_keyword=%s filters=%s",
        user_query,
        extracted_keyword,
        filters,
    )

    subcategory_result = _search_subcategory_impl(extracted_keyword)
    logger.debug("product_agent subcategory_result=%s", subcategory_result)

    selected_products: list[dict[str, Any]] = []
    product_search_result: dict[str, Any]
    fallback_used = False

    subcategory_id = None
    if subcategory_result.get("success"):
        subcategory = subcategory_result.get("subcategory") or {}
        if isinstance(subcategory, dict):
            subcategory_id = subcategory.get("id")

    if subcategory_id:
        product_search_result = _search_products_impl(subcategory_id=subcategory_id, **filters)
        selected_products = _pick_best_products(product_search_result)
        logger.debug("product_agent product_search_result=%s", product_search_result)

        if not selected_products:
            fallback_used = True
            product_search_result = _search_products_impl(product_name=user_query, **filters)
            selected_products = _pick_best_products(product_search_result)
            logger.debug("product_agent fallback_product_search_result=%s", product_search_result)
    else:
        fallback_used = True
        product_search_result = _search_products_impl(product_name=user_query, **filters)
        selected_products = _pick_best_products(product_search_result)
        logger.debug("product_agent fallback_product_search_result=%s", product_search_result)

    logger.debug(
        "product_agent final_selected_products=%s",
        [product.get("id") for product in selected_products],
    )

    if selected_products:
        response_text = "Product search completed."
    else:
        response_text = _format_no_results(extracted_keyword)

    return {
        "messages": [HumanMessage(content=response_text)],
        "agent_results": merge_agent_results(
            state.get("agent_results", {}),
            "product",
            {
                "task": current_task.get("goal", "Product lookup"),
                "original_query": user_query,
                "extracted_keyword": extracted_keyword,
                "filters": filters,
                "subcategory_search": subcategory_result,
                "product_search": product_search_result,
                "selected_products": selected_products,
                "fallback_used": fallback_used,
            },
        ),
        "current_step": state["current_step"] + 1,
    }