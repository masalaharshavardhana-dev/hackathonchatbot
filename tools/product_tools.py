from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

from langchain_core.tools import tool

from database.supabase_client import supabase


logger = logging.getLogger(__name__)


def _get_supabase_client():
    return supabase


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _score_text_match(query: str, candidate: str) -> float:
    normalized_query = _normalize_text(query)
    normalized_candidate = _normalize_text(candidate)

    if not normalized_query or not normalized_candidate:
        return 0.0

    if normalized_query == normalized_candidate:
        return 1.0

    if normalized_query in normalized_candidate or normalized_candidate in normalized_query:
        return 0.95

    return SequenceMatcher(None, normalized_query, normalized_candidate).ratio()


def _extract_search_terms(query: str) -> list[str]:
    stopwords = {
        "i",
        "want",
        "need",
        "show",
        "find",
        "search",
        "looking",
        "for",
        "me",
        "please",
        "give",
        "get",
        "any",
        "the",
        "a",
        "an",
        "of",
        "to",
        "do",
        "you",
        "have",
        "wanting",
        "available",
        "items",
        "item",
        "products",
        "product",
        "grocery",
        "groceries",
        "premium",
        "best",
        "good",
        "quality",
        "fresh",
        "organic",
        "pure",
        "top",
        "price",
        "below",
        "under",
        "over",
        "less",
        "more",
        "than",
        "or",
        "and",
        "with",
    }
    normalized = _normalize_text(query)
    return [token for token in normalized.split() if token and token not in stopwords]


def _validate_limit(limit: int | None) -> int:
    if limit is None:
        return 10
    try:
        parsed_limit = int(limit)
    except (TypeError, ValueError):
        return 10

    if parsed_limit <= 0:
        return 10
    return min(parsed_limit, 20)


def _normalize_product(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "slug": row.get("slug"),
        "name": row.get("name"),
        "brand": row.get("brand"),
        "seller_id": row.get("seller_id"),
        "category": row.get("category_slug") or row.get("category") or None,
        "sub_category": row.get("sub_category"),
        "subcategory_id": row.get("subcategory_id"),
        "wholesale_price": row.get("wholesale_price") or row.get("mrp"),
        "mrp": row.get("mrp"),
        "rating": row.get("rating"),
        "review_count": row.get("review_count"),
        "supplier": row.get("supplier"),
        "moq": row.get("moq"),
        "unit": row.get("unit"),
        "featured": row.get("featured"),
        "in_stock": row.get("in_stock"),
        "stock_count": row.get("stock_count"),
        "description": row.get("description"),
        "delivery_days": row.get("delivery_days") or row.get("delivery_estimate"),
    }


def _normalize_subcategory(row: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    payload = {
        "id": row.get("id"),
        "category_id": row.get("category_id"),
        "slug": row.get("slug"),
        "name": row.get("name"),
        "image": row.get("image"),
        "sort_order": row.get("sort_order"),
    }
    if score is not None:
        payload["score"] = round(score, 4)
    return payload


def _parse_bool_flag(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def _build_product_query(
    client,
    subcategory_id: str | None = None,
    product_name: str | None = None,
    brand: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_rating: float | None = None,
    featured: bool | None = None,
    in_stock: bool | None = None,
    max_moq: int | None = None,
    max_delivery_days: int | None = None,
    sort_by: str | None = None,
):
    query = client.table("products").select(
        "id,slug,name,brand,seller_id,category_slug,sub_category,subcategory_id,wholesale_price,mrp,moq,unit,supplier,rating,review_count,featured,in_stock,stock_count,description,delivery_days,delivery_estimate,created_at"
    )

    if subcategory_id:
        query = query.eq("subcategory_id", subcategory_id)

    if product_name:
        query = query.or_(
            f"name.ilike.%{product_name}%,brand.ilike.%{product_name}%,sub_category.ilike.%{product_name}%"
        )

    if brand:
        query = query.ilike("brand", f"%{brand}%")

    if min_price is not None:
        query = query.gte("wholesale_price", min_price)

    if max_price is not None:
        query = query.lte("wholesale_price", max_price)

    if min_rating is not None:
        query = query.gte("rating", min_rating)

    if featured is not None:
        query = query.eq("featured", featured)

    if in_stock is not None:
        query = query.eq("in_stock", in_stock)

    if max_moq is not None:
        query = query.lte("moq", max_moq)

    if max_delivery_days is not None:
        query = query.lte("delivery_days", max_delivery_days)

    if sort_by == "rating":
        query = query.order("rating", desc=True)
    elif sort_by == "price_low":
        query = query.order("wholesale_price", desc=False)
    elif sort_by == "price_high":
        query = query.order("wholesale_price", desc=True)
    elif sort_by == "featured":
        query = query.order("featured", desc=True).order("rating", desc=True)
    elif sort_by == "newest":
        query = query.order("created_at", desc=True)
    else:
        query = query.order("rating", desc=True).order("featured", desc=True)

    return query


def _search_subcategory_impl(query: str):
    client = _get_supabase_client()
    if client is None:
        return {
            "success": False,
            "message": "Supabase client is not configured.",
            "subcategory": None,
            "top_matches": [],
            "count": 0,
        }

    normalized_query = (query or "").strip()
    if not normalized_query:
        return {
            "success": False,
            "message": "Query is required to search subcategories.",
            "subcategory": None,
            "top_matches": [],
            "count": 0,
        }

    try:
        terms = _extract_search_terms(normalized_query)
        logger.debug(
            "search_subcategory query=%s extracted_terms=%s",
            normalized_query,
            terms,
        )

        response = client.table("subcategories").select(
            "id,category_id,slug,name,image,sort_order,created_at,updated_at"
        ).execute()
        rows = response.data or []

        scored_rows = []
        for row in rows:
            name = row.get("name", "")
            slug = row.get("slug", "")
            normalized_name = _normalize_text(name)
            normalized_slug = _normalize_text(slug)

            score = 0.0
            if normalized_query == normalized_name or normalized_query == normalized_slug:
                score = 1.0
            elif normalized_query in normalized_name or normalized_query in normalized_slug:
                score = 0.97
            elif normalized_name in normalized_query or normalized_slug in normalized_query:
                score = 0.94
            else:
                for term in terms:
                    if term == normalized_name or term == normalized_slug:
                        score = max(score, 0.95)
                    elif term in normalized_name or term in normalized_slug:
                        score = max(score, 0.9)
                    else:
                        score = max(
                            score,
                            _score_text_match(term, normalized_name),
                            _score_text_match(term, normalized_slug),
                        )

            if score > 0:
                scored_rows.append((score, row))

        scored_rows.sort(key=lambda item: (-item[0], item[1].get("sort_order") or 0, item[1].get("name") or ""))
        top_matches = [_normalize_subcategory(row, score) for score, row in scored_rows[:5]]
        logger.debug("search_subcategory result=%s", top_matches[:1])

        if not top_matches:
            return {
                "success": True,
                "found": False,
                "message": "No matching subcategory found.",
                "subcategory": None,
                "top_matches": [],
                "count": 0,
            }

        best_match = top_matches[0]
        return {
            "success": True,
            "found": True,
            "message": "Matching subcategory found.",
            "subcategory": {
                "id": best_match["id"],
                "name": best_match["name"],
                "slug": best_match["slug"],
            },
            "top_matches": top_matches,
            "count": len(top_matches),
        }
    except Exception as exc:
        return {
            "success": False,
            "message": f"Failed to search subcategories: {exc}",
            "subcategory": None,
            "top_matches": [],
            "count": 0,
        }


def _search_products_impl(
    subcategory_id: str | None = None,
    product_name: str | None = None,
    brand: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_rating: float | None = None,
    featured: bool | None = None,
    in_stock: bool | None = None,
    max_moq: int | None = None,
    max_delivery_days: int | None = None,
    limit: int = 10,
    sort_by: str | None = None,
):
    client = _get_supabase_client()
    if client is None:
        return {
            "success": False,
            "count": 0,
            "message": "Supabase client is not configured.",
            "products": [],
        }

    if not subcategory_id and not product_name:
        return {
            "success": False,
            "count": 0,
            "message": "subcategory_id or product_name is required.",
            "products": [],
        }

    limit = _validate_limit(limit)

    try:
        query = _build_product_query(
            client=client,
            subcategory_id=subcategory_id,
            product_name=product_name,
            brand=brand,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            featured=featured,
            in_stock=in_stock,
            max_moq=max_moq,
            max_delivery_days=max_delivery_days,
            sort_by=sort_by,
        )

        response = query.limit(limit).execute()
        rows = response.data or []
        products = [_normalize_product(row) for row in rows]
        logger.debug(
            "search_products query subcategory_id=%s product_name=%s brand=%s min_price=%s max_price=%s min_rating=%s featured=%s in_stock=%s max_moq=%s max_delivery_days=%s sort_by=%s limit=%s result_count=%s",
            subcategory_id,
            product_name,
            brand,
            min_price,
            max_price,
            min_rating,
            featured,
            in_stock,
            max_moq,
            max_delivery_days,
            sort_by,
            limit,
            len(products),
        )

        if not products:
            return {
                "success": True,
                "found": False,
                "count": 0,
                "message": "No products found.",
                "products": [],
            }

        return {
            "success": True,
            "found": True,
            "count": len(products),
            "message": "Products retrieved successfully.",
            "products": products,
        }
    except Exception as exc:
        return {
            "success": False,
            "count": 0,
            "message": f"Failed to fetch products: {exc}",
            "products": [],
        }


@tool
def search_subcategory(query: str):
    """
    Find the best matching product subcategory for a user request.

    Use this tool to identify the correct product category or subcategory from natural language
    requests, even when the user describes the product with extra words such as "coffee powder",
    "basmati rice", or "red chilli powder". It should be used to search product categories first
    so the agent can map user language to the right subcategory record.

    Matching order:
    - exact match on subcategory name or slug
    - partial match on subcategory name or slug
    - match any significant keyword from the query
    - fuzzy fallback on name and slug

    The response includes the best match plus top matches when several subcategories are similar.
    """
    return _search_subcategory_impl(query)


@tool
def search_products(
    subcategory_id: str | None = None,
    product_name: str | None = None,
    brand: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_rating: float | None = None,
    featured: bool | None = None,
    in_stock: bool | None = None,
    max_moq: int | None = None,
    max_delivery_days: int | None = None,
    limit: int = 10,
    sort_by: str | None = None,
):
    """
    Retrieve products using either a subcategory_id or a product_name fallback, and optionally
    apply product filters.

    Use this tool after the correct subcategory_id is known. If no subcategory is matched, it can
    still search by product name as a fallback. It supports brand filtering, price filtering,
    featured products, rating filtering, stock filtering, MOQ filtering, and delivery-time
    filtering.

    When subcategory_id is provided, the query always uses .eq("subcategory_id", subcategory_id).
    When product_name is provided, the query searches product name, brand, and sub_category text.

    Sorting modes supported:
    - rating
    - price_low
    - price_high
    - featured
    - newest

    The tool returns production-ready structured output with success, count, and products.
    """
    return _search_products_impl(
        subcategory_id=subcategory_id,
        product_name=product_name,
        brand=brand,
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
        featured=featured,
        in_stock=in_stock,
        max_moq=max_moq,
        max_delivery_days=max_delivery_days,
        limit=limit,
        sort_by=sort_by,
    )


@tool
def get_product_details(product_id: int):
    """
    Use this tool when the user wants detailed information about a specific product.

    It is appropriate for requests such as asking for full product details, specifications,
    highlights, packaging details, supplier information, or images for a known product ID.
    """
    client = _get_supabase_client()
    if client is None:
        return {
            "success": False,
            "message": "Supabase client is not configured.",
            "product": None,
        }

    try:
        response = client.table("products").select("*").eq("id", product_id).maybe_single().execute()
        row = response.data

        if not row:
            return {
                "success": True,
                "found": False,
                "message": f"No product found with id {product_id}.",
                "product": None,
            }

        return {
            "success": True,
            "found": True,
            "product": {
                "id": row.get("id"),
                "slug": row.get("slug"),
                "name": row.get("name"),
                "brand": row.get("brand"),
                "category": row.get("category_slug") or row.get("category"),
                "description": row.get("description"),
                "specifications": row.get("specifications"),
                "highlights": row.get("highlights"),
                "packaging_details": row.get("packaging_details"),
                "supplier": row.get("supplier"),
                "images": row.get("images"),
                "wholesale_price": row.get("wholesale_price") or row.get("mrp"),
                "mrp": row.get("mrp"),
                "rating": row.get("rating"),
                "review_count": row.get("review_count"),
                "featured": row.get("featured"),
                "in_stock": row.get("in_stock"),
                "stock_count": row.get("stock_count"),
                "delivery_days": row.get("delivery_days"),
                "delivery_estimate": row.get("delivery_estimate"),
            },
        }
    except Exception as exc:
        return {
            "success": False,
            "message": f"Failed to fetch product details: {exc}",
            "product": None,
        }