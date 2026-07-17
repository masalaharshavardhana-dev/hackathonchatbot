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


def _fetch_product_rows(client):
    response = client.table("products").select("id,category_slug,sub_category,subcategory_id").execute()
    return response.data or []


def _get_analytics_snapshot() -> dict[str, Any]:
    client = _get_supabase_client()
    if client is None:
        return {
            "success": False,
            "message": "Supabase client is not configured.",
            "total_products": 0,
            "categories": [],
            "subcategories": [],
            "total_categories": 0,
            "total_subcategories": 0,
        }

    try:
        product_rows = _fetch_product_rows(client)
        category_counts: dict[str, int] = {}
        subcategory_counts: dict[str, int] = {}

        for row in product_rows:
            category_name = (row.get("category_slug") or row.get("category") or "Uncategorized").strip() or "Uncategorized"
            category_counts[category_name] = category_counts.get(category_name, 0) + 1

            subcategory_name = (row.get("sub_category") or row.get("subcategory_id") or "Uncategorized").strip() or "Uncategorized"
            subcategory_counts[subcategory_name] = subcategory_counts.get(subcategory_name, 0) + 1

        categories = [
            {"category": category_name, "product_count": count}
            for category_name, count in sorted(category_counts.items(), key=lambda item: item[0].lower())
        ]
        subcategories = [
            {"subcategory": subcategory_name, "product_count": count}
            for subcategory_name, count in sorted(subcategory_counts.items(), key=lambda item: item[0].lower())
        ]

        subcategory_response = client.table("subcategories").select("id,name").execute()
        subcategory_rows = subcategory_response.data or []
        total_subcategories = len(subcategory_rows) or len(subcategories)

        return {
            "success": True,
            "total_products": len(product_rows),
            "categories": categories,
            "subcategories": subcategories,
            "total_categories": len(categories),
            "total_subcategories": total_subcategories,
        }
    except Exception as exc:
        return {
            "success": False,
            "message": f"Failed to fetch analytics: {exc}",
            "total_products": 0,
            "categories": [],
            "subcategories": [],
            "total_categories": 0,
            "total_subcategories": 0,
        }


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


# --- NEW PRODUCT AGENT SPECIFIC TOOL FUNCTIONS & FILTER TOOLS ---

@tool
def extract_filters(user_query: str) -> dict[str, Any]:
    """
    Extract search filters (e.g. max_price, min_price, min_rating, max_moq, max_delivery_days, featured, in_stock) from the user query.
    """
    filters: dict[str, Any] = {}
    if not user_query:
        return filters
    normalized = user_query.lower()

    # Extract specific matches first and remove them to avoid double matching as price
    rating_match = re.search(r"rating\s*(?:above|below|under|over|more than|less than|min|max)?\s*(\d+(?:\.\d+)?)", normalized)
    if rating_match:
        filters["min_rating"] = float(rating_match.group(1))
        normalized = normalized.replace(rating_match.group(0), "")

    moq_match = re.search(r"moq\s*(?:above|below|under|over|more than|less than|min|max)?\s*(\d+)", normalized)
    if moq_match:
        filters["max_moq"] = int(moq_match.group(1))
        normalized = normalized.replace(moq_match.group(0), "")

    delivery_match = re.search(r"(?:delivery|deliver)(?:\s*days)?\s*(?:within|under|below|max|in)?\s*(\d+)", normalized)
    if delivery_match:
        filters["max_delivery_days"] = int(delivery_match.group(1))
        normalized = normalized.replace(delivery_match.group(0), "")

    max_price_match = re.search(
        r"(?:below|under|upto|up to|less than|max(?:imum)?(?: price)?)[^\d?]*?\s*(\d+(?:\.\d+)?)",
        normalized,
    )
    if max_price_match:
        filters["max_price"] = float(max_price_match.group(1))
        normalized = normalized.replace(max_price_match.group(0), "")

    min_price_match = re.search(
        r"(?:above|over|more than|min(?:imum)?(?: price)?)[^\d?]*?\s*(\d+(?:\.\d+)?)",
        normalized,
    )
    if min_price_match:
        filters["min_price"] = float(min_price_match.group(1))
        normalized = normalized.replace(min_price_match.group(0), "")

    if "featured" in normalized:
        filters["featured"] = True

    if "out of stock" in normalized:
        filters["in_stock"] = False
    elif any(keyword in normalized for keyword in ["in stock", "available", "availability"]):
        filters["in_stock"] = True

    if any(k in normalized for k in ["highest price", "most expensive", "costly", "costliest", "highest"]):
        filters["sort_by"] = "price_high"
    elif any(k in normalized for k in ["lowest price", "cheapest", "least expensive", "lowest", "cheap"]):
        filters["sort_by"] = "price_low"

    # Remove weight units to avoid matching as result limits
    normalized = re.sub(r"\b\d+\s*(?:kg|g|gm|gms|ml|l|ltr|ltrs|oz|pcs|packet|pack|packs)\b", "", normalized)

    # Limit extraction
    limit_match = re.search(r"\b(\d+)\b\s*(?:products|items|goods)?", normalized)
    if limit_match:
        val = int(limit_match.group(1))
        if val > 0:
            filters["limit"] = val
    elif any(k in normalized for k in ["highest", "lowest", "cheapest", "most expensive", "costliest", "single", "one", "1"]):
        filters["limit"] = 1

    return filters


@tool
def extract_core_query(user_query: str) -> str:
    """
    Extract the core query term (removing stop words and filter phrases) from the user query.
    """
    if not user_query:
        return ""
    cleaned = user_query.lower()
    cleaned = re.sub(
        r"\b(?:below|under|upto|up to|less than|above|over|more than|min|max|price|rating|moq|delivery|deliver|days?|featured|in stock|out of stock|available|availability)\s*\d*(?:\.\d+)?\b",
        " ",
        cleaned,
    )
    stop_words = {
        "i", "want", "need", "show", "find", "search", "looking", "for", "me", "please",
        "give", "get", "any", "the", "a", "an", "of", "to", "do", "you", "have", "wanting",
        "products", "product", "items", "item", "grocery", "groceries", "select",
        "supplier", "suppliers", "suggest", "with", "best", "good", "top", "from",
        "vendors", "vendor", "manufacturer", "manufacturers",
        "highest", "lowest", "cheapest", "expensive", "cheap", "costly", "high", "low"
    }
    tokens = re.sub(r"[^\w\s]", " ", cleaned).split()
    core_tokens = [t for t in tokens if t not in stop_words]
    return " ".join(core_tokens).strip()


def normalize_slug(text: str) -> str:
    """
    Normalize user input to a slug format:
    - lowercase
    - trim spaces
    - replace spaces with "-"
    - remove duplicate spaces/dashes
    - ignore punctuation
    """
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation, keeping alphanumeric characters, spaces, and dashes
    text = re.sub(r"[^\w\s-]", "", text)
    # Replace spaces and underscores with dashes
    text = text.replace("_", "-")
    text = re.sub(r"\s+", "-", text)
    # Remove duplicate/multiple dashes
    text = re.sub(r"-+", "-", text)
    # Trim dashes from both ends
    text = text.strip("-")
    return text


def detect_brand(user_query: str) -> str | None:
    """
    Load distinct brands from products table and compare against user query.
    - case insensitive
    - ignore spaces
    - ignore punctuation
    - allow partial matching and fuzzy similarity matching
    """
    if not user_query:
        return None

    client = _get_supabase_client()
    if client is None:
        return None

    try:
        response = client.table("products").select("brand").execute()
        rows = response.data or []
        # Extract unique, non-null brand names
        brands = sorted(list({row.get("brand") for row in rows if row.get("brand")}))
    except Exception as exc:
        logger.error(f"Error fetching brands: {exc}")
        return None

    def clean_and_tokenize(text: str) -> list[str]:
        text = re.sub(r"[^\w\s]", "", text.lower())
        return [w for w in text.split() if w]

    query_tokens = clean_and_tokenize(user_query)
    if not query_tokens:
        return None

    best_match = None
    best_score = 0.0

    from difflib import SequenceMatcher

    for brand in brands:
        brand_tokens = clean_and_tokenize(brand)
        if not brand_tokens:
            continue

        # Case 1: Brand is contiguous subsequence of Query (with fuzzy similarity)
        if len(brand_tokens) <= len(query_tokens):
            for i in range(len(query_tokens) - len(brand_tokens) + 1):
                sub_query = query_tokens[i : i + len(brand_tokens)]
                if sub_query == brand_tokens:
                    score = 1.0
                else:
                    score = SequenceMatcher(None, "".join(sub_query), "".join(brand_tokens)).ratio()

                if score >= 0.8:
                    if score > best_score or (score == best_score and len(brand) > len(best_match or "")):
                        best_score = score
                        best_match = brand

        # Case 2: Query is contiguous subsequence of Brand (with fuzzy similarity)
        if len(query_tokens) <= len(brand_tokens):
            for i in range(len(brand_tokens) - len(query_tokens) + 1):
                sub_brand = brand_tokens[i : i + len(query_tokens)]
                if sub_brand == query_tokens:
                    score = 1.0
                else:
                    score = SequenceMatcher(None, "".join(sub_brand), "".join(query_tokens)).ratio()

                if score >= 0.8:
                    if score > best_score or (score == best_score and len(brand) > len(best_match or "")):
                        best_score = score
                        best_match = brand

    return best_match if best_score >= 0.8 else None


def _apply_filters_to_query(query: Any, filters: dict[str, Any]) -> Any:
    if not filters:
        return query
    if "min_price" in filters:
        query = query.gte("wholesale_price", filters["min_price"])
    if "max_price" in filters:
        query = query.lte("wholesale_price", filters["max_price"])
    if "min_rating" in filters:
        query = query.gte("rating", filters["min_rating"])
    if "max_moq" in filters:
        query = query.lte("moq", filters["max_moq"])
    if "max_delivery_days" in filters:
        query = query.lte("delivery_days", filters["max_delivery_days"])
    if "featured" in filters:
        query = query.eq("featured", filters["featured"])
    if "in_stock" in filters:
        query = query.eq("in_stock", filters["in_stock"])
    if "sort_by" in filters:
        if filters["sort_by"] == "price_high":
            query = query.order("wholesale_price", desc=True)
        elif filters["sort_by"] == "price_low":
            query = query.order("wholesale_price", desc=False)
        elif filters["sort_by"] == "rating":
            query = query.order("rating", desc=True)
    if "limit" in filters:
        query = query.limit(filters["limit"])
    return query


def search_brand_products(brand: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Retrieve products matching a brand case-insensitively.
    """
    client = _get_supabase_client()
    if client is None:
        return {"success": False, "message": "Supabase client not configured", "products": []}

    try:
        query = client.table("products").select("*").ilike("brand", brand)
        if filters:
            query = _apply_filters_to_query(query, filters)
        response = query.execute()
        products = [_normalize_product(row) for row in (response.data or [])]
        return {
            "success": True,
            "found": len(products) > 0,
            "count": len(products),
            "products": products,
            "brand": brand
        }
    except Exception as exc:
        return {"success": False, "message": f"Error searching brand products: {exc}", "products": []}


def search_category_products(category_slug: str, category_id: str | None = None, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Search categories, fetch category_id, search subcategories for it,
    then fetch and return products.
    """
    client = _get_supabase_client()
    if client is None:
        return {"success": False, "message": "Supabase client not configured", "products": []}

    try:
        if not category_id:
            category_res = client.table("categories").select("id, name").eq("slug", category_slug).maybe_single().execute()
            if not category_res or not category_res.data:
                category_res = client.table("categories").select("id, name").ilike("slug", category_slug).maybe_single().execute()

            if not category_res or not category_res.data:
                return {"success": True, "found": False, "message": f"Category '{category_slug}' not found", "products": []}

            category_id = category_res.data["id"]
            category_name = category_res.data["name"]
        else:
            category_res = client.table("categories").select("name").eq("id", category_id).maybe_single().execute()
            category_name = category_res.data.get("name") if category_res and category_res.data else category_slug

        # Fetch subcategory IDs
        subcat_res = client.table("subcategories").select("id").eq("category_id", category_id).execute()
        subcategory_ids = [row["id"] for row in (subcat_res.data or [])]

        if not subcategory_ids:
            return {
                "success": True,
                "found": False,
                "message": f"No subcategories found for category '{category_name}'",
                "products": [],
                "category_name": category_name
            }

        # Fetch products
        query = client.table("products").select("*").in_("subcategory_id", subcategory_ids)
        if filters:
            query = _apply_filters_to_query(query, filters)
        prod_res = query.execute()
        products = [_normalize_product(row) for row in (prod_res.data or [])]

        return {
            "success": True,
            "found": len(products) > 0,
            "count": len(products),
            "products": products,
            "category_name": category_name
        }
    except Exception as exc:
        return {"success": False, "message": f"Error searching category products: {exc}", "products": []}


def search_subcategory_products(subcategory_slug: str, subcategory_id: str | None = None, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Search subcategory by slug, get subcategory id, query products, and return them.
    """
    client = _get_supabase_client()
    if client is None:
        return {"success": False, "message": "Supabase client not configured", "products": []}

    try:
        if not subcategory_id:
            subcat_res = client.table("subcategories").select("id, name").eq("slug", subcategory_slug).maybe_single().execute()
            if not subcat_res or not subcat_res.data:
                subcat_res = client.table("subcategories").select("id, name").ilike("slug", subcategory_slug).maybe_single().execute()

            if not subcat_res or not subcat_res.data:
                return {"success": True, "found": False, "message": f"Subcategory '{subcategory_slug}' not found", "products": []}

            subcategory_id = subcat_res.data["id"]
            subcategory_name = subcat_res.data["name"]
        else:
            subcat_res = client.table("subcategories").select("name").eq("id", subcategory_id).maybe_single().execute()
            subcategory_name = subcat_res.data.get("name") if subcat_res and subcat_res.data else subcategory_slug

        # Call the helper function directly to fetch products
        res = _search_products_db(sub_category_id=subcategory_id, filters=filters)
        if res.get("success") and res.get("products"):
            return {
                "success": True,
                "found": True,
                "count": res.get("count", 0),
                "products": res.get("products", []),
                "subcategory_name": subcategory_name
            }

        return {
            "success": True,
            "found": False,
            "message": f"No products found under subcategory '{subcategory_name}'",
            "products": [],
            "subcategory_name": subcategory_name
        }
    except Exception as exc:
        return {"success": False, "message": f"Error searching subcategory products: {exc}", "products": []}


def _search_products_db(user_query: str = "", sub_category_id: str | None = None, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Underlying database coordinator function to prevent StructuredTool calling issues.
    """
    client = _get_supabase_client()
    if client is None:
        return {"success": False, "message": "Supabase client not configured", "products": []}

    # If sub_category_id is provided directly, search directly in products table
    if sub_category_id:
        try:
            query = client.table("products").select("*").eq("subcategory_id", sub_category_id)
            if filters:
                query = _apply_filters_to_query(query, filters)
            prod_res = query.execute()
            products = [_normalize_product(row) for row in (prod_res.data or [])]
            return {
                "success": True,
                "found": len(products) > 0,
                "count": len(products),
                "products": products
            }
        except Exception as exc:
            return {"success": False, "message": f"Error fetching products: {exc}", "products": []}

    if not user_query:
        return {"success": True, "found": False, "message": "No query provided", "products": []}

    # Analytics Intent Detection
    analytics_keywords = ["how many", "total", "count", "statistics", "summary", "inventory"]
    normalized_query = user_query.lower()
    if any(keyword in normalized_query for keyword in analytics_keywords):
        if "category" in normalized_query and "subcategory" not in normalized_query and "sub-category" not in normalized_query:
            res = get_category_statistics.invoke({})
            return {
                "result_type": "analytics",
                "selected_tool": "get_category_statistics",
                "analytics_result": res,
            }
        elif "subcategory" in normalized_query or "sub-category" in normalized_query:
            res = get_subcategory_statistics.invoke({})
            return {
                "result_type": "analytics",
                "selected_tool": "get_subcategory_statistics",
                "analytics_result": res,
            }
        elif "how many products" in normalized_query and "category" not in normalized_query:
            res = get_total_product_count.invoke({})
            return {
                "result_type": "analytics",
                "selected_tool": "get_total_product_count",
                "analytics_result": res,
            }
        else:
            res = get_inventory_summary.invoke({})
            return {
                "result_type": "analytics",
                "selected_tool": "get_inventory_summary",
                "analytics_result": res,
            }

    # Extract filters and the core search query
    extracted_filters = extract_filters.func(user_query)
    core_query = extract_core_query.func(user_query) or user_query

    # Combine passed filters and extracted filters
    merged_filters = dict(filters or {})
    merged_filters.update(extracted_filters)

    # Step 1: Brand Detection
    brand_match = detect_brand(core_query)
    if brand_match:
        logger.info(f"Brand detected: {brand_match}")
        res = search_brand_products(brand_match, filters=merged_filters)
        if not res.get("products") and merged_filters:
            fallback_filters = {k: v for k, v in merged_filters.items() if k not in ["min_price", "max_price"]}
            res_fallback = search_brand_products(brand_match, filters=fallback_filters)
            if res_fallback.get("products"):
                res = res_fallback
                res["fallback_used"] = True
        res["search_type"] = "brand"
        return res

    # Normalize input for Category and SubCategory searches
    normalized = normalize_slug(core_query)

    # Step 2: Category Search
    try:
        cat_res = client.table("categories").select("id, name, slug").eq("slug", normalized).maybe_single().execute()
        if not cat_res or not cat_res.data:
            cat_res = client.table("categories").select("id, name, slug").ilike("slug", normalized).maybe_single().execute()
        # Substring/fuzzy fallback for category slug
        if not cat_res or not cat_res.data:
            cat_res = client.table("categories").select("id, name, slug").ilike("slug", f"%{normalized}%").maybe_single().execute()

        if cat_res and cat_res.data:
            category_slug = cat_res.data["slug"]
            category_id = cat_res.data["id"]
            logger.info(f"Category detected: {cat_res.data['name']} (ID: {category_id})")
            res = search_category_products(category_slug=category_slug, category_id=category_id, filters=merged_filters)
            if not res.get("products") and merged_filters:
                fallback_filters = {k: v for k, v in merged_filters.items() if k not in ["min_price", "max_price"]}
                res_fallback = search_category_products(category_slug=category_slug, category_id=category_id, filters=fallback_filters)
                if res_fallback.get("products"):
                    res = res_fallback
                    res["fallback_used"] = True
            res["search_type"] = "category"
            return res
    except Exception as exc:
        logger.error(f"Error querying categories: {exc}")

    # Step 3: SubCategory Search
    try:
        subcat_res = client.table("subcategories").select("id, name, slug").eq("slug", normalized).maybe_single().execute()
        if not subcat_res or not subcat_res.data:
            subcat_res = client.table("subcategories").select("id, name, slug").ilike("slug", normalized).maybe_single().execute()
        # Substring/fuzzy fallback for subcategory slug
        if not subcat_res or not subcat_res.data:
            subcat_res = client.table("subcategories").select("id, name, slug").ilike("slug", f"%{normalized}%").maybe_single().execute()

        if subcat_res and subcat_res.data:
            subcategory_slug = subcat_res.data["slug"]
            subcategory_id = subcat_res.data["id"]
            logger.info(f"Subcategory detected: {subcat_res.data['name']} (ID: {subcategory_id})")
            res = search_subcategory_products(subcategory_slug=subcategory_slug, subcategory_id=subcategory_id, filters=merged_filters)
            if not res.get("products") and merged_filters:
                fallback_filters = {k: v for k, v in merged_filters.items() if k not in ["min_price", "max_price"]}
                res_fallback = search_subcategory_products(subcategory_slug=subcategory_slug, subcategory_id=subcategory_id, filters=fallback_filters)
                if res_fallback.get("products"):
                    res = res_fallback
                    res["fallback_used"] = True
            res["search_type"] = "subcategory"
            return res
    except Exception as exc:
        logger.error(f"Error querying subcategories: {exc}")

    return {
        "success": True,
        "found": False,
        "message": "Sorry, I couldn't find products matching your request.",
        "products": []
    }


@tool
def search_products(user_query: str = "", sub_category_id: str | None = None) -> dict[str, Any]:
    """
    Master function:
    Detect Brand -> YES -> search_brand_products()
    ELSE -> Normalize -> Category Search -> Found -> search_category_products()
    ELSE -> SubCategory Search -> Found -> search_subcategory_products()
    ELSE -> "No matching products found."
    """
    return _search_products_db(user_query=user_query, sub_category_id=sub_category_id)


def format_search_response(result: dict[str, Any]) -> str:
    """
    Format clean, natural language search responses.
    """
    if not result.get("success") or not result.get("found") or not result.get("products"):
        return "Sorry, I couldn't find products matching your request."

    products = result["products"]
    count = result["count"]
    search_type = result.get("search_type")

    examples = [f"• {p.get('name')}" for p in products[:4]]
    examples_str = "\n".join(examples)

    if search_type == "brand":
        brand = result.get("brand")
        return (
            f"Found {count} {brand} products.\n\n"
            f"Examples\n"
            f"{examples_str}\n\n"
            f"Would you like all {brand} products?"
        )
    elif search_type == "category":
        category_name = result.get("category_name")
        return (
            f"Found {count} products under {category_name}.\n\n"
            f"Examples:\n"
            f"{examples_str}\n\n"
            f"Would you like prices or all available products?"
        )
    elif search_type == "subcategory":
        subcategory_name = result.get("subcategory_name")
        return (
            f"Found {count} products under {subcategory_name}.\n\n"
            f"Examples:\n"
            f"{examples_str}\n\n"
            f"Would you like prices or all available products?"
        )
    else:
        return (
            f"Found {count} products.\n\n"
            f"Examples:\n"
            f"{examples_str}\n\n"
            f"Would you like prices or all available products?"
        )


# --- RETAINED ORIGINAL ANALYTIC/STATISTIC TOOLS ---

@tool
def get_total_product_count():
    """
    Return the total number of products available in the catalog.
    """
    snapshot = _get_analytics_snapshot()
    return {
        "success": snapshot.get("success", False),
        "message": snapshot.get("message"),
        "total_products": snapshot.get("total_products", 0),
    }


@tool
def get_category_statistics():
    """
    Return product counts grouped by category.
    """
    snapshot = _get_analytics_snapshot()
    return {
        "success": snapshot.get("success", False),
        "message": snapshot.get("message"),
        "categories": snapshot.get("categories", []),
        "total_products": snapshot.get("total_products", 0),
    }


@tool
def get_subcategory_statistics():
    """
    Return product counts grouped by subcategory.
    """
    snapshot = _get_analytics_snapshot()
    return {
        "success": snapshot.get("success", False),
        "message": snapshot.get("message"),
        "subcategories": snapshot.get("subcategories", []),
        "total_products": snapshot.get("total_products", 0),
    }


@tool
def get_inventory_summary():
    """
    Return a compact inventory summary with totals by category, subcategory, and product count.
    """
    snapshot = _get_analytics_snapshot()
    return {
        "success": snapshot.get("success", False),
        "message": snapshot.get("message"),
        "total_categories": snapshot.get("total_categories", 0),
        "total_subcategories": snapshot.get("total_subcategories", 0),
        "total_products": snapshot.get("total_products", 0),
    }


@tool
def search_subcategory(query: str):
    """
    Find the best matching product subcategory for a user request.
    """
    return _search_subcategory_impl(query)


@tool
def get_product_details(product_id: int):
    """
    Use this tool when the user wants detailed information about a specific product.
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