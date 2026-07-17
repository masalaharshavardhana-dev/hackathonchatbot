from typing import Any

from langchain_core.tools import tool

from database.supabase_client import supabase


def _get_supabase_client():
    return supabase


def _normalize_supplier(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "business_name": row.get("business_name"),
        "owner_name": row.get("owner_name"),
        "business_category": row.get("business_category"),
        "business_type": row.get("business_type"),
        "city": row.get("city"),
        "state": row.get("state"),
        "country": row.get("country"),
        "verification_status": row.get("verification_status"),
        "onboarding_completed": row.get("onboarding_completed"),
        "years_in_business": row.get("years_in_business"),
        "rating": row.get("rating"),
        "review_count": row.get("review_count"),
        "phone": row.get("phone"),
        "business_email": row.get("business_email"),
        "logo_url": row.get("logo_url"),
    }


def _validate_limit(limit: int | None) -> int:
    if limit is None:
        return 10
    try:
        parsed_limit = int(limit)
    except (TypeError, ValueError):
        return 10
    return max(1, min(parsed_limit, 20))


def _normalize_seller_ids(seller_ids: list[str]) -> list[str]:
    cleaned_ids = []
    seen = set()
    for seller_id in seller_ids or []:
        normalized = str(seller_id).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned_ids.append(normalized)
    return cleaned_ids


def _sort_suppliers(rows: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    def verification_rank(row: dict[str, Any]) -> int:
        status = str(row.get("verification_status") or "").strip().lower()
        return 0 if status in {"verified", "active", "approved"} else 1

    if sort_by == "reviews":
        return sorted(
            rows,
            key=lambda row: (
                verification_rank(row),
                -(row.get("review_count") or 0),
                -(row.get("rating") or 0),
            ),
        )

    if sort_by == "experience":
        return sorted(
            rows,
            key=lambda row: (
                verification_rank(row),
                -(row.get("years_in_business") or 0),
                -(row.get("rating") or 0),
            ),
        )

    if sort_by == "verified":
        return sorted(
            rows,
            key=lambda row: (
                verification_rank(row),
                -(row.get("rating") or 0),
                -(row.get("review_count") or 0),
            ),
        )

    return sorted(
        rows,
        key=lambda row: (
            verification_rank(row),
            -(row.get("rating") or 0),
            -(row.get("review_count") or 0),
        ),
    )


def _get_suppliers_by_ids_impl(seller_ids: list[str], sort_by: str = "rating", limit: int = 10):
    client = _get_supabase_client()
    if client is None:
        return {
            "success": False,
            "count": 0,
            "message": "Supabase client is not configured.",
            "suppliers": [],
        }

    cleaned_ids = _normalize_seller_ids(seller_ids)
    if not cleaned_ids:
        return {
            "success": False,
            "count": 0,
            "message": "No seller IDs were provided.",
            "suppliers": [],
        }

    limit = _validate_limit(limit)

    try:
        response = client.table("sellers").select(
            "id,business_name,owner_name,business_category,business_type,city,state,country,verification_status,onboarding_completed,years_in_business,rating,review_count,phone,business_email,logo_url"
        ).in_("id", cleaned_ids).execute()

        rows = response.data or []
        if not rows:
            return {
                "success": True,
                "found": False,
                "count": 0,
                "message": "No matching suppliers found.",
                "suppliers": [],
            }

        suppliers = [_normalize_supplier(row) for row in rows]
        suppliers = _sort_suppliers(suppliers, sort_by)[:limit]

        return {
            "success": True,
            "found": True,
            "count": len(suppliers),
            "message": "Suppliers retrieved successfully.",
            "suppliers": suppliers,
        }
    except Exception as exc:
        return {
            "success": False,
            "count": 0,
            "message": f"Failed to fetch suppliers: {exc}",
            "suppliers": [],
        }


@tool
def search_suppliers(
    supplier_name: str | None = None,
    business_name: str | None = None,
    business_category: str | None = None,
    business_type: str | None = None,
    city: str | None = None,
    state: str | None = None,
    country: str | None = None,
    verification_status: str | None = None,
    onboarding_completed: bool | None = None,
    years_in_business: int | None = None,
    limit: int = 10,
    sort_by: str | None = None,
):
    """
    Use this tool to search, recommend, compare, or filter suppliers.

    Call it when the user asks to:
    - search suppliers
    - recommend suppliers
    - compare suppliers
    - find suppliers by category
    - find suppliers by city or state
    - find verified suppliers
    - find experienced suppliers
    - find wholesalers or manufacturers

    The search is dynamic and applies only the filters the user provided.
    """
    client = _get_supabase_client()
    if client is None:
        return {
            "success": False,
            "count": 0,
            "message": "Supabase client is not configured.",
            "suppliers": [],
        }

    try:
        limit = max(1, min(int(limit), 20))
    except (TypeError, ValueError):
        limit = 10

    try:
        query = client.table("sellers").select(
            "id,business_name,owner_name,business_category,business_type,city,state,country,verification_status,onboarding_completed,years_in_business,phone,business_email,logo_url"
        )

        search_text = None
        if supplier_name:
            search_text = supplier_name
        elif business_name:
            search_text = business_name

        if search_text:
            query = query.or_(
                f"business_name.ilike.%{search_text}%,"
                f"owner_name.ilike.%{search_text}%,"
                f"business_category.ilike.%{search_text}%"
            )

        if business_category:
            query = query.ilike("business_category", f"%{business_category}%")

        if business_type:
            query = query.ilike("business_type", f"%{business_type}%")

        if city:
            query = query.ilike("city", f"%{city}%")

        if state:
            query = query.ilike("state", f"%{state}%")

        if country:
            query = query.ilike("country", f"%{country}%")

        if verification_status:
            query = query.ilike("verification_status", f"%{verification_status}%")

        if onboarding_completed is not None:
            query = query.eq("onboarding_completed", onboarding_completed)

        if years_in_business is not None:
            query = query.gte("years_in_business", years_in_business)

        if sort_by == "experience":
            query = query.order("years_in_business", desc=True)
        elif sort_by == "name":
            query = query.order("business_name", desc=False)
        elif sort_by == "verified":
            query = query.order("verification_status", desc=True)
        elif sort_by == "recent":
            query = query.order("id", desc=True)

        response = query.limit(limit).execute()
        rows = response.data or []
        suppliers = [_normalize_supplier(row) for row in rows]

        if not suppliers:
            return {
                "success": True,
                "found": False,
                "count": 0,
                "message": "No suppliers found matching your criteria.",
                "suppliers": [],
            }

        return {
            "success": True,
            "found": True,
            "count": len(suppliers),
            "message": "Suppliers retrieved successfully.",
            "suppliers": suppliers,
        }
    except Exception as exc:
        return {
            "success": False,
            "count": 0,
            "message": f"Database query failed: {exc}",
            "suppliers": [],
        }


@tool
def get_supplier_details(supplier_id: str):
    """
    Use this tool when the user wants complete information about a specific supplier.

    It is suitable for requests such as asking for contact details, address, business
    registration details, verification status, onboarding status, website, or media links
    for a known supplier ID.
    """
    client = _get_supabase_client()
    if client is None:
        return {
            "success": False,
            "message": "Supabase client is not configured.",
            "supplier": None,
        }

    try:
        response = client.table("sellers").select("*").eq("id", supplier_id).maybe_single().execute()
        row = response.data

        if not row:
            return {
                "success": True,
                "found": False,
                "message": f"No supplier found with id {supplier_id}.",
                "supplier": None,
            }

        return {
            "success": True,
            "found": True,
            "supplier": {
                "id": row.get("id"),
                "business_name": row.get("business_name"),
                "owner_name": row.get("owner_name"),
                "email": row.get("email"),
                "phone": row.get("phone"),
                "whatsapp": row.get("whatsapp"),
                "business_email": row.get("business_email"),
                "website": row.get("website"),
                "address": row.get("address"),
                "city": row.get("city"),
                "state": row.get("state"),
                "country": row.get("country"),
                "pincode": row.get("pincode"),
                "gst_number": row.get("gst_number"),
                "pan_number": row.get("pan_number"),
                "business_type": row.get("business_type"),
                "business_category": row.get("business_category"),
                "years_in_business": row.get("years_in_business"),
                "verification_status": row.get("verification_status"),
                "onboarding_completed": row.get("onboarding_completed"),
                "logo_url": row.get("logo_url"),
                "shop_image_url": row.get("shop_image_url"),
            },
        }
    except Exception as exc:
        return {
            "success": False,
            "message": f"Failed to fetch supplier details: {exc}",
            "supplier": None,
        }


@tool
def get_suppliers_by_ids(seller_ids: list[str], sort_by: str = "rating"):
    """
    Retrieve suppliers using seller IDs returned by the Product Agent.

    Use this tool when recommending suppliers, finding the best suppliers, comparing suppliers,
    ranking suppliers, or retrieving supplier information for products already returned by the
    Product Agent. This tool expects seller IDs from prior product search results and does not
    search by business name.

    Sorting modes:
    - rating: rating DESC, with verified sellers prioritized
    - reviews: review_count DESC, with verified sellers prioritized
    - experience: years_in_business DESC, with verified sellers prioritized
    - verified: verified sellers first, then rating/reviews
    """
    return _get_suppliers_by_ids_impl(seller_ids=seller_ids, sort_by=sort_by)