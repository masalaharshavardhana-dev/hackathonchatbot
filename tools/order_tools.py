from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from database.supabase_client import supabase


def _get_supabase_client():
    return supabase


def _validate_limit(limit: int | None) -> int:
    if limit is None:
        return 10
    try:
        parsed_limit = int(limit)
    except (TypeError, ValueError):
        return 10
    return max(1, min(parsed_limit, 50))


def _normalize_order(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_number": row.get("order_number"),
        "status": row.get("status"),
        "subtotal": row.get("subtotal"),
        "grand_total": row.get("grand_total"),
        "payment_status": row.get("payment_status"),
        "payment_method": row.get("payment_method"),
        "estimated_delivery": row.get("estimated_delivery"),
        "tracking_number": row.get("tracking_number"),
        "delivery_partner": row.get("delivery_partner"),
        "shipping_address": row.get("shipping_address"),
        "created_at": row.get("created_at"),
    }


def _extract_product_name(product_snapshot: Any) -> str | None:
    if isinstance(product_snapshot, dict):
        return product_snapshot.get("name") or product_snapshot.get("product_name") or product_snapshot.get("title")

    if isinstance(product_snapshot, str):
        try:
            parsed = json.loads(product_snapshot)
            if isinstance(parsed, dict):
                return parsed.get("name") or parsed.get("product_name") or parsed.get("title")
        except Exception:
            return None

    return None


def _normalize_order_item(row: dict[str, Any]) -> dict[str, Any]:
    product_snapshot = row.get("product_snapshot")
    product_name = _extract_product_name(product_snapshot)

    item = {
        "product_name": product_name,
        "product_brand": None,
        "quantity": row.get("quantity"),
        "unit_price": row.get("unit_price"),
        "gst_rate": row.get("gst_rate"),
        "gst_amount": row.get("gst_amount"),
        "discount_amount": row.get("discount_amount"),
        "line_total": row.get("line_total"),
    }

    if isinstance(product_snapshot, dict):
        item["product_brand"] = product_snapshot.get("brand")
    elif isinstance(product_snapshot, str):
        try:
            parsed = json.loads(product_snapshot)
            if isinstance(parsed, dict):
                item["product_brand"] = parsed.get("brand")
        except Exception:
            pass

    return {key: value for key, value in item.items() if value not in (None, "", [], {}, ())}


def _sort_orders(rows: list[dict[str, Any]], sort_by: str | None) -> list[dict[str, Any]]:
    if sort_by == "oldest":
        return sorted(rows, key=lambda row: row.get("created_at") or "")
    if sort_by == "amount_high":
        return sorted(rows, key=lambda row: row.get("grand_total") or 0, reverse=True)
    if sort_by == "amount_low":
        return sorted(rows, key=lambda row: row.get("grand_total") or 0)
    return sorted(rows, key=lambda row: row.get("created_at") or "", reverse=True)


def _search_orders_impl(
    buyer_id: str,
    status: str | None = None,
    payment_status: str | None = None,
    payment_method: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 10,
    sort_by: str | None = "recent",
):
    client = _get_supabase_client()
    if client is None:
        return {
            "success": False,
            "error": "Supabase client is not configured.",
            "count": 0,
            "orders": [],
        }

    buyer_id = (buyer_id or "").strip()
    if not buyer_id:
        return {
            "success": False,
            "error": "Unauthorized access. Buyer ID not found.",
            "count": 0,
            "orders": [],
        }

    limit = _validate_limit(limit)

    try:
        query = client.table("orders").select(
            "id,order_number,user_id,status,subtotal,grand_total,payment_status,payment_method,estimated_delivery,tracking_number,delivery_partner,shipping_address,created_at"
        ).eq("user_id", buyer_id)

        if status:
            query = query.ilike("status", f"%{status}%")
        if payment_status:
            query = query.ilike("payment_status", f"%{payment_status}%")
        if payment_method:
            query = query.ilike("payment_method", f"%{payment_method}%")
        if from_date:
            query = query.gte("created_at", from_date)
        if to_date:
            query = query.lte("created_at", to_date)

        query = query.order("created_at", desc=(sort_by != "oldest"))
        if sort_by == "amount_high":
            query = query.order("grand_total", desc=True)
        elif sort_by == "amount_low":
            query = query.order("grand_total", desc=False)
        elif sort_by == "recent":
            query = query.order("created_at", desc=True)
        elif sort_by == "oldest":
            query = query.order("created_at", desc=False)

        response = query.limit(limit).execute()
        order_rows = response.data or []

        if not order_rows:
            return {
                "success": True,
                "found": False,
                "count": 0,
                "message": "No matching orders found.",
                "orders": [],
            }

        orders: list[dict[str, Any]] = []
        for order_row in order_rows:
            order_id = order_row.get("id")
            items_response = client.table("order_items").select(
                "order_id,product_id,product_snapshot,quantity,unit_price,gst_rate,gst_amount,discount_amount,line_total,seller_id,buyer_id,created_at"
            ).eq("order_id", order_id).execute()
            item_rows = items_response.data or []

            order = _normalize_order(order_row)
            order["items"] = [_normalize_order_item(item_row) for item_row in item_rows]
            orders.append(order)

        orders = _sort_orders(orders, sort_by)

        return {
            "success": True,
            "found": True,
            "count": len(orders),
            "message": "Orders retrieved successfully.",
            "orders": orders,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Database error: {exc}",
            "count": 0,
            "orders": [],
        }


def _get_order_details_impl(buyer_id: str, order_number: str):
    client = _get_supabase_client()
    if client is None:
        return {
            "success": False,
            "error": "Supabase client is not configured.",
            "order": None,
        }

    buyer_id = (buyer_id or "").strip()
    order_number = (order_number or "").strip()
    if not buyer_id:
        return {
            "success": False,
            "error": "Unauthorized access. Buyer ID not found.",
            "order": None,
        }
    if not order_number:
        return {
            "success": False,
            "error": "Order number is required.",
            "order": None,
        }

    try:
        response = client.table("orders").select(
            "id,order_number,user_id,status,subtotal,grand_total,payment_status,payment_method,estimated_delivery,tracking_number,delivery_partner,shipping_address,created_at"
        ).eq("user_id", buyer_id).eq("order_number", order_number).maybe_single().execute()
        order_row = response.data

        if not order_row:
            return {
                "success": True,
                "found": False,
                "message": f"Order {order_number} not found or you don't have permission to access it.",
                "order": None,
            }

        items_response = client.table("order_items").select(
            "order_id,product_id,product_snapshot,quantity,unit_price,gst_rate,gst_amount,discount_amount,line_total,seller_id,buyer_id,created_at"
        ).eq("order_id", order_row.get("id")).execute()
        item_rows = items_response.data or []

        order = _normalize_order(order_row)
        order.update(
            {
                "items": [_normalize_order_item(item_row) for item_row in item_rows],
            }
        )

        return {
            "success": True,
            "found": True,
            "message": "Order retrieved successfully.",
            "order": order,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Database error: {exc}",
            "order": None,
        }


@tool
def search_orders(
    buyer_id: str,
    status: str | None = None,
    payment_status: str | None = None,
    payment_method: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 10,
    sort_by: str | None = "recent",
):
    """
    Retrieve the authenticated buyer's orders together with each order's items.

    Use this tool for requests such as:
    - my orders
    - recent orders
    - order history
    - delivered orders
    - pending orders
    - cancelled orders

    The buyer_id must come from the authenticated backend state. Do not ask the user for it.
    This tool always filters by user_id == buyer_id before applying any other filters.

    Sorting modes:
    - recent
    - oldest
    - amount_high
    - amount_low
    """
    return _search_orders_impl(
        buyer_id=buyer_id,
        status=status,
        payment_status=payment_status,
        payment_method=payment_method,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
        sort_by=sort_by,
    )


@tool
def get_order_details(buyer_id: str, order_number: str):
    """
    Retrieve the authenticated buyer's full order details together with all order items.

    Use this tool for requests such as:
    - show order details
    - track order
    - order summary
    - invoice information

    The buyer_id must come from the authenticated backend state. The tool verifies that the order
    belongs to the current buyer, then returns the order and all items without exposing internal IDs.
    """
    return _get_order_details_impl(buyer_id=buyer_id, order_number=order_number)