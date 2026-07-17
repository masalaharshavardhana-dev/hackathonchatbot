from database.supabase_client import supabase
from context.buyer_context import get_buyer_id

BUYER_FIELD_LABELS = {
    "name": "Name",
    "company": "Company",
    "email": "Email",
    "phone": "Phone",
    "address": "Address",
    "whatsapp": "WhatsApp",
    "shipping_address": "Shipping address",
}


def _clean_buyer_data(data: dict) -> dict:
    return {
        key: value
        for key, value in data.items()
        if value not in (None, "", [], {}, ())
    }


def format_buyer_details(buyer_details: dict, include_buyer_id: bool = False) -> str:
    if buyer_details.get("error"):
        return str(buyer_details["error"])

    cleaned_details = _clean_buyer_data(buyer_details)
    ordered_keys = [
        "name",
        "company",
        "email",
        "phone",
        "address",
        "whatsapp",
        "shipping_address",
    ]

    labels = dict(BUYER_FIELD_LABELS)
    if include_buyer_id and cleaned_details.get("buyer_id"):
        ordered_keys = ["buyer_id", *ordered_keys]
        labels["buyer_id"] = "Buyer ID"

    lines = []
    for key in ordered_keys:
        value = cleaned_details.get(key)
        if value is None:
            continue

        if isinstance(value, (dict, list)):
            import json

            value_text = json.dumps(value, ensure_ascii=False, indent=2)
        else:
            value_text = str(value)

        label = labels.get(key, key.replace("_", " ").title())
        lines.append(f"{label}: {value_text}")

    return "\n".join(lines) if lines else "No buyer details found."


def _is_buyer_error(buyer_details: dict) -> bool:
    """Return True only for genuine system errors (not 'not found')."""
    if not isinstance(buyer_details, dict):
        return False
    return buyer_details.get("success") is False or "error" in buyer_details


def _is_buyer_not_found(buyer_details: dict) -> bool:
    """Return True when query succeeded but buyer profile does not exist."""
    if not isinstance(buyer_details, dict):
        return False
    return buyer_details.get("success") is True and buyer_details.get("found") is False

def get_buyer_details():
    """Get the current buyer's details"""
    buyer_id = get_buyer_id()

    try:
        response = supabase.table("buyers").select("*").eq("id", buyer_id).maybe_single().execute()
        if response and response.data:
            buyer_data = {
                "buyer_id": response.data.get("id"),
                "name": response.data.get("full_name") or response.data.get("name"),
                "company": response.data.get("business_name") or response.data.get("company"),
                "email": response.data.get("email"),
                "phone": response.data.get("phone"),
                "address": response.data.get("address"),
                "whatsapp": response.data.get("whatsapp"),
                "shipping_address": response.data.get("shipping_address"),
            }
            cleaned = _clean_buyer_data(buyer_data)
            return {"success": True, "found": True, **cleaned}

        return {
            "success": True,
            "found": False,
            "message": "Buyer profile not found.",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch buyer details: {str(e)}"}
