from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm_with_tools


def _format_buyer_info(buyer_info: dict) -> str:
    from tools.buyer_tools import format_buyer_details

    return format_buyer_details(buyer_info)


def _is_error_payload(payload: object) -> bool:
    """True only for genuine system-error payloads (success == False)."""
    if isinstance(payload, dict):
        # Explicit success:False is a system error
        if payload.get("success") is False:
            return True
        # Legacy: dict has an "error" key but no success field (old format)
        if "error" in payload and "success" not in payload:
            return True
    if isinstance(payload, str):
        lowered = payload.lower()
        return "exception" in lowered or "traceback" in lowered
    return False


def _build_customer_summary_input(agent_results: dict) -> str:
    lines = []
    for agent_name, payload in sorted(agent_results.items()):
        if _is_error_payload(payload):
            continue
        lines.append(f"{agent_name}: {repr(payload)}")
    return "\n".join(lines)


def _format_money(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if number.is_integer():
        return f"₹{int(number)}"
    return f"₹{number:.2f}"


def _format_unit_price(value: object) -> str:
    formatted = _format_money(value)
    return f"{formatted}/unit" if formatted.startswith("₹") else formatted


def _get_supplier_rating(supplier: object) -> float:
    if not isinstance(supplier, dict):
        return 0.0
    try:
        return float(supplier.get("rating") or 0)
    except (TypeError, ValueError):
        return 0.0


def _get_supplier_name(supplier: object) -> str:
    if isinstance(supplier, dict):
        name = supplier.get("business_name") or supplier.get("name") or supplier.get("owner_name")
        if name:
            return str(name).strip()
    if isinstance(supplier, str):
        return supplier.strip()
    return "Unknown supplier"


def _get_supplier_location(supplier: object) -> str | None:
    if not isinstance(supplier, dict):
        return None
    location = supplier.get("location")
    if location:
        return str(location).strip()

    city = supplier.get("city")
    state = supplier.get("state")
    if city and state:
        return f"{city}, {state}"
    if city:
        return str(city)
    if state:
        return str(state)
    return None


def _format_supplier(supplier: object) -> str | None:
    if not isinstance(supplier, dict):
        return None

    name = supplier.get("name") or supplier.get("business_name") or supplier.get("owner_name")
    rating = supplier.get("rating")
    location = supplier.get("location")
    if not location:
        city = supplier.get("city")
        state = supplier.get("state")
        if city and state:
            location = f"{city}, {state}"
        elif city:
            location = city
        elif state:
            location = state

    parts = []
    if name:
        parts.append(str(name))
    if rating is not None:
        parts.append(f"rating {rating}")
    if location:
        parts.append(str(location))
    if supplier.get("verified") is True or str(supplier.get("verification_status") or "").lower() == "verified":
        parts.append("verified")

    return ", ".join(parts) if parts else None


def _sort_products_by_supplier_rating(products: list[dict]) -> list[dict]:
    return sorted(products, key=lambda product: _get_supplier_rating(product.get("supplier")), reverse=True)


def _group_products_by_supplier(products: list[dict]) -> list[tuple[str, str | None, list[dict]]]:
    grouped: list[tuple[str, str | None, list[dict]]] = []
    for product in products:
        supplier = product.get("supplier")
        supplier_name = _get_supplier_name(supplier)
        location = _get_supplier_location(supplier)

        existing_group = None
        for index, (group_name, group_location, group_products) in enumerate(grouped):
            if group_name == supplier_name and group_location == location:
                existing_group = index
                break

        if existing_group is None:
            grouped.append((supplier_name, location, [product]))
        else:
            grouped[existing_group][2].append(product)

    return grouped


def _format_analytics_result(product_payload: dict) -> str | None:
    if not isinstance(product_payload, dict):
        return None

    analytics_result = product_payload.get("analytics_result") or {}
    if not isinstance(analytics_result, dict):
        return None

    if analytics_result.get("success") is False:
        return analytics_result.get("message") or "I couldn't retrieve the requested inventory information right now."

    selected_tool = product_payload.get("selected_tool")
    if selected_tool == "get_total_product_count":
        return f"Total products available: {analytics_result.get('total_products', 0)}"

    if selected_tool == "get_inventory_summary":
        return (
            "Inventory Summary\n\n"
            f"• Total Categories: {analytics_result.get('total_categories', 0)}\n"
            f"• Total Subcategories: {analytics_result.get('total_subcategories', 0)}\n"
            f"• Total Products: {analytics_result.get('total_products', 0)}"
        )

    if selected_tool == "get_subcategory_statistics":
        subcategories = analytics_result.get("subcategories") or []
        if not subcategories:
            return "I couldn't find any subcategory statistics right now."
        lines = ["Subcategory Statistics", ""]
        for item in subcategories:
            lines.append(f"• {item.get('subcategory', 'Unknown')} – {item.get('product_count', 0)} products")
        lines.append("")
        lines.append(f"Total Products: {analytics_result.get('total_products', 0)}")
        return "\n".join(lines)

    categories = analytics_result.get("categories") or []
    if not categories:
        return "I couldn't find any category statistics right now."

    lines = ["Category Statistics", ""]
    for item in categories:
        lines.append(f"• {item.get('category', 'Unknown')} – {item.get('product_count', 0)} products")
    lines.append("")
    lines.append(f"Total Products: {analytics_result.get('total_products', 0)}")
    return "\n".join(lines)


def _format_product_result(product_payload: dict) -> str | None:
    if not isinstance(product_payload, dict):
        return None

    if product_payload.get("result_type") == "analytics":
        return _format_analytics_result(product_payload)

    selected_products = product_payload.get("selected_products") or []
    if not selected_products:
        product_search = product_payload.get("product_search")
        if isinstance(product_search, dict):
            # Check for system error first
            if product_search.get("success") is False:
                return "I'm having trouble retrieving products right now. Please try again in a few moments."
            # Check for no results found
            if product_search.get("found") is False:
                keyword = product_payload.get("extracted_keyword") or product_payload.get("original_query") or "your request"
                return f"I couldn't find any products matching '{keyword}'. Try a different search term or browse our categories."
            selected_products = product_search.get("products") or []
        else:
            result_payload = product_payload.get("result")
            if isinstance(result_payload, dict):
                if result_payload.get("success") is False:
                    return "I'm having trouble retrieving products right now. Please try again in a few moments."
                if result_payload.get("found") is False:
                    keyword = product_payload.get("extracted_keyword") or product_payload.get("original_query") or "your request"
                    return f"I couldn't find any products matching '{keyword}'. Try a different search term or browse our categories."
                selected_products = result_payload.get("products") or []

    if not selected_products:
        keyword = product_payload.get("extracted_keyword") or product_payload.get("original_query") or "your request"
        return f"I couldn't find any products matching '{keyword}'. Try a different search term or browse our categories."

    original_query = (product_payload.get("original_query") or "").lower()
    if any(k in original_query for k in ["single", "one item", "one product", "highest", "lowest", "cheapest", "most expensive", "costliest"]):
        selected_products = selected_products[:1]

    selected_products = _sort_products_by_supplier_rating([product for product in selected_products if isinstance(product, dict)])
    grouped_products = _group_products_by_supplier(selected_products[:10])

    total_count = len(selected_products)
    if len(grouped_products) == 1:
        supplier_name, location, products = grouped_products[0]
        header = f"I found {total_count} product{'s' if total_count != 1 else ''} from {supplier_name} matching your request."
        if location:
            header = f"I found {total_count} product{'s' if total_count != 1 else ''} from {supplier_name} in {location} matching your request."
        lines = [header, ""]
        for index, product in enumerate(products, start=1):
            name = product.get("name") or "Unnamed product"
            brand = product.get("brand")
            price = product.get("wholesale_price")
            delivery_days = product.get("delivery_days")

            lines.append(f"{index}. {name}")
            if price is not None:
                lines.append(f"   • Price: {_format_unit_price(price)}")
            if brand:
                lines.append(f"   • Brand: {brand}")
            lines.append(f"   • Supplier: {supplier_name}")
            if delivery_days is not None:
                lines.append(f"   • Delivery: {delivery_days} days")
            lines.append("")

        return "\n".join(line for line in lines if line is not None).strip()

    lines = [f"I found {total_count} product{'s' if total_count != 1 else ''} matching your request.", ""]
    global_index = 1
    for supplier_name, location, products in grouped_products:
        supplier_header = supplier_name
        if location:
            supplier_header = f"{supplier_name} - {location}"
        lines.append(f"{supplier_header}:")
        for product in products:
            name = product.get("name") or "Unnamed product"
            brand = product.get("brand")
            price = product.get("wholesale_price")
            delivery_days = product.get("delivery_days")

            lines.append(f"{global_index}. {name}")
            if price is not None:
                lines.append(f"   • Price: {_format_unit_price(price)}")
            if brand:
                lines.append(f"   • Brand: {brand}")
            lines.append(f"   • Supplier: {supplier_name}")
            if delivery_days is not None:
                lines.append(f"   • Delivery: {delivery_days} days")
            lines.append("")
            global_index += 1

    return "\n".join(line for line in lines if line is not None).strip()


def _format_money(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if number.is_integer():
        return f"₹{int(number):,}"
    return f"₹{number:,.2f}"


def _format_order_item(item: dict) -> str | None:
    if not isinstance(item, dict):
        return None

    product_name = item.get("product_name") or "Unnamed product"
    quantity = item.get("quantity")
    unit_price = item.get("unit_price")
    line_total = item.get("line_total")

    parts = [product_name]
    if quantity is not None:
        parts.append(f"Qty: {quantity}")
    if unit_price is not None:
        parts.append(f"Unit price: {_format_money(unit_price)}")
    if line_total is not None:
        parts.append(f"Line total: {_format_money(line_total)}")

    return " • ".join(parts)


def _format_order_result(order_payload: dict) -> str | None:
    if not isinstance(order_payload, dict):
        return None

    result = order_payload.get("result") or {}
    if not isinstance(result, dict):
        return None

    # Case 1: System error (DB down, network issue, exception)
    if result.get("success") is False:
        return (
            "I'm having trouble retrieving your orders right now. "
            "Please try again in a few moments."
        )

    # Case 2: Query succeeded but no orders exist for this buyer
    if result.get("found") is False:
        return "It looks like you haven't placed any orders yet."

    # Case 3: Query succeeded and orders were found

    order = result.get("order")
    orders = result.get("orders") or []

    if isinstance(order, dict):
        order_number = order.get("order_number") or "Your order"
        status = order.get("status")
        payment_status = order.get("payment_status")
        payment_method = order.get("payment_method")
        grand_total = order.get("grand_total")
        estimated_delivery = order.get("estimated_delivery")
        tracking_number = order.get("tracking_number")
        delivery_partner = order.get("delivery_partner")
        shipping_address = order.get("shipping_address")
        items = order.get("items") or []

        lines = [f"Order {order_number}"]
        if status:
            lines.append(f"Status: {status}")
        if payment_status:
            lines.append(f"Payment Status: {payment_status}")
        if payment_method:
            lines.append(f"Payment Method: {payment_method}")
        if grand_total is not None:
            lines.append(f"Total Amount: {_format_money(grand_total)}")
        if estimated_delivery:
            lines.append(f"Estimated Delivery: {estimated_delivery}")
        if delivery_partner:
            lines.append(f"Delivery Partner: {delivery_partner}")
        if tracking_number:
            lines.append(f"Tracking: {tracking_number}")
        if shipping_address:
            lines.append(f"Shipping Address: {shipping_address}")

        if items:
            lines.append("")
            lines.append("Items:")
            for index, item in enumerate(items, start=1):
                formatted_item = _format_order_item(item)
                if formatted_item:
                    lines.append(f"{index}. {formatted_item}")

        return "\n".join(lines).strip()

    if orders:
        lines = [f"I found {len(orders)} orders matching your request.", ""]
        for index, order_item in enumerate(orders, start=1):
            if not isinstance(order_item, dict):
                continue

            order_number = order_item.get("order_number") or "Unknown order"
            status = order_item.get("status")
            payment_status = order_item.get("payment_status")
            grand_total = order_item.get("grand_total")
            estimated_delivery = order_item.get("estimated_delivery")
            items = order_item.get("items") or []

            lines.append(f"{index}. Order {order_number}")
            if status:
                lines.append(f"   • Status: {status}")
            if payment_status:
                lines.append(f"   • Payment Status: {payment_status}")
            if grand_total is not None:
                lines.append(f"   • Total Amount: {_format_money(grand_total)}")
            if estimated_delivery:
                lines.append(f"   • Estimated Delivery: {estimated_delivery}")

            if items:
                first_item = items[0]
                if isinstance(first_item, dict):
                    product_name = first_item.get("product_name")
                    if product_name:
                        lines.append(f"   • Product: {product_name}")
            lines.append("")

        return "\n".join(lines).strip()

    return None


def _format_supplier_result(supplier_payload: dict) -> str | None:
    """Format supplier results with 3-case dispatch."""
    if not isinstance(supplier_payload, dict):
        return None

    supplier_result = supplier_payload.get("supplier_result") or supplier_payload.get("result") or {}
    if not isinstance(supplier_result, dict):
        return None

    # Case 1: System error
    if supplier_result.get("success") is False:
        return (
            "I'm having trouble retrieving supplier information right now. "
            "Please try again in a few moments."
        )

    # Case 2: Query succeeded but no suppliers found
    if supplier_result.get("found") is False:
        return "I couldn't find any suppliers matching your request."

    # Case 3: Suppliers found — format them
    suppliers = supplier_result.get("suppliers") or []
    if not suppliers:
        return "I couldn't find any suppliers matching your request."

    count = len(suppliers)
    lines = [f"I found {count} supplier{'s' if count != 1 else ''} matching your request.", ""]
    for index, supplier in enumerate(suppliers, start=1):
        if not isinstance(supplier, dict):
            continue
        name = supplier.get("business_name") or supplier.get("owner_name") or "Unknown supplier"
        city = supplier.get("city")
        state = supplier.get("state")
        rating = supplier.get("rating")
        verification = supplier.get("verification_status")
        years = supplier.get("years_in_business")

        lines.append(f"{index}. {name}")
        if city or state:
            location = ", ".join(part for part in [city, state] if part)
            lines.append(f"   • Location: {location}")
        if rating is not None:
            lines.append(f"   • Rating: {rating}")
        if verification:
            lines.append(f"   • Status: {verification}")
        if years is not None:
            lines.append(f"   • Experience: {years} years")
        lines.append("")

    return "\n".join(lines).strip()


def supervisor_summary_agent(state):
    agent_results = state.get("agent_results", {})
    responses = []

    # --- Buyer Profile ---
    buyer_info = agent_results.get("buyer_info") or agent_results.get("buyer")
    if isinstance(buyer_info, dict):
        if buyer_info.get("success") is False:
            responses.append("I'm having trouble retrieving your profile right now. Please try again in a few moments.")
        elif buyer_info.get("found") is False:
            responses.append("I couldn't find your profile information. Please contact support.")
        else:
            # Legacy format (no success key) or success:True,found:True
            formatted_details = _format_buyer_info(buyer_info)
            responses.append(f"Here are your details:\n{formatted_details}")

    # --- Product Results ---
    product_info = agent_results.get("product")
    if isinstance(product_info, dict):
        formatted_products = _format_product_result(product_info)
        if formatted_products:
            responses.append(formatted_products)

    # --- Order Results ---
    order_info = agent_results.get("orders")
    if isinstance(order_info, dict):
        formatted_orders = _format_order_result(order_info)
        if formatted_orders:
            responses.append(formatted_orders)

    # --- Supplier Results ---
    supplier_info = agent_results.get("supplier")
    if isinstance(supplier_info, dict):
        formatted_suppliers = _format_supplier_result(supplier_info)
        if formatted_suppliers:
            responses.append(formatted_suppliers)

    if responses:
        final_text = "\n\n".join(responses)
        return {"final_response": final_text, "messages": [HumanMessage(content=final_text)]}

    result_text = "\n".join(
        f"{agent_name}: {repr(payload)}" for agent_name, payload in sorted(agent_results.items())
    )
    customer_result_text = _build_customer_summary_input(agent_results)

    llm = get_llm_with_tools([])
    response = llm.invoke(
        [
            SystemMessage(content="""
You are a customer-facing wholesale assistant.

Write a short, direct response using only the successful data provided.
Do not mention internal steps, tools, database queries, failed lookups, errors, or exceptions.
Do not apologize for missing backend data.
If some information is unavailable, simply omit it.
Return only the useful customer-facing result.
""".strip()),
            HumanMessage(content=f"Agent results:\n{customer_result_text or result_text}"),
        ]
    )

    return {"final_response": response.content, "messages": [response]}
