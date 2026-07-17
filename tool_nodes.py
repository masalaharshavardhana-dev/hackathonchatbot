from langgraph.prebuilt import ToolNode
from tools.order_tools import (
    search_orders,
    get_order_details,
)
from tools.product_tools import (
    get_category_statistics,
    get_inventory_summary,
    get_subcategory_statistics,
    get_total_product_count,
    search_products,
    search_subcategory,
    extract_filters,
    extract_core_query,
)
from tools.supplier_tools import (
    search_suppliers,
    get_supplier_details,
)


product_tool_node = ToolNode([
    search_subcategory,
    search_products,
    get_total_product_count,
    get_category_statistics,
    get_subcategory_statistics,
    get_inventory_summary,
    extract_filters,
    extract_core_query,
])
order_tool_node = ToolNode(
    [
        search_orders,
        get_order_details,
    ]
)

supplier_tool_node = ToolNode(
    [
        search_suppliers,
        get_supplier_details,
    ]
)