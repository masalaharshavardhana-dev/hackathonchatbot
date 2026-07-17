from langgraph.prebuilt import ToolNode
from tools.order_tools import (
    search_orders,
    get_order_details,
)
from tools.product_tools import search_products, search_subcategory
from tools.supplier_tools import (
    search_suppliers,
    get_supplier_details,
)


product_tool_node = ToolNode([
    search_subcategory,
    search_products
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