from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition

from agents.orders_agent import orders_agent
from agents.product_agent import product_agent
from agents.supplier_agent import supplier_agent
from agents.supervisor_summary import supervisor_summary_agent
from graph.router import supervisor_agent
from graph.routing import execution_router, route_execution
from graph.state import AgentState
from tool_nodes import order_tool_node, product_tool_node, supplier_tool_node

builder = StateGraph(AgentState)

builder.add_node("supervisor_planner", supervisor_agent)
builder.add_node("execution_router", execution_router)
builder.add_node("product", product_agent)
builder.add_node("orders", orders_agent)
builder.add_node("supplier", supplier_agent)
builder.add_node("supervisor_summary", supervisor_summary_agent)

builder.add_node("product_tools", product_tool_node)
builder.add_node("order_tools", order_tool_node)
builder.add_node("supplier_tools", supplier_tool_node)

builder.set_entry_point("supervisor_planner")
builder.add_edge("supervisor_planner", "execution_router")

builder.add_conditional_edges(
    "execution_router",
    route_execution,
    {
        "product": "product",
        "orders": "orders",
        "supplier": "supplier",
        "end": END,
        "identity": END,
        "supervisor_summary": "supervisor_summary",
    },
)

builder.add_conditional_edges(
    "product",
    tools_condition,
    {
        "tools": "product_tools",
        END: "execution_router",
    },
)
builder.add_conditional_edges(
    "orders",
    tools_condition,
    {
        "tools": "order_tools",
        END: "execution_router",
    },
)

builder.add_conditional_edges(
    "supplier",
    tools_condition,
    {
        "tools": "supplier_tools",
        END: "execution_router",
    },
)

builder.add_edge("product_tools", "product")
builder.add_edge("order_tools", "orders")
builder.add_edge("supplier_tools", "supplier")
builder.add_edge("supervisor_summary", END)

graph = builder.compile()