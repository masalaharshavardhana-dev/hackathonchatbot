import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm_with_tools
from tools.buyer_tools import format_buyer_details, get_buyer_details

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """
You are the Supervisor Planner for a B2B wholesale marketplace assistant.

Your only job is to classify the user's intent and produce a routing plan.
You must return only valid JSON — no explanation, no markdown, no extra text.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTENT CLASSIFICATION (use in priority order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIORITY 1 — ORDER INTENT
Route to agent "orders" when the user asks about THEIR OWN orders, purchases, or order history.

Triggers (any of these = Order Intent):
- "my orders", "my order", "my purchases", "my previous orders"
- "order history", "order status", "order details", "order summary"
- "have I placed", "what did I buy", "what have I ordered", "what I ordered"
- "show my orders", "list my orders", "view my orders"
- "track my order", "track order"
- "delivered orders", "pending orders", "cancelled orders", "recent orders"
- "last order", "previous order", "past order"
- "Have I ordered [product]", "Show my [product] orders"

CRITICAL: The presence of a product name (rice, coffee, oil, etc.) inside an order question
does NOT change the intent. "Show my coffee orders" → orders agent, NOT product agent.

PRIORITY 2 — RFQ INTENT
Route to agent "rfq" when the user wants to:
- Create a quotation or request a quote
- View their RFQ history or quotation status
- "my RFQs", "my quotations", "create RFQ", "send quotation"

PRIORITY 3 — SUPPLIER INTENT
Route to agent "supplier" when the user asks about suppliers themselves.
- "find suppliers", "show suppliers", "best supplier for X"
- "compare suppliers", "top-rated suppliers", "supplier details"
- "verified suppliers", "suppliers in [city/state]"

PRIORITY 4 — PRODUCT INTENT
Route to agent "product" ONLY when the user wants to discover or browse products
and none of the higher-priority intents apply.
- "show me X", "find X", "search for X", "I need X", "I want X"
- "product availability", "product price", "compare products"
- "recommend products", "featured products"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMATS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Format 1 — Greeting (only pure greetings: hi, hello, hey, namaste, good morning):
{
    "route": "end",
    "intent": "greeting",
    "reason": "User sent a greeting.",
    "final_response": "Hi [buyer name], how can I help you today?"
}

Format 2 — Identity (user asks who they are or what their name is):
{
    "route": "identity",
    "intent": "identity",
    "reason": "User asked about their identity.",
    "final_response": "[buyer name]"
}

Format 3 — Planning (everything else):
{
    "route": "plan",
    "intent": "orders",
    "reason": "The user is asking about their order history.",
    "tasks": [
        {"agent": "orders", "goal": "Retrieve the buyer's order history"}
    ]
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MULTI-AGENT PLANNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If a query clearly requires multiple agents, list tasks in execution order.

Example: "I ordered coffee. Show the supplier."
→ tasks: [{"agent": "orders", "goal": "..."}, {"agent": "supplier", "goal": "..."}]

Example: "I want coffee powder and create an RFQ."
→ tasks: [{"agent": "product", "goal": "..."}, {"agent": "rfq", "goal": "..."}]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Always include "intent" and "reason" fields in the JSON response.
- Use Priority 1 before 2 before 3 before 4. Never skip priority checks.
- Return only JSON. No markdown code fences. No prose.
- Never route order queries to the product agent.
- The buyer name is provided in the context — use it in greetings.
""".strip()


def _log_routing_decision(user_query: str, intent: str, agent: str, reason: str) -> None:
    """Print a formatted routing decision to the console for debugging."""
    border = "─" * 55
    print(f"\n┌{border}┐")
    print(f"│  SUPERVISOR ROUTING DECISION")
    print(f"├{border}┤")
    print(f"│  User Query  : {user_query[:80]!r}")
    print(f"│  Intent      : {intent}")
    print(f"│  Agent       : {agent}")
    print(f"│  Reason      : {reason[:120]}")
    print(f"└{border}┘\n")
    logger.debug(
        "supervisor_routing user_query=%r intent=%s agent=%s reason=%s",
        user_query,
        intent,
        agent,
        reason,
    )


def supervisor_agent(state):
    user_query = ""
    buyer_name = state.get("buyer_name")
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            user_query = message.content
            break

    # Inline shortcut: buyer profile queries (no LLM needed)
    buyer_detail_keywords = ["details of me", "my details", "about me", "tell me about me"]
    if any(keyword in user_query.lower() for keyword in buyer_detail_keywords):
        buyer_details = get_buyer_details()
        # Use new success/found contract for the conversation message
        if buyer_details.get("success") is False:
            message_text = "I'm having trouble retrieving your profile right now."
        elif buyer_details.get("found") is False:
            message_text = "I couldn't find your profile information."
        else:
            message_text = f"Here are your details:\n{format_buyer_details(buyer_details)}"

        _log_routing_decision(user_query, "buyer_profile", "buyer", "User asked for their own profile details.")
        return {
            "messages": [HumanMessage(content=message_text)],
            "execution_plan": [],
            "current_step": 0,
            "agent_results": {"buyer_info": buyer_details},
            "route": "identity",
            "final_response": "",
        }

    llm = get_llm_with_tools([])
    response = llm.invoke(
        [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=f"Buyer name: {buyer_name or 'Unknown'}\nUser message: {user_query}"),
        ]
    )

    raw_content = response.content.strip()
    # Strip markdown fences if the LLM wraps output in ```json ... ```
    if raw_content.startswith("```"):
        lines = raw_content.splitlines()
        raw_content = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    parsed_content = json.loads(raw_content)
    route = parsed_content.get("route", "plan")
    intent = parsed_content.get("intent", "unknown")
    reason = parsed_content.get("reason", "")
    execution_plan = parsed_content.get("tasks", [])

    # Determine the first agent for logging
    first_agent = execution_plan[0].get("agent", "unknown") if execution_plan else route
    _log_routing_decision(user_query, intent, first_agent, reason)

    if route == "end":
        return {
            "messages": [response],
            "execution_plan": [{"agent": "end", "goal": "End the conversation"}],
            "current_step": 0,
            "agent_results": {},
            "final_response": parsed_content.get("final_response", response.content),
            "route": "end",
        }

    if route == "identity":
        return {
            "messages": [response],
            "execution_plan": [{"agent": "end", "goal": "Return the buyer name"}],
            "current_step": 0,
            "agent_results": {},
            "final_response": (parsed_content.get("final_response") or buyer_name or ""),
            "route": "identity",
        }

    return {
        "messages": [response],
        "execution_plan": execution_plan,
        "current_step": 0,
        "agent_results": {},
        "route": "plan",
        "final_response": "",
    }