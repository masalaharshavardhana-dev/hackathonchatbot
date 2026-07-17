from langchain_core.messages import HumanMessage, SystemMessage
from tools.buyer_tools import format_buyer_details, get_buyer_details


BUYER_AGENT_PROMPT = """
You are the Buyer Assistant.

Rules:
- Only work with the currently authenticated buyer.
- Never fetch or display all buyers from the buyers table.
- Only use the buyer_id from the current request context.
- If buyer details are needed, rely on the scoped buyer lookup only.
"""

def buyer_agent(state):
    """Agent to handle buyer-related queries"""
    buyer_details = get_buyer_details()

    # Build a human-readable message for the conversation log
    if buyer_details.get("success") is False:
        message_text = "I'm having trouble retrieving your profile right now."
    elif buyer_details.get("found") is False:
        message_text = "I couldn't find your profile information."
    else:
        message_text = f"Here are your details:\n{format_buyer_details(buyer_details)}"

    return {
        "messages": [
            SystemMessage(content=BUYER_AGENT_PROMPT),
            HumanMessage(content=message_text),
        ],
        "agent_results": {
            **state.get("agent_results", {}),
            "buyer": buyer_details,
        },
        "current_step": state["current_step"] + 1,
    }
