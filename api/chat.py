from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import traceback

from graph.graph import graph
from auth.auth import get_current_buyer
from context.buyer_context import set_buyer_id

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
def chat(request: ChatRequest, buyer_data: dict = Depends(get_current_buyer)):
    """
    Chat endpoint - Requires authentication token in Authorization header
    """
    try:
        buyer_id = buyer_data["buyer_id"]
        buyer_profile = buyer_data.get("profile", {})
        buyer_name = (
            buyer_data.get("buyer_name")
            or buyer_profile.get("full_name")
            or buyer_profile.get("name")
            or buyer_profile.get("business_name")
        )
        
        # Set the buyer context for the current request
        set_buyer_id(buyer_id)
        
        state = {
            "messages": [HumanMessage(content=request.message)],
            "buyer_id": buyer_id,
            "buyer_name": buyer_name,
            "execution_plan": [],
            "current_step": 0,
            "agent_results": {},
            "final_response": ""
        }
        result = graph.invoke(state)
        return result
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@router.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "message": "Backend is running"}