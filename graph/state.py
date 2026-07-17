from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    buyer_id: str
    buyer_name: str | None
    route: str | None
    execution_plan: list
    current_step: int
    agent_results: dict
    final_response: str

def merge_agent_results(existing_results: dict, agent_name: str, agent_result: dict) -> dict:
    results = dict(existing_results or {})
    results[agent_name] = agent_result
    return results