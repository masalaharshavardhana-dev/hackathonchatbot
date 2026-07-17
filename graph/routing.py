def execution_router(state):
    return {}


def route_execution(state):
    route = state.get("route")
    if route in {"end", "identity"}:
        return route

    plan = state.get("execution_plan", [])
    step = state.get("current_step", 0)

    if step >= len(plan):
        return "supervisor_summary"

    task = plan[step]
    return task.get("agent", "supervisor_summary")