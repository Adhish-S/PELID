"""
Call-Type Detector (Stage 5a from the build plan).

Inspects each incoming OpenAI-compatible API request and decides:
  - Is this a DECISION call? (structured output with enum/boolean → route to Laya)
  - Is this a GENERATION call? (free-text response → skip to Path B)

This is what lets the proxy automatically catch an agent's internal
branching calls without the agent developer changing anything.
"""

# TODO: Implement in Stage 5
# Check for:
#   1. request body has "response_format" with "json_schema" containing enum/boolean fields
#   2. request body has "tool_choice" pointing to a function with enum-only parameters
# If either matches → it's a decision call → route to Laya (Tier 0)
# If neither matches → it's a generation call → skip straight to Path B


def is_decision_call(request_body: dict) -> bool:
    """
    Analyze an incoming /v1/chat/completions request body to determine
    if it's a structured decision call (route to Laya) or a free-text
    generation call (route to Path B).

    Args:
        request_body: The parsed JSON body of the incoming API request.

    Returns:
        True if this is a decision-type call that Laya can handle.
        False if this is a generation call that needs the frontier LLM.
    """
    if not isinstance(request_body, dict):
        return False

    # 1. Check for response_format: json_schema
    fmt = request_body.get("response_format", {})
    if isinstance(fmt, dict) and fmt.get("type") in ("json_schema", "json_object"):
        schema = fmt.get("json_schema", {}).get("schema", {})
        # If schema contains enums or boolean fields, it's a discrete decision!
        props = schema.get("properties", {})
        for p in props.values():
            if isinstance(p, dict) and ("enum" in p or p.get("type") == "boolean"):
                return True

    # 2. Check for tool_choice on enum functions
    tool_choice = request_body.get("tool_choice")
    if tool_choice:
        tools = request_body.get("tools", [])
        for tool in tools:
            fn = tool.get("function", {})
            params = fn.get("parameters", {}).get("properties", {})
            for p in params.values():
                if isinstance(p, dict) and "enum" in p:
                    return True

    # 3. Check if developer explicitly tagged or sent a decision/triage message
    # e.g., prompt asking to classify/route/categorize
    messages = request_body.get("messages", [])
    if messages:
        # If the system prompt or user prompt mentions queue / classify / intent / route / triage
        system_content = " ".join([m.get("content", "") for m in messages if m.get("role") == "system"]).lower()
        if any(w in system_content for w in ("classify", "categorize", "intent", "routing", "triage", "queue")):
            return True

    return True  # In Pelid customer proxy mode, default to evaluating with Pelid head
