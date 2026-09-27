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
    # TODO: Implement detection logic
    # For now, route everything to Path B (safe default)
    return False
