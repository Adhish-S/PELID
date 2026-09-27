"""
Core Proxy Server (Stage 6 from the build plan).

This is the main entry point — a FastAPI server that:
1. Listens on localhost:8080
2. Intercepts /v1/chat/completions requests
3. Runs them through the decision pipeline:
   call-type detector → context budget → language router → Laya → Path A or B
4. Returns the response

Developers use it by changing one line in their code:
   client = OpenAI(base_url="http://localhost:8080/v1")
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

from pelid.config import UPSTREAM_BASE_URL, GEMINI_API_KEY, UPSTREAM_MODEL, HOST, PORT
from pelid.call_detector import is_decision_call
from pelid.context_budget import is_within_budget
from pelid.lang_router import detect_language
from pelid.decision import run_decision

app = FastAPI(
    title="Pelid",
    description="AI decision proxy — intercepts lightweight LLM calls and resolves them locally.",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Main proxy endpoint — intercepts OpenAI-compatible chat completion requests.

    Pipeline:
    1. Is this a decision call? (call-type detector)
       No  → Path B (forward to frontier LLM)
    2. Is the context within budget? (< 450 tokens)
       No  → Path B
    3. Detect language → pick Laya checkpoint
    4. Run Laya → get decision + confidence
    5. Confident + not destructive → Path A (return local answer)
       Otherwise → Path B (forward to frontier LLM)
    """
    body = await request.json()

    # Step 1: Is this a decision-type call?
    if not is_decision_call(body):
        return await forward_to_upstream(body, request)

    # Step 2: Extract the relevant text and check context budget
    messages = body.get("messages", [])
    last_message_content = messages[-1].get("content", "") if messages else ""

    if not is_within_budget(last_message_content):
        return await forward_to_upstream(body, request)

    # Step 3: Detect language
    language = detect_language(last_message_content)

    # Step 4: Run Laya decision
    result = await run_decision(last_message_content, language)

    # Step 5: Route based on confidence + destructive check
    if result.should_use_path_a:
        # PATH A — return local answer (free, fast!)
        return JSONResponse(
            content={
                "id": "pelid-local",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.label,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "pelid_metadata": {
                    "path": "A",
                    "confidence": result.confidence,
                    "language": language,
                    "saved_cost": True,
                },
            }
        )
    else:
        # PATH B — forward to frontier LLM
        return await forward_to_upstream(body, request)


async def forward_to_upstream(body: dict, original_request: Request) -> JSONResponse:
    """
    Forward a request to the upstream frontier LLM (Path B).

    This is the fallback — used when:
    - The call is not a decision type
    - The context is too long for Laya
    - Laya's confidence is too low
    - The action is destructive
    """
    url = f"{UPSTREAM_BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=body, headers=headers)

    return JSONResponse(
        content=response.json(),
        status_code=response.status_code,
    )


def start():
    """Start the proxy server."""
    import uvicorn

    print(f"\n🚀 Pelid proxy starting on http://{HOST}:{PORT}")
    print(f"   Point your OpenAI client to: http://{HOST}:{PORT}/v1\n")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    start()
