"""
Pelid — Tri-Language Ergonomic Response Engine (Stage 8)

Transforms raw machine decision labels (e.g. 'order_status') into human-friendly,
conversational customer support responses for Path A resolutions.

Supported Languages:
- 'en': Professional English
- 'hi': Courteous Romanized Hindi (Hinglish)
- 'ml': Courteous Romanized Malayalam (Manglish)

Modes supported:
- 'template' (Default): Sub-millisecond, zero-RAM, zero-cost conversational responder
  with entity extraction and mock CRM integration.
- 'free_gemini': Generative one-sentence response via free-tier Gemini API without
  taxing local system RAM.
- 'slm': Local Small Language Model (e.g., Qwen2.5-0.5B / SmolLM) running on GPU/CPU.
"""

import os
from typing import Optional
from pelid.mock_crm import extract_order_id, lookup_order
from pelid.slot_extractor import extract_slots
from pelid.action_engine import get_action_engine

# Default response mode: "template", "free_gemini", or "slm"
RESPONDER_MODE = os.getenv("PELID_RESPONDER_MODE", "template").lower()

# ─── English Template Bank ───────────────────────────────────

RESPONSES_EN = {
    "order_status": (
        "I would be glad to check the status of your order! Could you please provide your Order ID or tracking number?"
    ),
    "order_delayed_or_missing": (
        "We sincerely apologize for the delay with your delivery. Please share your Order ID and we will immediately trace it with our logistics partner."
    ),
    "wrong_or_defective_item": (
        "We are very sorry that your item arrived defective or damaged. Please provide your Order ID and a photo of the package, and we will arrange an immediate replacement."
    ),
    "return_or_exchange": (
        "We offer hassle-free 7-day returns and exchanges. You can initiate a return directly from the 'My Orders' section of your account, or reply with your Order ID."
    ),
    "payment_failure": (
        "If money was debited from your bank but the order was not confirmed, it is typically reversed automatically within 3–5 business days. Please share your payment reference ID if you'd like us to trace it."
    ),
    "invoice_or_receipt": (
        "You can download your official GST tax invoice directly from the 'Orders' section in your account. If you'd like it emailed, please provide your Order ID."
    ),
    "login_or_password_issue": (
        "To regain access to your account, click 'Forgot Password' on the sign-in page to receive a password reset link, or request an instant login OTP via SMS."
    ),
    "update_account_details": (
        "You can update your shipping address, phone number, and name anytime under 'Account Settings' -> 'Profile'. Saved changes will apply to all future orders."
    ),
    "technical_bug_report": (
        "Thank you for reporting this issue. Our engineering team has been notified. In the meantime, try clearing your browser cache or updating the mobile app to the latest version."
    ),
    "how_to_question": (
        "You can find detailed step-by-step guides in our Help Center. If you have a specific step you are stuck on, please let me know and I will walk you through it!"
    ),
    "pricing_or_plan_question": (
        "We offer flexible monthly and annual plans. You can view all features, pricing tiers, and current discounts on our Plans page."
    ),
    "feature_request_or_feedback": (
        "Thank you for your valuable feedback! We've forwarded your suggestion to our product development team."
    ),
    "complaint_or_escalate_to_human": (
        "We apologize for the frustration caused. I am escalating your conversation to a senior support representative right away. An agent will connect with you shortly."
    ),
    "general_chitchat_or_faq": (
        "Hello! Welcome to Customer Support. How can I assist you with your orders, deliveries, or account today?"
    ),
    "other_unclear": (
        "Could you please share a bit more detail about your request so I can assist you better?"
    ),
}

# ─── Hinglish Template Bank (Hindi in Roman script) ───────────

RESPONSES_HINGLISH = {
    "order_status": (
        "Aapka order track karne ke liye please apna Order ID (#...) provide karein. Hum turant live tracking details check karke batayenge!"
    ),
    "order_delayed_or_missing": (
        "Delivery me delay ke liye hum sharminda hain. Please apna Order ID share karein, hum courier team se turant status confirm karte hain."
    ),
    "wrong_or_defective_item": (
        "Item toota/damaged aane ke liye hum maafi chahte hain. Please apna Order ID aur photo share karein, hum turant replacement arrange karenge."
    ),
    "return_or_exchange": (
        "Aap 7 din ke andar asani se return ya exchange kar sakte hain. 'My Orders' section se return initiate karein ya order ID share karein."
    ),
    "payment_failure": (
        "Paise kat gaye lekin order confirm nahi hua? 3-5 working days me bank account me auto-refund ho jayega. Transaction ID share karein."
    ),
    "invoice_or_receipt": (
        "Aapka official GST invoice 'My Orders' page se download ho sakta hai. Agar email par chahiye toh Order ID batayein."
    ),
    "login_or_password_issue": (
        "Account login ke liye 'Forgot Password' click karke link paayein, ya mobile number pe instant OTP mangwayein."
    ),
    "how_to_question": (
        "Step-by-step jankari ke liye hamare Help Center ko visit karein. Agar koi specific step samajh nahi aa raha toh batayein, hum guide karenge!"
    ),
    "general_chitchat_or_faq": (
        "Namaste! Customer Support me aapka swagat hai. Aaj hum aapki kya madad kar sakte hain?"
    ),
    "other_unclear": (
        "Kya aap thoda aur detail me bata sakte hain taaki hum aapki behtar madad kar sakein?"
    ),
}

# ─── Manglish Template Bank (Malayalam in Roman script) ───────

RESPONSES_MANGLISH = {
    "order_status": (
        "Ningalude order track cheyyan vendi please Order ID (#...) share cheyyamo? Njangal udan thanne live tracking details check cheythu ariyikkam!"
    ),
    "order_delayed_or_missing": (
        "Delivery delay aayathil khedikkunnu. ദയവായി (Please) order ID share cheyyamo? Njangal courier team-umayi check cheyyam."
    ),
    "wrong_or_defective_item": (
        "Item damaged/pottiya aayi vannathil khedikkunnu. Please order number and photo share cheyyamo? Njangal udan thanne free replacement arrange cheyyam."
    ),
    "return_or_exchange": (
        "7 divasathinnullil asani aayi return/exchange cheyyam. 'My Orders' section vazhi initiate cheyyam, allekil order ID share cheyyu."
    ),
    "payment_failure": (
        "Paise debit aayi pakshe order confirm aayille? 3-5 working days-il bank account-il auto-refund aakum. Transaction ID undo?"
    ),
    "invoice_or_receipt": (
        "Ningalude official GST invoice 'My Orders' page-il ninnu download cheyyam. Email-il venamenkil Order ID ayakkuka."
    ),
    "how_to_question": (
        "Step-by-step sahayam njangalude Help Center-il labhyamanu. Ethelum specific stepil doubt undenkil parayamo, njangal sahayikkam!"
    ),
    "login_or_password_issue": (
        "Login cheyyan kazhiyunille? 'Forgot Password' link use cheyyu, allekil mobile number-il OTP request cheyyu."
    ),
    "general_chitchat_or_faq": (
        "Namaskaram! Customer Support-il ninnum engane sahayikkanam? Orders, delivery, returns sambhavikkan enthelum undo?"
    ),
    "other_unclear": (
        "Kooduthal vivarangal share cheyyamo? Enkil njangalkku ningale nannayi sahayikkan kazhiyum."
    ),
}


def generate_crm_response(query: str, intent: str, language: str = "en") -> str:
    """
    Generate an ergonomic, human-like response for Path A resolutions.
    Integrates with Declarative Action Engine and Mock CRM for live order lookups.
    """
    # 1. Extract dynamic parameter slots from query (e.g. order_id, email)
    slots = extract_slots(query)

    # 2. Check if Declarative Action Engine has a defined action in pelid.yaml
    action_engine = get_action_engine()
    intent_cfg = action_engine.get_intent_config(intent)
    if intent_cfg:
        required_slots = intent_cfg.get("slots", intent_cfg.get("required_slots", []))
        missing = [s for s in required_slots if s not in slots or not slots[s]]
        if missing:
            prompt_templates = intent_cfg.get("prompt_for_slots", {})
            return prompt_templates.get(language, prompt_templates.get("en", f"Could you please provide your {missing[0].replace('_', ' ')}?"))

        # Synchronous execution uses mock data resolver
        mock_data = action_engine._execute_mock_action(intent, slots)
        if mock_data:
            context = dict(slots)
            context.update(mock_data)
            return action_engine._render_template(intent_cfg.get("response", {}), context, language)
        else:
            return action_engine._render_not_found(intent_cfg, slots, language)

    # 3. Select Language-Appropriate Conversational Template from Bank
    if language == "ml" and intent in RESPONSES_MANGLISH:
        return RESPONSES_MANGLISH[intent]
    elif language == "hi" and intent in RESPONSES_HINGLISH:
        return RESPONSES_HINGLISH[intent]
    elif intent in RESPONSES_EN:
        return RESPONSES_EN[intent]

    # Default friendly fallback
    return f"I have routed your request regarding '{intent}'. How else may I assist you today?"


async def resolve_path_a_response(
    query: str,
    intent: str,
    language: str = "en",
    session_id: str = "",
) -> str:
    """
    Entry point for generating the final assistant message for Path A.
    Dispatches to:
    1. 'template' (Default): Executes declarative action from pelid.yaml (REST API / CRM lookup).
    2. 'slm': On-device / local small language model (e.g. Ollama / llama.cpp / vLLM Qwen2.5-0.5B).
    3. 'free_gemini': Grounded micro-LLM via Google Gemini API with local intent gating.
    """
    if RESPONDER_MODE == "template":
        # 1. Dynamic Slot Extraction
        slots = extract_slots(query)

        # 2. Asynchronous Declarative Action Execution (REST API / Webhook)
        action_engine = get_action_engine()
        rendered_action = await action_engine.execute_action(intent, slots, language, session_id=session_id)
        if rendered_action:
            return rendered_action

        # Fallback to standard CRM templates
        return generate_crm_response(query, intent, language)

    elif RESPONDER_MODE == "slm":
        import httpx

        slm_url = os.getenv("PELID_SLM_URL", "http://127.0.0.1:11434/v1/chat/completions")
        slm_model = os.getenv("PELID_SLM_MODEL", "qwen2.5:0.5b")
        lang_prompt = {
            "ml": "Malayalam (Romanized / Manglish)",
            "hi": "Hindi (Romanized / Hinglish)",
            "en": "English",
        }.get(language, "English")

        prompt = (
            f"You are a helpful customer support assistant. The customer's intent is verified as: '{intent}'.\n"
            f"Customer query: \"{query}\"\n"
            f"Write a friendly, accurate 1-sentence reply in {lang_prompt}. Do not hallucinate order numbers."
        )
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.post(
                    slm_url,
                    json={
                        "model": slm_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 80,
                        "temperature": 0.3,
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    if "choices" in data and data["choices"]:
                        return data["choices"][0]["message"]["content"].strip()
        except Exception:
            # If local SLM daemon is not running, gracefully fallback to deterministic CRM template
            pass

    elif RESPONDER_MODE == "free_gemini":
        from pelid.config import GEMINI_API_KEY, UPSTREAM_BASE_URL, UPSTREAM_MODEL
        import httpx

        lang_prompt = {
            "ml": "Malayalam (Romanized / Manglish)",
            "hi": "Hindi (Romanized / Hinglish)",
            "en": "English",
        }.get(language, "English")

        prompt = (
            f"You are a courteous customer support assistant. The user's intent is classified as: '{intent}'.\n"
            f"User message: \"{query}\"\n"
            f"Reply in 1 or 2 polite sentences matching their language ({lang_prompt})."
        )
        url = f"{UPSTREAM_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": UPSTREAM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.3,
        }
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    if "choices" in data and data["choices"]:
                        return data["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    # Default robust fallback to CRM template
    return generate_crm_response(query, intent, language)
