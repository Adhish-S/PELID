"""
Pelid — Built-in Mock Support CRM

Provides a self-contained in-memory / SQLite database of orders, tracking statuses,
and customer accounts for zero-dependency local testing and production-grade demos.
"""

import re
from typing import Optional, Dict, Any

MOCK_ORDERS: Dict[str, Dict[str, Any]] = {
    "9921": {
        "order_id": "#9921",
        "item": "Sony WH-1000XM5 Wireless Headphones",
        "status": "Out for Delivery",
        "courier": "BlueDart Express",
        "tracking_num": "BD90124881",
        "amount": "₹24,990",
        "eta": "Today by 6:00 PM",
    },
    "1042": {
        "order_id": "#1042",
        "item": "Nike Air Pegasus 40 Running Shoes",
        "status": "In Transit (Bengaluru Hub)",
        "courier": "Delhivery",
        "tracking_num": "DL78219402",
        "amount": "₹9,495",
        "eta": "Tomorrow by 2:00 PM",
    },
    "89214": {
        "order_id": "#89214",
        "item": "Apple MacBook Air M3 (Space Gray)",
        "status": "Dispatched from Mumbai Fulfillment Center",
        "courier": "BlueDart Aviation",
        "tracking_num": "BD11492044",
        "amount": "₹1,14,900",
        "eta": "In 2 business days",
    },
    "5512": {
        "order_id": "#5512",
        "item": "Ergonomic High-Back Office Chair",
        "status": "Delivered on 28 Sep",
        "courier": "Shadowfax",
        "tracking_num": "SF44182910",
        "amount": "₹14,200",
        "eta": "Delivered",
    },
    "3301": {
        "order_id": "#3301",
        "item": "Keychron K2 Mechanical Keyboard",
        "status": "Out for Delivery",
        "courier": "BlueDart Express",
        "tracking_num": "BD88201944",
        "amount": "₹7,999",
        "eta": "Today by 7:30 PM",
    },
    "7740": {
        "order_id": "#7740",
        "item": "Samsung Galaxy S24 Ultra (512GB)",
        "status": "Dispatched",
        "courier": "BlueDart Express",
        "tracking_num": "BD77192031",
        "amount": "₹1,29,999",
        "eta": "Tomorrow by 4:00 PM",
    },
    "4418": {
        "order_id": "#4418",
        "item": "Noise ColorFit Pulse Smartwatch",
        "status": "Returned & Refund Processed",
        "courier": "Delhivery Reverse",
        "tracking_num": "DL99018421",
        "amount": "₹2,499",
        "eta": "Refund credited to original bank account",
    },
    "1209": {
        "order_id": "#1209",
        "item": "Levi's Classic Denim Jacket",
        "status": "Out for Delivery",
        "courier": "Delhivery",
        "tracking_num": "DL12094812",
        "amount": "₹3,499",
        "eta": "Today by 5:00 PM",
    },
}


def extract_order_id(text: str) -> Optional[str]:
    """
    Extract order ID reference from a user query (e.g. #9921, 9921, ORD-1042).
    """
    # 1. Match explicit #1234 or ORD-1234
    match = re.search(r"(?:#|ord-?|order\s*(?:id|no|#)?\s*)([0-9]{3,6})\b", text, re.IGNORECASE)
    if match:
        return match.group(1)

    # 2. Check known IDs directly in text
    for oid in MOCK_ORDERS.keys():
        if oid in text:
            return oid

    return None


def lookup_order(order_id: str) -> Optional[Dict[str, Any]]:
    """Lookup order details by ID in mock CRM."""
    clean_id = order_id.replace("#", "").strip()
    return MOCK_ORDERS.get(clean_id)
