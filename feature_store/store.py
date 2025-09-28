# feature_store/store.py
"""
OnlineFeatureStore stub and utility methods used by the API at runtime.
For now this builds minimal online features; later we can connect Redis
and rolling aggregates.
"""
from __future__ import annotations
from typing import Dict, Any
import hashlib

class OnlineFeatureStore:
    def __init__(self):
        # In a full implementation: connect to Redis and prefill counters
        pass

    def build_online_features(self, ticket) -> Dict[str, Any]:
        text = "\n".join([
            getattr(ticket, "subject", "") or "",
            getattr(ticket, "description", "") or "",
            getattr(ticket, "error_logs", "") or "",
            getattr(ticket, "stack_trace", "") or "",
        ])
        return {
            "text": text,
            "product": getattr(ticket, "product", None) or "unknown",
            "tier": getattr(ticket, "customer_tier", None) or "unknown",
            "has_error_code": ("ERROR_" in text) or (" error " in text.lower()),
            "text_hash": hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest(),
        }
