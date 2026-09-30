"""Alert engine with de-duplication (cooldown) and optional webhook delivery.

Webhook failures never block the edge loop: alerts are always stored locally
first; delivery is best-effort (e.g. WhatsApp Business / Slack / ERP endpoint).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Dict, Optional, Tuple

from .storage.db import EdgeDB

log = logging.getLogger("retail_ai.alerts")


class AlertManager:
    def __init__(self, db: EdgeDB, store_id: str, cooldown_minutes: float = 10,
                 webhook_url: str = "", notify_levels=("critical", "warning")):
        self.db = db
        self.store_id = store_id
        self.cooldown_s = cooldown_minutes * 60
        self.webhook_url = webhook_url
        self.notify_levels = set(notify_levels)
        self._last: Dict[Tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def raise_alert(self, level: str, type_: str, key: str, message: str,
                    payload: Optional[dict] = None, ts: Optional[float] = None) -> Optional[int]:
        ts = time.time() if ts is None else ts
        with self._lock:
            last = self._last.get((type_, key))
            if last is not None and ts - last < self.cooldown_s:
                return None
            self._last[(type_, key)] = ts
        alert_id = self.db.add_alert(ts, self.store_id, level, type_, key, message, payload)
        log.info("ALERT [%s] %s %s: %s", level, type_, key, message)
        if self.webhook_url and level in self.notify_levels:
            threading.Thread(target=self._post, args=(level, type_, key, message, payload, ts), daemon=True).start()
        return alert_id

    def _post(self, level, type_, key, message, payload, ts):
        try:
            import requests
            requests.post(self.webhook_url, timeout=5, data=json.dumps({
                "store_id": self.store_id, "ts": ts, "level": level, "type": type_,
                "key": key, "message": message, "payload": payload or {}}),
                headers={"Content-Type": "application/json"})
        except Exception as exc:  # offline is a normal condition at the edge
            log.warning("webhook delivery failed (kept locally): %s", exc)
