"""Store-and-forward synchronisation to a central HQ server.

* Runs in a background thread; if `central_url` is empty the store runs 100 % offline.
* Sends only aggregated anonymous metrics, alerts and snapshots (a few KB per minute
  instead of video streams -> >99 % bandwidth reduction vs cloud video analytics).
* On network failure it backs off exponentially and retries; nothing is lost because
  rows stay `synced=0` in the local SQLite database until the server confirms.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

from .storage.db import EdgeDB

log = logging.getLogger("retail_ai.sync")


class SyncAgent:
    def __init__(self, db: EdgeDB, store_id: str, central_url: str, api_key_env: str = "RETAIL_SYNC_KEY",
                 batch_size: int = 500, interval_s: float = 60, store_name: str = ""):
        self.db = db
        self.store_id = store_id
        self.store_name = store_name
        self.url = central_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.batch = batch_size
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.backoff = interval_s
        self.last_success: Optional[float] = None
        self.last_error: Optional[str] = None

    def sync_once(self) -> int:
        """Push one batch per table. Returns number of rows confirmed by the server."""
        import requests

        sent = 0
        for table in ("alerts", "metrics", "snapshots"):
            rows = self.db.unsynced(table, self.batch)
            if not rows:
                continue
            body = {"store_id": self.store_id, "store_name": self.store_name, "table": table, "rows": rows}
            r = requests.post(f"{self.url}/api/ingest", data=json.dumps(body), timeout=15,
                              headers={"Content-Type": "application/json", "X-API-Key": self.api_key})
            r.raise_for_status()
            self.db.mark_synced(table, [row["id"] for row in rows])
            sent += len(rows)
        return sent

    def _run(self):
        while not self._stop.is_set():
            try:
                n = self.sync_once()
                self.last_success, self.last_error = time.time(), None
                self.backoff = self.interval
                if n:
                    log.info("synced %d rows to HQ", n)
            except Exception as exc:
                self.last_error = str(exc)
                self.backoff = min(self.backoff * 2, 3600)
                log.warning("HQ unreachable, working offline (retry in %.0fs): %s", self.backoff, exc)
            self._stop.wait(self.backoff)

    def start(self):
        if not self.url:
            log.info("sync disabled (no central_url) - fully offline mode")
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="sync")
        self._thread.start()

    def stop(self):
        self._stop.set()
