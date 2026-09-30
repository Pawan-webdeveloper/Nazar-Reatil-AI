"""RetailSense AI - command line entry point.

    python run.py edge       # run all cameras + dashboard/API on http://localhost:8000
    python run.py edge --no-api --fast --max-frames 300 --video-out outputs/demo_videos
    python run.py dashboard  # API + dashboard only (e.g. HQ server receiving store syncs)
    python run.py demo-data  # fill the local DB with 4 weeks of simulated history (for demos)
    python run.py report --day 2026-09-28
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from retail_ai.config import load_config  # noqa: E402
from retail_ai.storage.db import EdgeDB  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["edge", "dashboard", "demo-data", "report"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "store_config.yaml"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-api", action="store_true", help="edge: do not start the dashboard")
    ap.add_argument("--fast", action="store_true", help="edge: process video files as fast as possible "
                                                        "(simulated clock) instead of real time")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--video-out", default="", help="edge: write annotated (privacy-blurred) videos here")
    ap.add_argument("--cameras", nargs="*", help="edge: only these camera ids")
    ap.add_argument("--loop", action="store_true", help="edge: replay video files / image folders forever (demo)")
    ap.add_argument("--day", default=None)
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    db = EdgeDB(cfg.resolve(cfg.storage.db_path))

    if a.command == "demo-data":
        from retail_ai.demo_data import populate
        n = populate(db, cfg, days=a.days)
        print(f"inserted {n:,} rows of simulated history into {cfg.storage.db_path}")
        return 0

    if a.command == "report":
        from retail_ai import reports
        print(json.dumps(reports.daily_report(db, cfg.store.store_id, a.day), indent=2))
        return 0

    node = None
    if a.command == "edge":
        from retail_ai.edge.pipeline import EdgeNode
        node = EdgeNode(cfg, realtime=not a.fast, max_frames=a.max_frames, video_dir=a.video_out or None,
                        db=db, cameras=a.cameras, loop_files=a.loop)
        node.start()
        signal.signal(signal.SIGINT, lambda *_: node.stop())
        if a.no_api:
            node.join()
            print(json.dumps(node.health(), indent=2))
            return 0

    import uvicorn
    from retail_ai.api.server import create_app
    uvicorn.run(create_app(cfg, db, node), host=a.host, port=a.port, log_level="info")
    if node:
        node.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
