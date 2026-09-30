"""Configuration loading and validation (pydantic)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
Point = Tuple[float, float]


class StoreCfg(BaseModel):
    store_id: str
    name: str = ""
    timezone: str = "Asia/Kolkata"
    open_hour: int = 8
    close_hour: int = 22


class ModelsCfg(BaseModel):
    person_detector: str = "models/person_detector.pt"
    shelf_detector: str = "models/shelf_detector.pt"
    queue_forecaster: str = "models/queue_forecaster.joblib"
    device: str = "cpu"
    person_conf: float = 0.35
    shelf_conf: float = 0.25
    imgsz: int = 640


class PrivacyCfg(BaseModel):
    store_frames: bool = False
    blur_people_in_previews: bool = True
    # full = blur the whole preview frame + pixelate tracked people (safe even while a new person
    # is not yet confirmed by the tracker); people = pixelate tracked people only
    preview_mode: str = "full"
    id_salt_rotation_hours: int = 24
    min_group_size_for_reports: int = 3


class Zone(BaseModel):
    name: str
    polygon: List[Point]


class Counter(BaseModel):
    name: str
    queue_polygon: List[Point]
    service_polygon: List[Point]


class ShelfSection(BaseModel):
    name: str
    polygon: Optional[List[Point]] = None
    expected_facings: int = 0
    expected_rows: int = 0
    sku_group: str = ""


class CameraCfg(BaseModel):
    id: str
    role: str  # entrance | floor | queue | shelf
    source: str
    process_width: int = 960
    target_fps: float = 10
    count_line: Optional[List[Point]] = None
    entry_direction: str = "down"
    imgsz: Optional[int] = None  # per-camera inference size (entrance: 960 counts small / distant people better)
    count_margin: float = 8.0  # dead band (pixels) around the count line: no jitter double counts
    zones: List[Zone] = Field(default_factory=list)
    heatmap_cell: int = 16
    counters: List[Counter] = Field(default_factory=list)
    shelf_sections: List[ShelfSection] = Field(default_factory=list)
    snapshot_interval_s: float = 60

    @field_validator("role")
    @classmethod
    def _role(cls, v):
        allowed = {"entrance", "floor", "queue", "shelf"}
        if v not in allowed:
            raise ValueError(f"camera role must be one of {allowed}")
        return v

    @field_validator("entry_direction")
    @classmethod
    def _dir(cls, v):
        allowed = {"down", "up", "left", "right"}
        if v not in allowed:
            raise ValueError(f"entry_direction must be one of {allowed}")
        return v


class QueueCfg(BaseModel):
    max_queue_per_counter: int = 4
    target_wait_minutes: float = 5
    avg_service_minutes: float = 2.0
    forecast_horizon_minutes: int = 15
    counters_total: int = 6


class InventoryCfg(BaseModel):
    low_stock_ratio: float = 0.5
    out_of_stock_gap_ratio: float = 1.5
    min_void_area_ratio: float = 0.01
    texture_ratio: float = 0.6       # empty-shelf check: void edge density < ratio x product edge density
    baseline_min_cover: float = 0.3  # fixed camera: a baseline product position counts as empty below this coverage
    baseline_texture_ratio: float = 0.0


class AlertsCfg(BaseModel):
    cooldown_minutes: float = 10
    webhook_url: str = ""
    notify_levels: List[str] = Field(default_factory=lambda: ["critical", "warning"])


class SyncCfg(BaseModel):
    central_url: str = ""
    api_key_env: str = "RETAIL_SYNC_KEY"
    batch_size: int = 500
    interval_s: float = 60


class StorageCfg(BaseModel):
    db_path: str = "data/retail_edge.db"


class IntegrationsCfg(BaseModel):
    pos_csv_dir: str = "data/pos_exports"
    inventory_csv: str = "data/inventory_master.csv"


class AppConfig(BaseModel):
    store: StoreCfg
    models: ModelsCfg = ModelsCfg()
    privacy: PrivacyCfg = PrivacyCfg()
    cameras: List[CameraCfg] = Field(default_factory=list)
    queue: QueueCfg = QueueCfg()
    inventory: InventoryCfg = InventoryCfg()
    alerts: AlertsCfg = AlertsCfg()
    sync: SyncCfg = SyncCfg()
    storage: StorageCfg = StorageCfg()
    integrations: IntegrationsCfg = IntegrationsCfg()

    def resolve(self, p: str) -> Path:
        """Resolve a path relative to the project root."""
        path = Path(p)
        return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path = ROOT / "configs" / "store_config.yaml") -> AppConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)
