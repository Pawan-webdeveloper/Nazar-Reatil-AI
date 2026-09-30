import math

import numpy as np
import pytest

from retail_ai.analytics.dwell import ZoneDwellTracker
from retail_ai.analytics.footfall import LineCounter
from retail_ai.analytics.forecasting import expected_wait_minutes, erlang_c_prob_wait, recommend_counters
from retail_ai.analytics.heatmap import Heatmap
from retail_ai.analytics.queue_monitor import QueueMonitor
from retail_ai.analytics.shelf import ShelfAnalyzer, cluster_rows
from retail_ai.geometry import greedy_match, iou_matrix, point_in_polygon, segments_intersect
from retail_ai.privacy import AnonymousIdMapper, blur_regions, k_anonymize


# ------------------------------------------------------------------ geometry
def test_point_in_polygon():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), sq)
    assert not point_in_polygon((15, 5), sq)
    assert not point_in_polygon((-1, -1), sq)


def test_segments_intersect():
    assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))
    assert not segments_intersect((0, 0), (1, 1), (5, 5), (6, 7))


def test_iou_and_matching():
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    b = np.array([[0, 0, 10, 10], [21, 21, 31, 31], [100, 100, 110, 110]])
    m = iou_matrix(a, b)
    assert m.shape == (2, 3)
    assert m[0, 0] == pytest.approx(1.0)
    pairs = greedy_match(m, 0.5)
    assert set(pairs) == {(0, 0), (1, 1)}
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


# ------------------------------------------------------------------ footfall
def test_line_counter_entries_and_exits():
    lc = LineCounter(((0, 100), (200, 100)), entry_direction="down")
    lc.update(0.0, {1: (50, 80), 2: (150, 130)})
    lc.update(0.1, {1: (50, 120), 2: (150, 90)})  # 1 goes down (entry), 2 goes up (exit)
    assert (lc.entries, lc.exits) == (1, 1)
    # jitter back and forth across the line within recount window counts once per direction
    lc.update(0.2, {1: (50, 95)})
    lc.update(0.3, {1: (50, 105)})
    assert lc.entries == 1
    assert lc.occupancy == 0


def test_line_counter_dead_band_and_missed_frames():
    lc = LineCounter(((0, 100), (200, 100)), "down", margin=8)
    # a person standing on the line with a jittering foot point is never counted
    for t, y in enumerate([97, 104, 96, 103, 99, 105]):
        lc.update(float(t), {1: (50, y)})
    assert (lc.entries, lc.exits) == (0, 0)
    # tracker loses the person right at the line; the remembered side still gives the crossing
    lc.update(10.0, {2: (60, 60)})
    lc.update(11.0, {})
    lc.update(12.0, {2: (60, 150)})
    assert lc.entries == 1


def test_line_counter_stitches_id_switch():
    lc = LineCounter(((0, 100), (200, 100)), "down", margin=8, stitch_px=60)
    lc.update(0.0, {1: (50, 70)})          # id 1 committed above the line
    lc.update(0.5, {7: (55, 90)})          # tracker switched to id 7 (inside dead band)
    lc.update(1.0, {7: (58, 140)})         # id 7 now below the line
    assert lc.entries == 1
    far = LineCounter(((0, 100), (200, 100)), "down", margin=8, stitch_px=60)
    far.update(0.0, {1: (50, 70)})
    far.update(0.5, {9: (190, 140)})       # new person far away: no inheritance
    assert far.entries == 0


def test_line_counter_ignores_crossing_outside_segment():
    lc = LineCounter(((0, 100), (100, 100)), "down")
    lc.update(0, {1: (150, 50)})
    lc.update(1, {1: (150, 150)})
    assert lc.entries == 0


# ------------------------------------------------------------------ dwell
def test_dwell_visits():
    zt = ZoneDwellTracker({"promo": [(0, 0), (100, 0), (100, 100), (0, 100)]}, min_visit_s=1, engaged_s=10,
                          lost_timeout_s=2)
    for t in range(0, 15):
        zt.update(float(t), {7: (50, 50)})
    zt.update(15.0, {7: (500, 500)})  # left zone
    s = zt.summary()["promo"]
    assert s["visits"] == 1
    assert s["avg_dwell_s"] == pytest.approx(14.0)
    assert s["engaged_visits"] == 1


def test_dwell_short_pass_by_ignored_and_lost_track_closed():
    zt = ZoneDwellTracker({"z": [(0, 0), (10, 0), (10, 10), (0, 10)]}, min_visit_s=2, lost_timeout_s=1)
    zt.update(0.0, {1: (5, 5)})
    zt.update(0.5, {1: (50, 50)})
    assert zt.summary()["z"]["visits"] == 0
    zt.update(1.0, {2: (5, 5)})
    zt.update(4.0, {2: (5, 5)})
    zt.update(10.0, {})  # track 2 lost -> visit closed
    assert zt.summary()["z"]["visits"] == 1


# ------------------------------------------------------------------ heatmap
def test_heatmap_accumulates_and_normalizes():
    hm = Heatmap(160, 160, 16)
    hm.add([(10, 10)] * 5 + [(150, 150)])
    n = hm.normalized()
    assert n.max() == pytest.approx(1.0)
    assert hm.grid.sum() == 6
    assert hm.render().shape == (160, 160, 3)


# ------------------------------------------------------------------ queue
def _counter():
    return [{"name": "c1", "queue_polygon": [(0, 0), (100, 0), (100, 100), (0, 100)],
             "service_polygon": [(100, 0), (150, 0), (150, 100), (100, 100)]}]


def test_queue_monitor_wait_and_service_time():
    qm = QueueMonitor(_counter(), lost_timeout_s=1, min_service_s=1)
    for t in range(0, 30):  # person 1 waits 30 s
        qm.update(float(t), {1: (50, 50), 2: (60, 60)})
    assert qm.status(29)[0]["queue_length"] == 2
    for t in range(30, 90):  # person 1 served for 60 s, person 2 still waiting
        qm.update(float(t), {1: (120, 50), 2: (60, 60)})
    st = qm.status(89)[0]
    assert st["queue_length"] == 1 and st["in_service"] == 1
    assert st["avg_wait_s"] == pytest.approx(30.0)
    qm.update(95.0, {2: (120, 50)})  # 1 left, 2 moves to service
    st = qm.status(95)[0]
    assert st["avg_service_s"] == pytest.approx(59.0)
    assert st["measured_services"] == 1


# ------------------------------------------------------------------ Erlang C
def test_erlang_c_known_values():
    # M/M/1: P(wait)=rho, Wq = rho/(mu-lam)
    assert erlang_c_prob_wait(0.5, 1.0, 1) == pytest.approx(0.5)
    assert expected_wait_minutes(0.5, 1.0, 1) == pytest.approx(1.0)
    # textbook: a=2 Erlang, c=3 -> P(wait)=0.4444
    assert erlang_c_prob_wait(2.0, 1.0, 3) == pytest.approx(0.4444, abs=1e-3)
    assert math.isinf(expected_wait_minutes(3.0, 1.0, 2))


def test_recommend_counters_monotonic():
    low = recommend_counters(0.5, 2.0, 3, 8)["recommended_counters"]
    high = recommend_counters(3.0, 2.0, 3, 8)["recommended_counters"]
    assert 1 <= low < high <= 8
    r = recommend_counters(3.0, 2.0, 3, 8)
    assert r["expected_wait_min"] <= 3


# ------------------------------------------------------------------ shelf
def _shelf(n_rows=3, per_row=10, w=40, h=80, gap=4, remove=()):
    boxes = []
    for r in range(n_rows):
        for i in range(per_row):
            if (r, i) in remove:
                continue
            x = 10 + i * (w + gap)
            y = 10 + r * (h + 30)
            boxes.append([x, y, x + w, y + h])
    return np.array(boxes, dtype=float)


def test_row_clustering():
    rows = cluster_rows(_shelf())
    assert len(rows) == 3 and all(len(r) == 10 for r in rows)


def test_full_shelf_is_ok():
    rep = ShelfAnalyzer().analyze(_shelf(), (600, 400))
    assert rep.status == "OK" and not rep.voids and rep.facings == 30


def test_void_detected_in_middle_of_row():
    boxes = _shelf(remove={(1, 4), (1, 5), (1, 6)})
    rep = ShelfAnalyzer(gap_ratio=1.5).analyze(boxes, (600, 400))
    assert len(rep.voids) == 1
    v = rep.voids[0]
    assert v.row == 1 and v.missing_facings == 3
    assert rep.capacity_estimate == 30


def test_empty_section_is_out_of_stock():
    rep = ShelfAnalyzer().analyze(np.zeros((0, 4)), (600, 400), expected_facings=20)
    assert rep.status == "OUT_OF_STOCK" and rep.stock_ratio_vs_planogram == 0.0


def test_planogram_low_stock():
    boxes = _shelf(n_rows=2, per_row=5)
    rep = ShelfAnalyzer().analyze(boxes, (600, 400), expected_facings=40, expected_rows=4)
    assert rep.status in ("LOW_STOCK", "OUT_OF_STOCK")
    assert rep.planogram_compliance < 0.8
    assert any("expected 4 shelf rows" in i for i in rep.issues)


# ------------------------------------------------------------------ privacy
def test_anonymous_ids_rotate_and_do_not_leak_track_id():
    m = AnonymousIdMapper(rotation_hours=24, secret=b"k" * 16)
    a = m.anon("cam", 5, ts=1000)
    assert a == m.anon("cam", 5, ts=2000)
    assert a != m.anon("cam", 5, ts=1000 + 86400)
    assert "5" != a and len(a) == 12


def test_blur_changes_only_box():
    img = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = blur_regions(img, [[10, 10, 60, 60]])
    assert not np.array_equal(out[10:60, 10:60], img[10:60, 10:60])
    assert np.array_equal(out[70:, 70:], img[70:, 70:])


def test_k_anonymity():
    assert k_anonymize(12.0, 2, 3) is None
    assert k_anonymize(12.0, 3, 3) == 12.0


def test_baseline_voids_fixed_camera():
    from retail_ai.analytics.shelf import baseline_voids
    full = _shelf()
    now = _shelf(remove={(2, 3), (2, 4), (2, 5)})
    v = baseline_voids(full, now)
    assert len(v) == 1 and v[0].missing_facings == 3
    assert baseline_voids(full, full) == []
    rep = ShelfAnalyzer().analyze(now, (600, 400), baseline_boxes=full)
    assert len(rep.voids) == 1 and rep.capacity_estimate == 30 and len(rep.boxes) == 27
