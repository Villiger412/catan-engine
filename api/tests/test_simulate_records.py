"""
Tests 1-16: API endpoints for /api/simulate-records and /api/run-strategy.
Run with:  cd api && python -m pytest tests/ -v
"""
import pytest

# ── /api/health ────────────────────────────────────────────────────────────────

def test_health_ok(client):                          # 1
    r = client.get("/api/health")
    assert r.status_code == 200

def test_health_has_engine_field(client):            # 2
    r = client.get("/api/health")
    assert "engine" in r.json()


# ── /api/simulate-records ──────────────────────────────────────────────────────

SMALL = {"n_games": 200, "policy": "rule_based", "seed": 1}

def test_simulate_records_200(client):               # 3
    r = client.post("/api/simulate-records", json=SMALL)
    assert r.status_code == 200

def test_simulate_records_games_run_matches(client): # 4
    r = client.post("/api/simulate-records", json=SMALL)
    assert r.json()["games_run"] == SMALL["n_games"]

def test_simulate_records_four_seats(client):        # 5
    r = client.post("/api/simulate-records", json=SMALL)
    assert len(r.json()["seat_stats"]) == 4

def test_simulate_records_win_rates_sum_to_1(client):# 6
    r = client.post("/api/simulate-records", json=SMALL)
    total = sum(s["win_rate"] for s in r.json()["seat_stats"])
    assert abs(total - 1.0) < 0.01

def test_simulate_records_turns_monotone(client):    # 7
    r = client.post("/api/simulate-records", json=SMALL)
    d = r.json()
    assert d["turns_min"] <= d["turns_p25"] <= d["turns_median"]
    assert d["turns_median"] <= d["turns_p75"] <= d["turns_max"]

def test_simulate_records_avg_vp_positive(client):   # 8
    r = client.post("/api/simulate-records", json=SMALL)
    for s in r.json()["seat_stats"]:
        assert s["avg_vp"] > 0

def test_simulate_records_rates_in_unit_interval(client):  # 9
    r = client.post("/api/simulate-records", json=SMALL)
    for s in r.json()["seat_stats"]:
        assert 0.0 <= s["lr_rate"] <= 1.0
        assert 0.0 <= s["la_rate"] <= 1.0

def test_simulate_records_bad_policy_422(client):    # 10
    r = client.post("/api/simulate-records", json={**SMALL, "policy": "unicorn"})
    assert r.status_code == 422

def test_simulate_records_n_games_too_small_422(client):  # 11
    r = client.post("/api/simulate-records", json={**SMALL, "n_games": 10})
    assert r.status_code == 422

def test_simulate_records_n_games_too_large_422(client):  # 12
    r = client.post("/api/simulate-records", json={**SMALL, "n_games": 100_001})
    assert r.status_code == 422

def test_simulate_records_has_perf_fields(client):   # 13
    r = client.post("/api/simulate-records", json=SMALL)
    d = r.json()
    assert d["elapsed_ms"] > 0
    assert d["games_per_sec"] > 0
    assert d["policy"] != ""


# ── /api/run-strategy ─────────────────────────────────────────────────────────

EMPTY_STRATEGY = """\
from catan_research import Strategy

class MyStrat(Strategy):
    pass
"""

SETUP_HOOK_STRATEGY = """\
from catan_research import Strategy

class HighestVertex(Strategy):
    def score_setup_vertex(self, ctx, vertex_id):
        return float(vertex_id)
"""

def test_run_strategy_default_200(client):           # 14
    r = client.post("/api/run-strategy", json={
        "code": EMPTY_STRATEGY,
        "class_name": "MyStrat",
        "n_games": 100,
        "seed": 7,
    })
    assert r.status_code == 200

def test_run_strategy_result_schema(client):         # 15
    r = client.post("/api/run-strategy", json={
        "code": EMPTY_STRATEGY,
        "class_name": "MyStrat",
        "n_games": 100,
        "seed": 7,
    })
    d = r.json()
    assert "seat_stats" in d and "turns_median" in d and "games_run" in d

def test_run_strategy_bad_syntax_400(client):        # 16
    r = client.post("/api/run-strategy", json={
        "code": "def broken(::\n",
        "class_name": "Broken",
        "n_games": 100,
        "seed": 7,
    })
    assert r.status_code == 400
