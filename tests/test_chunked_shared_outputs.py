"""MODEL_DIR's shared files survive being written by several chunk runs.

Three files describe ALL assets while one train_hybrid run only knows the assets
it trained: champion_registry.json, tuned_thresholds.json and quality_report.json.
Writing them whole loses the other chunks' assets - sequentially it left
tuned_thresholds.json holding only the last chunk of a 208-asset retrain, and
concurrently the second process to finish erased the first one's champions.
"""

import json

import train_chunked
import train_hybrid


def test_a_second_chunk_does_not_erase_the_first_ones_thresholds(tmp_path):
    path = str(tmp_path / "tuned_thresholds.json")
    train_hybrid._merged_map_write(path, {"AAPL": 0.51, "MSFT": 0.48})
    train_hybrid._merged_map_write(path, {"SBER": 0.55})
    assert json.loads(open(path, encoding="utf-8").read()) == {
        "AAPL": 0.51, "MSFT": 0.48, "SBER": 0.55}


def test_a_later_chunk_still_updates_an_asset_it_retrained(tmp_path):
    path = str(tmp_path / "champion_registry.json")
    train_hybrid._merged_map_write(path, {"AAPL": {"score": 1.0}})
    train_hybrid._merged_map_write(path, {"AAPL": {"score": 2.0}})
    assert json.loads(open(path, encoding="utf-8").read())["AAPL"]["score"] == 2.0


def test_the_quality_report_accumulates_rows_across_chunks(tmp_path):
    path = str(tmp_path / "quality_report.json")
    train_hybrid._merged_rows_write(path, [{"Asset": "AAPL", "Score": 1.5}])
    train_hybrid._merged_rows_write(path, [{"Asset": "SBER", "Score": 2.5}])
    rows = json.loads(open(path, encoding="utf-8").read())
    assert {r["Asset"]: r["Score"] for r in rows} == {"AAPL": 1.5, "SBER": 2.5}


def test_an_unreadable_shared_file_is_replaced_rather_than_inherited(tmp_path):
    """A half-written file must not take the run's results down with it."""
    path = str(tmp_path / "tuned_thresholds.json")
    open(path, "w", encoding="utf-8").write("{not json")
    train_hybrid._merged_map_write(path, {"AAPL": 0.51})
    assert json.loads(open(path, encoding="utf-8").read()) == {"AAPL": 0.51}


# --- the chunk runner -------------------------------------------------------

def test_workers_are_divided_between_parallel_chunk_processes():
    """Two processes at the full worker count would double the concurrent
    trainings on a card with no room for them."""
    solo = train_chunked._chunk_env(["AAPL"], False, 1)
    pair = train_chunked._chunk_env(["AAPL"], False, 2)
    assert solo["GTRADE_WORKERS"] == train_chunked.LIGHT_ENV["GTRADE_WORKERS"]
    assert int(pair["GTRADE_WORKERS"]) == int(solo["GTRADE_WORKERS"]) // 2


def test_the_chunk_environment_names_only_that_chunks_assets():
    env = train_chunked._chunk_env(["AAPL", "MSFT"], True, 2)
    assert env["GTRADE_ASSETS"] == "AAPL,MSFT"
    assert env["GTRADE_FORCE_PROMOTE"] == "1"
