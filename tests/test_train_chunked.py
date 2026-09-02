"""Tests for train_chunked.py, the RAM-safe form of the full retrain.

The module and this file used to be gitignored together, grouped with the local
ab_* harnesses. That was an accident of neighbourhood: train_chunked is not a
harness, it is the training path the menu runs for a full rebuild, and the
README documented it while the repository did not carry it.
"""

import os

import auto_loop
import auto_research as ar
import train_chunked


def test_workers_are_divided_between_parallel_chunk_processes():
    """Two processes at the full worker count would double the concurrent
    trainings on a card with no room for them."""
    solo = train_chunked._chunk_env(["AAPL"], False, 1)
    pair = train_chunked._chunk_env(["AAPL"], False, 2)
    assert solo["GTRADE_WORKERS"] == train_chunked.LIGHT_ENV["GTRADE_WORKERS"]
    assert int(pair["GTRADE_WORKERS"]) == int(solo["GTRADE_WORKERS"]) // 2


def test_the_vram_pool_is_divided_and_fits_the_card_twice():
    """The binding constraint is VRAM, not RAM. train_hybrid reserves the pool
    as a PERCENTAGE, so two processes at the default ask for 120% of the card;
    measured on this box, one process alone already holds 3507 of 4096 MiB."""
    pair = train_chunked._chunk_env(["AAPL"], False, 2)
    assert float(pair["GTRADE_TF_POOL_PCT"]) * 2 <= 0.60, "two pools exceed one default pool"
    # The floor this used to cite was 1024 MB, and it was the reason the pair
    # peaked at 3956 of 4096 MiB whatever the percentage said: both processes
    # took 1024 regardless, so lowering the share did nothing. train_hybrid now
    # floors at 640 MiB, which is 0.156 of a 4 GB card, so a share below that
    # is the value that buys nothing.
    assert float(pair["GTRADE_TF_POOL_PCT"]) >= 0.15
    assert pair["GTRADE_NEURAL_SLOTS"] == "1"


def test_a_single_process_run_is_left_at_the_full_profile():
    """One job must stay byte-identical to what has trained this list before."""
    solo = train_chunked._chunk_env(["AAPL"], False, 1)
    assert "GTRADE_TF_POOL_PCT" not in solo or solo["GTRADE_TF_POOL_PCT"] == os.environ.get("GTRADE_TF_POOL_PCT")


def test_the_chunk_environment_names_only_that_chunks_assets():
    env = train_chunked._chunk_env(["AAPL", "MSFT"], True, 2)
    assert env["GTRADE_ASSETS"] == "AAPL,MSFT"
    assert env["GTRADE_FORCE_PROMOTE"] == "1"


def test_the_pool_base_comes_from_the_campaign_not_a_local_number(monkeypatch):
    # Two launchers of one trainer disagreeing about the load is how the same
    # arm took 27 percent longer purely because of who started it. The chunked
    # retrain now reads the same base the unattended campaign runs at.
    monkeypatch.delenv("GTRADE_TF_POOL_PCT", raising=False)
    assert train_chunked._pool_base() == float(auto_loop.CAMPAIGN["GTRADE_TF_POOL_PCT"])


def test_an_explicit_pool_wins_over_the_campaign(monkeypatch):
    # Somebody tuning one run should not have to edit the campaign to do it.
    monkeypatch.setenv("GTRADE_TF_POOL_PCT", "0.22")
    assert train_chunked._pool_base() == 0.22


def test_the_chunk_env_divides_the_card_between_processes(monkeypatch):
    monkeypatch.setenv("GTRADE_TF_POOL_PCT", "0.34")
    env = train_chunked._chunk_env(["AAPL"], False, jobs=2)
    assert float(env["GTRADE_TF_POOL_PCT"]) == 0.17
    # One slot per process, always: a second slot INSIDE a process is what
    # emptied 27 genomes by handing models the wrong sequence length.
    assert env["GTRADE_NEURAL_SLOTS"] == "1"
    assert int(env["GTRADE_WORKERS"]) < int(train_chunked.LIGHT_ENV["GTRADE_WORKERS"])


def test_the_chunked_retrain_asks_the_card_before_taking_two(monkeypatch):
    # The menu could still reach the 2026-08-24 stall, which is not an error
    # but a run that never finishes, because only the research path checked.
    monkeypatch.setattr(ar, "free_vram_mb", lambda: 1680)
    assert train_chunked._fit_jobs(2) == 1
    monkeypatch.setattr(ar, "free_vram_mb", lambda: 4096)
    assert train_chunked._fit_jobs(2) == 2
    # One is never widened, and never costs a call.
    assert train_chunked._fit_jobs(1) == 1


def test_only_the_first_parallel_chunk_writes_the_real_progress_file(monkeypatch):
    # Two trainers writing ar_progress_unit.json make the per-asset ETA - on the
    # research page and on this run's status line - a mix of two chunks shown as
    # one. Same rule auto_research already follows for its own chunks.
    # The suite isolates ar_progress by exporting AR_PROGRESS_DIR (conftest), and
    # a chunk env is a copy of the environment, so that has to go first or every
    # chunk would look redirected.
    monkeypatch.delenv("AR_PROGRESS_DIR", raising=False)
    first = train_chunked._chunk_env(["AAPL"], False, jobs=2)
    rest = train_chunked._chunk_env(["MSFT"], False, jobs=2,
                                    progress_dir="/scratch/x")
    assert "AR_PROGRESS_DIR" not in first
    assert rest["AR_PROGRESS_DIR"] == "/scratch/x"
