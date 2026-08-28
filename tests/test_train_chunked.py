

def test_the_pool_base_comes_from_the_campaign_not_a_local_number(monkeypatch):
    # Two launchers of one trainer disagreeing about the load is how the same
    # arm took 27 percent longer purely because of who started it. The chunked
    # retrain now reads the same base the unattended campaign runs at.
    import auto_loop
    import train_chunked as tc
    monkeypatch.delenv("GTRADE_TF_POOL_PCT", raising=False)
    assert tc._pool_base() == float(auto_loop.CAMPAIGN["GTRADE_TF_POOL_PCT"])


def test_an_explicit_pool_wins_over_the_campaign(monkeypatch):
    # Somebody tuning one run should not have to edit the campaign to do it.
    import train_chunked as tc
    monkeypatch.setenv("GTRADE_TF_POOL_PCT", "0.22")
    assert tc._pool_base() == 0.22


def test_the_chunk_env_divides_the_card_between_processes(monkeypatch):
    import train_chunked as tc
    monkeypatch.setenv("GTRADE_TF_POOL_PCT", "0.34")
    env = tc._chunk_env(["AAPL"], False, jobs=2)
    assert float(env["GTRADE_TF_POOL_PCT"]) == 0.17
    # One slot per process, always: a second slot INSIDE a process is what
    # emptied 27 genomes by handing models the wrong sequence length.
    assert env["GTRADE_NEURAL_SLOTS"] == "1"
    assert int(env["GTRADE_WORKERS"]) < int(tc.LIGHT_ENV["GTRADE_WORKERS"])


def test_a_single_job_leaves_the_whole_card_alone(monkeypatch):
    import train_chunked as tc
    env = tc._chunk_env(["AAPL"], False, jobs=1)
    assert "GTRADE_TF_POOL_PCT" not in env or env.get("GTRADE_NEURAL_SLOTS") is None \
        or env["GTRADE_WORKERS"] == tc.LIGHT_ENV["GTRADE_WORKERS"]


def test_the_chunked_retrain_asks_the_card_before_taking_two(monkeypatch):
    # The menu could still reach the 2026-08-24 stall, which is not an error
    # but a run that never finishes, because only the research path checked.
    import auto_research as ar
    import train_chunked as tc
    monkeypatch.setattr(ar, "free_vram_mb", lambda: 1680)
    assert tc._fit_jobs(2) == 1
    monkeypatch.setattr(ar, "free_vram_mb", lambda: 4096)
    assert tc._fit_jobs(2) == 2
    # One is never widened, and never costs a call.
    assert tc._fit_jobs(1) == 1
