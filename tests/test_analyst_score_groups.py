from core.analyst import score


def _row(mode, brain, direction, ret, asset="SP500", date="2026-08-20", h=1):
    return {"mode": mode, "brain": brain, "direction": direction, "realized_ret": ret,
            "conviction": 3, "asset": asset, "date": date, "horizon": h}


def test_by_group_splits_hits_and_counts_old_rows_as_solo():
    rows = [_row("solo", "a", "up", 0.01), _row("solo", "a", "up", -0.01),
            _row(None, None, "down", -0.02), _row("team", "b", "down", 0.01),
            _row("team", "b", "flat", 0.0)]
    g = score.by_group(rows, "mode")
    assert g["solo"] == {"n": 3, "hit": 2 / 3}
    assert g["team"] == {"n": 1, "hit": 0.0}
    assert score.by_group(rows, "brain")["unknown"]["n"] == 1


def test_clean_hit_drops_moves_inside_the_noise_band():
    bars = [(("2026-07-%02d" % (i + 1)), 0, 0, 0, 100.0 * (1.01 if i % 2 else 1.0))
            for i in range(30)]          # every daily move is about 1%
    rows = [_row("solo", "a", "up", 0.002), _row("solo", "a", "up", 0.02),
            _row("solo", "a", "up", -0.03)]
    out = score.clean_hit(rows, bars_of=lambda asset, date: bars)
    assert out["n"] == 2 and out["hit"] == 0.5 and round(out["noise_share"], 3) == 0.333


def test_score_prints_mode_brain_critic_and_noise_lines(monkeypatch, capsys):
    import json

    import analyst
    rows = [dict(_row("solo", "ollama:g", "down", -0.02),
                 pre_critic_json=json.dumps({"direction": "up"})),
            _row("solo", "ollama:g", "up", 0.01)]
    monkeypatch.setattr(analyst, "_score_baselines", lambda table: (rows, {}, 0, 0))
    monkeypatch.setattr(analyst, "_load_table", lambda: {"asset": {}, "class": {}})
    monkeypatch.setattr(analyst.os.path, "exists", lambda p: True)
    monkeypatch.setattr("core.analyst.score.clean_hit",
                        lambda rows, bars_of=None: {"n": 1, "hit": 1.0, "noise_share": 0.5})
    analyst.cmd_score(type("A", (), {"fields": False})())
    out = capsys.readouterr().out
    assert "mode  solo" in out and "brain ollama:g" in out
    assert "critic solo" in out and "hit before 0.0 after 1.0" in out
    assert "without noise: n=1 hit 1.000 (noise share 50%)" in out
