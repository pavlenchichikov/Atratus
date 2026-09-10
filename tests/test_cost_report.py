"""The slippage sign convention, which is the whole measurement."""
import cost_report as C


class _Con:
    """A market.db stand-in: every asset closed at 100 on every date."""

    def execute(self, sql, params):
        class R:
            def fetchone(_s):
                return (100.0,)
        return R()


def _row(side, price, exit_price=None):
    return {"asset": "SBER", "side": side, "price": price,
            "entry_date": "2026-09-01", "exit_date": "2026-09-05" if exit_price else None,
            "exit_price": exit_price}


def test_buying_above_the_close_costs_money():
    legs = C.legs([_row("BUY", 101.0)], _Con())
    assert legs[0][3] > 0 and abs(legs[0][3] - 0.01) < 1e-9


def test_selling_below_the_close_also_costs_money():
    """Positive control on the sign. Raw differences would cancel: a buy above
    and a sell below are both losses, and averaging them unsigned reports zero
    slippage on a book that is bleeding."""
    legs = C.legs([_row("BUY", 100.0, exit_price=99.0)], _Con())
    exit_leg = next(x for x in legs if x[2] == "exit")
    assert exit_leg[3] > 0 and abs(exit_leg[3] - 0.01) < 1e-9


def test_a_short_is_measured_the_other_way_round():
    assert C.legs([_row("SELL", 99.0)], _Con())[0][3] > 0
    assert C.legs([_row("SELL", 101.0)], _Con())[0][3] < 0


def test_a_perfect_fill_costs_nothing():
    assert abs(C.legs([_row("BUY", 100.0)], _Con())[0][3]) < 1e-12


def test_required_accuracy_rises_with_cost_and_falls_with_horizon():
    cheap, dear = C.implied(0.001), C.implied(0.005)
    for h in ("day", "week", "month"):
        assert dear[h] > cheap[h]
    assert cheap["day"] > cheap["week"] > cheap["month"]


def test_the_measured_moves_are_the_ones_from_the_panel():
    assert C.MOVES["day"] < C.MOVES["week"] < C.MOVES["month"]
    assert 0.5 < C.ACCURACY < 0.6
