from datetime import timedelta

from synth import T0, event, market

from pmrk.polymarket.categories import categorize
from pmrk.polymarket.markets import markets_frame, parse_ts, winner_index
from pmrk.polymarket.store import load_markets, save_markets


def test_parse_ts_formats():
    a = parse_ts("2026-03-20 13:57:49+00")
    b = parse_ts("2026-03-20T13:57:49Z")
    c = parse_ts("2026-03-20T13:57:49.30512Z")
    assert a == b and c.tzinfo is not None and parse_ts(None) is None


def test_winner_index_handles_void_and_open():
    assert winner_index('["1", "0"]', True) == 0
    assert winner_index('["0", "1"]', True) == 1
    assert winner_index('["0.5", "0.5"]', True) is None  # 50-50 / voided
    assert winner_index('["1", "0"]', False) is None  # not closed
    assert winner_index(None, True) is None


def test_clean_flag_excludes_void_disputed_and_arch_copies():
    df = markets_frame(
        [
            event("1", [market("a")]),
            event("2", [market("b", prices=("0.5", "0.5"))]),
            event("3", [market("c", statuses=("proposed", "disputed", "proposed"))]),
            event("4", [market("d")], slug="arch-event-4"),
        ]
    )
    clean = dict(zip(df["market_id"], df["clean"], strict=True))
    assert clean == {"a": True, "b": False, "c": False, "d": False}


def test_negrisk_structure_and_generic_outcome_labels():
    ev = event("9", [market("x", prices=("0", "1")), market("y"), market("z", prices=("0", "1"))], neg_risk=True)
    solo = event("8", [market("o", outcomes=("Over", "Under"), prices=("0", "1"))], tags=("nba",))
    df = markets_frame([ev, solo]).sort("market_id")
    assert df.filter(event_id="9")["structure"].unique().to_list() == ["negrisk"]
    assert df.filter(event_id="9")["markets_in_event"].unique().to_list() == [3]
    o = df.filter(market_id="o").row(0, named=True)
    assert (o["outcome0"], o["outcome1"], o["winner"], o["structure"], o["category"]) == (
        "Over",
        "Under",
        1,
        "binary",
        "sports",
    )


def test_known_ts_is_earliest_of_end_close_resolution():
    early = T0 + timedelta(days=3)  # "by date" market resolved long before its end date
    df = markets_frame([event("1", [market("a", end=T0 + timedelta(days=10), closed=early)])])
    r = df.row(0, named=True)
    assert r["known_ts"] == early and r["resolved_early_1h"]


def test_fee_schedule_parsed_per_market():
    df = markets_frame([event("1", [market("a", fees=True, rate=0.04), market("b")])]).sort("market_id")
    assert df["fees_enabled"].to_list() == [True, False]
    assert df["fee_rate"].to_list() == [0.04, 0.0]


def test_categorize_uses_slugs_first_match_wins():
    assert categorize(["up-or-down", "crypto"]) == "crypto_updown"
    assert categorize(["Crypto"]) == "crypto"
    assert categorize([]) == "other"
    assert categorize(["x"], mapping=[("custom", frozenset({"x"}))]) == "custom"


def test_save_markets_merges_and_recomputes_event_counts():
    save_markets(markets_frame([event("1", [market("a")], neg_risk=True)]))
    out = save_markets(markets_frame([event("1", [market("b")], neg_risk=True)]))
    assert sorted(out["market_id"]) == ["a", "b"]
    assert load_markets()["markets_in_event"].to_list() == [2, 2]
