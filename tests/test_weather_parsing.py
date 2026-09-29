from datetime import date

import numpy as np
import polars as pl
import pytest

from pmrk.weather.emos import interval_prob
from pmrk.weather.markets import parse_icao, parse_label, parse_local_date, parse_source, parse_unit
from pmrk.weather.metar import decode, parse_metar, parse_sky_wind
from pmrk.weather.model import bucket_edges_c
from pmrk.weather.settlement import MetarSettlement, round_half_up, to_f


def test_us_metar_with_tgroup():
    r = parse_metar("KLGA 010051Z 23010G19KT 10SM FEW050 M01/M12 A2976 RMK AO2 SLP077 T10061117")
    assert r == {"temp_c": -1.0, "temp_c_precise": -0.6, "max6h_c": None}


def test_rvr_and_fraction_visibility_not_confused_with_temp():
    r = parse_metar("KLGA 011102Z 31031G50KT 1 1/2SM R04/4500VP6000FT -SN BKN032 01/M07 A2956 RMK AO2 T00111072")
    assert r["temp_c"] == 1.0 and r["temp_c_precise"] == 1.1


def test_international_metar_and_missing_groups():
    assert parse_metar("LFPB 201200Z 24008KT CAVOK 23/11 Q1018 NOSIG") == {
        "temp_c": 23.0,
        "temp_c_precise": None,
        "max6h_c": None,
    }
    assert parse_metar("KSEA 012353Z 00000KT 10SM CLR 15/ A3001 RMK AO2 10156 20061 T01500061")["max6h_c"] == 15.6
    assert parse_metar("EGLC 011250Z AUTO 22010KT 9999 NCD ///// Q1010")["temp_c"] is None


def test_sky_wind_parser():
    assert parse_sky_wind("UUWW 241051Z 28005MPS 9999 BKN044CB 23/10 Q1017") == (0.75, True, 5 * 1.943844)
    assert parse_sky_wind("LFPB 201200Z 24008KT CAVOK 23/11 Q1018 NOSIG") == (0.0, False, 8.0)
    assert parse_sky_wind("KXXX 011200Z VRB03KT 10SM 20/10 A3000 RMK OVC") == (None, False, 3.0)


def test_routine_minute_inferred_from_share():
    valid = [f"2026-06-10 {h:02d}:51" for h in range(10)] + ["2026-06-10 03:17"]
    raw = pl.DataFrame({"station_icao": "KXXX", "valid": valid, "metar": [f"KXXX R{i} 20/10" for i in range(11)]})
    obs = decode(raw)
    assert obs.filter(pl.col("minute") == 17)["is_routine"].to_list() == [False]
    assert obs.filter(pl.col("minute") == 51)["is_routine"].all()


def test_labels():
    assert parse_label("44-45°F") == (44, 45)
    assert parse_label("26°C or below") == (None, 26)
    assert parse_label("36°C or higher") == (36, None)
    assert parse_label("-3°C") == (-3, -3)
    assert parse_label("<40°F") == (None, 39)
    assert parse_label("61–62 °F") == (61, 62)
    with pytest.raises(ValueError):
        parse_label("warm")


def test_rules_from_description():
    desc = (
        "This market will resolve to the highest temperature recorded at the LaGuardia Airport Station on "
        "22 Nov '25. The resolution source for this market will be Wunderground, specifically "
        "https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA."
    )
    assert parse_icao(desc, None) == "KLGA" and parse_source(desc) == "wu"
    assert parse_local_date(desc, "2025-11-23") == date(2025, 11, 22)
    assert parse_unit(["50-51°F"], desc) == "F" and parse_unit(["12°C"], desc) == "C"
    assert parse_source("resolution source for this market will be NOAA ...") == "noaa"
    assert parse_icao("see https://www.weather.gov/wrh/timeseries?site=EGLC now", None) == "EGLC"


def test_f_rounding_half_up_from_tenths_c():
    s = pl.DataFrame({"c": [19.4, 2.5, -0.6]}).select(f=round_half_up(to_f(pl.col("c"))))["f"].to_list()
    assert s == [67.0, 37.0, 31.0]


def _edges(unit, lo, hi):
    df = pl.DataFrame(
        {"unit": [unit], "lo": [lo], "hi": [hi]}, schema={"unit": pl.Utf8, "lo": pl.Int64, "hi": pl.Int64}
    )
    a, b = bucket_edges_c(pl.col("unit"), pl.col("lo"), pl.col("hi"))
    return df.select(a=a, b=b).row(0)


def test_bucket_edges():
    a, b = _edges("F", 64, 65)
    assert np.isclose(a * 1.8 + 32, 63.5) and np.isclose(b * 1.8 + 32, 65.5)
    a, b = _edges("C", None, 26)
    assert a < -500 and b == 26.5


def test_truncation_mass_goes_to_bucket_with_observed_max():
    a, b = np.array([23.5, 24.5, 25.5]), np.array([24.5, 25.5, 26.5])
    p = interval_prob(a, b, np.full(3, 20.0), np.full(3, 1.0), np.full(3, 25.0))
    assert p[0] == 0.0 and p[1] > 0.999 and p[2] < 1e-6


def test_pre_day_probs_sum_to_one():
    edges = np.arange(10.5, 30.5, 1.0)
    a, b = np.concatenate([[-1e3], edges]), np.concatenate([edges, [1e3]])
    p = interval_prob(a, b, np.full(len(a), 20.3), np.full(len(a), 1.7), np.full(len(a), -np.inf))
    assert np.isclose(p.sum(), 1.0)


def test_metar_settlement_reproduces_bucket_per_unit():
    daily = pl.DataFrame(
        {
            "station_icao": ["KXXX", "EGLC"],
            "local_date": [date(2026, 6, 10)] * 2,
            "n_obs": [24, 24],
            "max_f_from_precise": [67.0, None],
            "max_c_int": [None, 21.0],
        }
    )
    rules = pl.DataFrame(
        {
            "market_id": ["f1", "f2", "c1", "c2", "h1"],
            "station_icao": ["KXXX", "KXXX", "EGLC", "EGLC", "VHHH"],
            "source": ["wu", "wu", "noaa", "noaa", "hko"],
            "unit": ["F", "F", "C", "C", "C"],
            "local_date": [date(2026, 6, 10)] * 5,
            "bucket_low": [66, 68, 21, None, 20],
            "bucket_high": [67, 69, 21, 20, 20],
        }
    )
    out = MetarSettlement(daily, rules).reproduce(rules.select("market_id")).sort("market_id")
    assert dict(zip(out["market_id"], out["winner_reproduced"], strict=True)) == {"c1": 0, "c2": 1, "f1": 0, "f2": 1}
