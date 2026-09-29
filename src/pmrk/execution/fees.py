"""Polymarket taker fee.

Documented formula (docs.polymarket.com/trading/fees): `fee = shares * feeRate * p * (1 - p)`, taker only; makers
pay nothing. Each market carries its own `feesEnabled` flag and `feeSchedule` (rate varies within a category, and
fees were switched on at different dates per category), so the rate is read per market, never from a table. The
schedule also has an `exponent` field (1 for almost all markets); it is applied as `(p * (1 - p)) ** exponent`,
which reduces to the documented formula for exponent 1.
"""

from __future__ import annotations

import polars as pl


def taker_fee_per_share(
    price: pl.Expr, *, enabled: pl.Expr | None = None, rate: pl.Expr | None = None, exponent: pl.Expr | None = None
) -> pl.Expr:
    """Fee per share bought at `price`; defaults read `fees_enabled`, `fee_rate`, `fee_exponent` columns."""
    enabled = pl.col("fees_enabled").fill_null(False) if enabled is None else enabled
    rate = pl.col("fee_rate").fill_null(0.0) if rate is None else rate
    exponent = pl.col("fee_exponent").fill_null(1.0) if exponent is None else exponent
    return pl.when(enabled).then(rate * (price * (1.0 - price)).pow(exponent)).otherwise(0.0)
