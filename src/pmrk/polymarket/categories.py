"""Map Polymarket tag slugs to coarse research categories.

Tag *slugs* are stable machine identifiers (not display text), so set membership on them is language-independent.
The default table was built from the observed tag frequency table of all closed markets since 2025-01-01; order
matters (first match wins), e.g. 5m/15m/1h crypto "Up or Down" markets are split out before generic crypto.
Pass your own `mapping` to override.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

CategoryMap = Sequence[tuple[str, frozenset[str]]]

DEFAULT_CATEGORIES: CategoryMap = [
    ("crypto_updown", frozenset({"up-or-down"})),
    (
        "crypto",
        frozenset(
            {"crypto", "crypto-prices", "bitcoin", "ethereum", "solana", "xrp", "dogecoin", "hype", "bnb", "zcash"}
        ),
    ),
    ("weather", frozenset({"weather", "daily-temperature", "highest-temperature", "lowest-temperature", "climate"})),
    ("mentions", frozenset({"mention-markets", "mentions", "tweets-markets", "elon-tweets"})),
    (
        "esports",
        frozenset(
            {
                "esports",
                "counter-strike-2",
                "league-of-legends",
                "dota-2",
                "valorant",
                "cs2",
                "lol",
                "rainbow-six-siege",
                "honor-of-kings",
                "mobile-legends-bang-bang",
                "call-of-duty",
                "overwatch",
                "starcraft-2",
                "rocket-league",
            }
        ),
    ),
    (
        "sports",
        frozenset(
            {
                "sports",
                "games",
                "soccer",
                "tennis",
                "basketball",
                "baseball",
                "hockey",
                "cricket",
                "nba",
                "nfl",
                "mlb",
                "nhl",
                "ncaa",
                "ufc",
                "golf",
                "f1",
                "formula1",
                "table-tennis",
                "darts",
                "rugby",
            }
        ),
    ),
    (
        "finance",
        frozenset(
            {
                "finance",
                "equities",
                "stocks",
                "indicies",
                "commodities",
                "forex",
                "earnings",
                "stock-prices",
                "etf",
                "finance-updown",
                "pyth-finance",
                "business",
            }
        ),
    ),
    (
        "economics",
        frozenset({"economy", "fed", "fed-rates", "inflation", "economic-policy", "jobs", "gdp", "interest-rates"}),
    ),
    ("tech", frozenset({"tech", "ai", "big-tech", "openai", "science", "space"})),
    (
        "culture",
        frozenset(
            {
                "pop-culture",
                "movies",
                "music",
                "awards",
                "celebrities",
                "youtube",
                "netflix",
                "top-netflix",
                "tv",
                "culture",
            }
        ),
    ),
    (
        "geopolitics",
        frozenset(
            {"geopolitics", "world", "middle-east", "iran", "israel", "ukraine", "russia", "china", "gaza", "war"}
        ),
    ),
    (
        "politics",
        frozenset(
            {
                "politics",
                "elections",
                "trump",
                "trump-presidency",
                "global-elections",
                "us-presidential-election",
                "primaries",
                "primary-elections",
            }
        ),
    ),
]


def categorize(tag_slugs: Iterable[str] | None, mapping: CategoryMap = DEFAULT_CATEGORIES) -> str:
    """First category whose slug set intersects the event's tag slugs, else `other`."""
    tags = {t.lower() for t in tag_slugs or ()}
    return next((name for name, keys in mapping if tags & keys), "other")
