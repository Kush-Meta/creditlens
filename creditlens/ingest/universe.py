"""The curated issuer universe.

Selected for **credit dispersion**, not market capitalisation. A corpus of
mega-cap technology names is broad but analytically flat: leverage and coverage
cluster in a narrow band, so a ratio engine and a scorecard cannot be seen to
differentiate. This universe deliberately spans the spectrum from net-cash
issuers to heavily leveraged ones, across sectors whose capital structures
differ for structural reasons (utilities and REITs carry leverage by design;
airlines and cruise operators carry it cyclically).

`profile` is a **sampling label**, not a credit rating and not an assessment.
It records why an issuer is in the universe. The actual credit view is computed
from that issuer's own filings by `creditlens.finance.scorecard`, and comparing
the two is a useful sanity check on the scorecard - not a target to fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Profile = Literal["net_cash", "investment_grade", "leveraged", "structurally_levered", "cyclical_stressed"]

PROFILE_NOTES: dict[str, str] = {
    "net_cash": "cash and investments typically exceed debt",
    "investment_grade": "moderate leverage, comfortable coverage",
    "leveraged": "debt-funded M&A or capital return; leverage a live question",
    "structurally_levered": "asset-backed model where high leverage is normal (utilities, REITs)",
    "cyclical_stressed": "leverage driven by cycle or shock; coverage can be thin",
}


@dataclass(frozen=True)
class Issuer:
    ticker: str
    name: str
    sector: str
    profile: Profile
    note: str = ""


UNIVERSE: tuple[Issuer, ...] = (
    # --- technology: net cash ------------------------------------------
    Issuer("MSFT", "Microsoft", "Technology", "net_cash"),
    Issuer("AAPL", "Apple", "Technology", "net_cash"),
    Issuer("GOOGL", "Alphabet", "Technology", "net_cash"),
    Issuer("NVDA", "NVIDIA", "Technology", "net_cash"),
    Issuer("ADBE", "Adobe", "Technology", "net_cash"),
    # --- technology & telecom: leveraged --------------------------------
    Issuer("ORCL", "Oracle", "Technology", "leveraged", "debt-funded acquisition and AI capex"),
    Issuer("T", "AT&T", "Telecom", "leveraged"),
    Issuer("VZ", "Verizon", "Telecom", "leveraged"),
    Issuer("CHTR", "Charter Communications", "Telecom", "leveraged", "classic high-leverage cable capital structure"),
    Issuer("CMCSA", "Comcast", "Telecom", "investment_grade"),
    # --- industrials -----------------------------------------------------
    Issuer("CAT", "Caterpillar", "Industrials", "investment_grade", "captive finance arm"),
    Issuer("HON", "Honeywell", "Industrials", "investment_grade"),
    Issuer("GE", "GE Aerospace", "Industrials", "investment_grade"),
    Issuer("LMT", "Lockheed Martin", "Industrials", "investment_grade"),
    Issuer("BA", "Boeing", "Industrials", "cyclical_stressed", "negative equity and heavy debt load"),
    # --- autos ------------------------------------------------------------
    Issuer("F", "Ford Motor", "Autos", "leveraged", "captive finance arm; segment-level debt tagging"),
    Issuer("GM", "General Motors", "Autos", "leveraged", "captive finance arm"),
    Issuer("TSLA", "Tesla", "Autos", "net_cash"),
    # --- retail -----------------------------------------------------------
    Issuer("WMT", "Walmart", "Retail", "investment_grade"),
    Issuer("HD", "Home Depot", "Retail", "investment_grade", "negative equity from buybacks"),
    Issuer("TGT", "Target", "Retail", "investment_grade"),
    Issuer("KSS", "Kohl's", "Retail", "cyclical_stressed", "weak comparable sales, lease-heavy"),
    Issuer("M", "Macy's", "Retail", "cyclical_stressed"),
    # --- energy -----------------------------------------------------------
    Issuer("XOM", "Exxon Mobil", "Energy", "investment_grade"),
    Issuer("CVX", "Chevron", "Energy", "investment_grade"),
    Issuer("OXY", "Occidental Petroleum", "Energy", "leveraged", "acquisition-driven leverage"),
    # --- utilities: structurally levered ----------------------------------
    Issuer("DUK", "Duke Energy", "Utilities", "structurally_levered"),
    Issuer("SO", "Southern Company", "Utilities", "structurally_levered"),
    Issuer("NEE", "NextEra Energy", "Utilities", "structurally_levered"),
    # --- REITs: structurally levered --------------------------------------
    Issuer("SPG", "Simon Property Group", "Real Estate", "structurally_levered"),
    Issuer("O", "Realty Income", "Real Estate", "structurally_levered"),
    Issuer("PLD", "Prologis", "Real Estate", "structurally_levered"),
    # --- healthcare --------------------------------------------------------
    Issuer("JNJ", "Johnson & Johnson", "Healthcare", "net_cash"),
    Issuer("PFE", "Pfizer", "Healthcare", "leveraged", "acquisition-funded debt"),
    Issuer("ABBV", "AbbVie", "Healthcare", "leveraged"),
    Issuer("CVS", "CVS Health", "Healthcare", "leveraged", "acquisition-funded debt, thin margins"),
    # --- consumer staples ---------------------------------------------------
    Issuer("KO", "Coca-Cola", "Consumer Staples", "investment_grade"),
    Issuer("PEP", "PepsiCo", "Consumer Staples", "investment_grade"),
    Issuer("KHC", "Kraft Heinz", "Consumer Staples", "leveraged", "post-merger leverage and impairments"),
    # --- travel and leisure: cyclical -----------------------------------------
    Issuer("DAL", "Delta Air Lines", "Travel & Leisure", "cyclical_stressed"),
    Issuer("UAL", "United Airlines", "Travel & Leisure", "cyclical_stressed"),
    Issuer("CCL", "Carnival", "Travel & Leisure", "cyclical_stressed", "heavily leveraged post-pandemic"),
)

BY_TICKER: dict[str, Issuer] = {issuer.ticker: issuer for issuer in UNIVERSE}
SECTORS: tuple[str, ...] = tuple(dict.fromkeys(issuer.sector for issuer in UNIVERSE))
PROFILES: tuple[str, ...] = tuple(dict.fromkeys(issuer.profile for issuer in UNIVERSE))


def tickers(
    *, sector: str | None = None, profile: str | None = None
) -> list[str]:
    return [
        issuer.ticker for issuer in UNIVERSE
        if (sector is None or issuer.sector == sector)
        and (profile is None or issuer.profile == profile)
    ]


def composition() -> dict[str, dict[str, int]]:
    """Counts by sector and by sampling profile - printed by the CLI."""
    sectors: dict[str, int] = {}
    profiles: dict[str, int] = {}
    for issuer in UNIVERSE:
        sectors[issuer.sector] = sectors.get(issuer.sector, 0) + 1
        profiles[issuer.profile] = profiles.get(issuer.profile, 0) + 1
    return {"by_sector": dict(sorted(sectors.items())),
            "by_profile": dict(sorted(profiles.items()))}
