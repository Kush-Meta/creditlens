"""XBRL companyfacts -> normalized, deduplicated, period-aligned facts.

This is where most of the real data engineering lives. companyfacts is a
firehose of every tagged value a filer ever reported, with three properties
that break naive ingestion:

1. **Duplicates and restatements.** The same (concept, period) appears in
   every filing that showed a comparative. We keep the latest-filed value and
   prefer the higher-priority alias, recording what was dropped.
2. **Mixed period lengths.** A single tag carries 90-day, 180-day, 270-day
   (year-to-date) and 365-day durations. Summing them silently double-counts.
   We classify each entry by its day span and keep only clean quarters and
   full years.
3. **Missing Q4.** Filers never file a 10-Q for Q4, so quarterly flow series
   have a hole. We reconstruct Q4 = FY - (Q1+Q2+Q3) for flow concepts and mark
   it `derived-q4` so nothing downstream mistakes it for a reported number.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from creditlens.finance import taxonomy
from creditlens.observability import get_logger

log = get_logger(__name__)

QUARTER_DAYS = (75, 115)
HALF_DAYS = (150, 210)
NINE_MONTH_DAYS = (240, 300)
ANNUAL_DAYS = (330, 400)
QUARTERS = ("Q1", "Q2", "Q3", "Q4")


@dataclass
class FiscalCalendar:
    """A filer's fiscal calendar, inferred from its own annual filings.

    This exists because of a trap in companyfacts: the `fy`/`fp` fields on a
    fact describe *the filing the fact appeared in*, not the period the fact
    covers. Every 10-K restates two prior years as comparatives, and all three
    carry the same `fy`. Trusting `fy` therefore files FY2023 revenue under
    FY2025 - and, because the dedup key includes the fiscal year, silently
    mixes periods inside one "year".

    So we ignore `fy` for period assignment and derive the label from the
    period end date, calibrated once per company:

    * `fye_month` comes from the modal end month of full-year durations;
    * `year_offset` is calibrated from annual facts whose `fy` is trustworthy
      (the filing's own primary period), which handles both Oracle-style
      calendars (FY ends May 2026 = FY2026) and retail calendars (FY ends
      Jan 2025 = fiscal 2024).
    """

    fye_month: int = 12
    year_offset: int = 0
    calibrated: bool = False

    #: 52/53-week filers close a few days either side of the nominal month end;
    #: a year end that slips into the following month must not roll the whole
    #: period into the next fiscal year.
    DRIFT_TOLERANCE_DAYS = 10

    @staticmethod
    def _month_end(year: int, month: int) -> dt.date:
        if month == 12:
            return dt.date(year, 12, 31)
        return dt.date(year, month + 1, 1) - dt.timedelta(days=1)

    def _closing_year(self, period_end: dt.date) -> int:
        """Calendar year of the fiscal year end that this period closes into."""
        nominal = self._month_end(period_end.year, self.fye_month)
        if period_end <= nominal:
            return period_end.year
        if (period_end - nominal).days <= self.DRIFT_TOLERANCE_DAYS:
            return period_end.year  # drifted past the month end, same fiscal year
        return period_end.year + 1

    def fiscal_year(self, period_end: dt.date) -> int:
        """Fiscal year label for a period ending on `period_end`.

        A period belongs to the fiscal year that *closes* on or after it, so a
        quarter ending September 2025 for a June-year-end filer is FY2026, not
        FY2025 - the single most common mislabelling in XBRL pipelines.
        """
        return self._closing_year(period_end) + self.year_offset

    def year_end_date(self, period_end: dt.date) -> dt.date:
        return self._month_end(self._closing_year(period_end), self.fye_month)

    def fiscal_period(self, period_end: dt.date, is_annual: bool) -> str | None:
        """Quarter label, tolerant of 52/53-week calendars.

        Distance to the fiscal year end is measured in days and snapped to the
        nearest quarter boundary, so a filer whose Q4 lands on 1 October rather
        than 30 September is still Q4 instead of being discarded.
        """
        if is_annual:
            return "FY"
        months_before = round((self.year_end_date(period_end) - period_end).days / 30.44)
        snapped = min((0, 3, 6, 9), key=lambda q: abs(q - months_before))
        if abs(snapped - months_before) > 1:
            return None
        return {9: "Q1", 6: "Q2", 3: "Q3", 0: "Q4"}[snapped]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fiscal_year_end_month": self.fye_month,
            "year_offset": self.year_offset,
            "calibrated": self.calibrated,
        }


def infer_calendar(payload: dict[str, Any]) -> FiscalCalendar:
    """Infer the fiscal calendar from annual durations in companyfacts."""
    month_counts: dict[int, int] = {}
    offset_counts: dict[int, int] = {}

    for taxonomy_name in ("us-gaap", "ifrs-full"):
        for tag_payload in payload.get("facts", {}).get(taxonomy_name, {}).values():
            for entries in tag_payload.get("units", {}).values():
                for entry in entries:
                    start = _parse_date(entry.get("start"))
                    end = _parse_date(entry.get("end"))
                    if start is None or end is None:
                        continue
                    if _classify_duration(start, end) != "annual":
                        continue
                    month_counts[end.month] = month_counts.get(end.month, 0) + 1
                    # `fy` is trustworthy only on the filing's own primary
                    # period: the annual span that ends on the report date.
                    fy, form = entry.get("fy"), entry.get("form")
                    filed = _parse_date(entry.get("filed"))
                    if (
                        form == "10-K" and isinstance(fy, int) and filed is not None
                        and 0 < (filed - end).days < 120
                    ):
                        offset = fy - end.year
                        if -1 <= offset <= 1:
                            offset_counts[offset] = offset_counts.get(offset, 0) + 1

    calendar = FiscalCalendar()
    if month_counts:
        calendar.fye_month = max(month_counts, key=lambda m: month_counts[m])
    if offset_counts:
        calendar.year_offset = max(offset_counts, key=lambda o: offset_counts[o])
        calendar.calibrated = True
    return calendar


@dataclass
class NormalizedFact:
    concept: str
    raw_concept: str
    statement: str
    value: float
    unit: str
    period_type: str
    period_start: dt.date | None
    period_end: dt.date
    fiscal_year: int
    fiscal_period: str
    accession: str | None
    filed: dt.date | None
    source: str = "sec-edgar-xbrl"
    alias_priority: int = 0
    is_synthetic: bool = False
    #: 6 or 9 for a year-to-date span staged for quarter reconstruction, else 0
    ytd_months: int = 0

    @property
    def key(self) -> tuple[str, int, str, dt.date]:
        return (self.concept, self.fiscal_year, self.fiscal_period, self.period_end)


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def _classify_duration(start: dt.date | None, end: dt.date) -> str | None:
    """Return 'quarter', 'annual', 'ytd', or None for an unusable span."""
    if start is None:
        return "instant"
    days = (end - start).days
    if QUARTER_DAYS[0] <= days <= QUARTER_DAYS[1]:
        return "quarter"
    if ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]:
        return "annual"
    if HALF_DAYS[0] <= days <= HALF_DAYS[1] or NINE_MONTH_DAYS[0] <= days <= NINE_MONTH_DAYS[1]:
        return "ytd"
    return None


def normalize_company_facts(
    payload: dict[str, Any],
    *,
    taxonomies: tuple[str, ...] = ("us-gaap", "ifrs-full"),
    units: tuple[str, ...] = ("USD", "USD/shares", "shares"),
    min_fiscal_year: int | None = None,
) -> list[NormalizedFact]:
    facts_root = payload.get("facts", {})
    calendar = infer_calendar(payload)
    candidates: list[NormalizedFact] = []

    for taxonomy_name in taxonomies:
        for tag, tag_payload in facts_root.get(taxonomy_name, {}).items():
            resolved = taxonomy.resolve(tag)
            if resolved is None:
                continue
            concept, alias_priority = resolved
            spec = taxonomy.spec(concept)
            assert spec is not None
            for unit_name, entries in tag_payload.get("units", {}).items():
                if unit_name not in units:
                    continue
                for entry in entries:
                    fact = _entry_to_fact(
                        entry, tag, concept, spec, unit_name, alias_priority, calendar
                    )
                    if fact is None:
                        continue
                    if min_fiscal_year and fact.fiscal_year < min_fiscal_year:
                        continue
                    candidates.append(fact)

    deduped = deduplicate(candidates + mirror_year_end_instants(candidates))
    with_quarters = difference_ytd_to_quarters(deduped)
    reconstructed = reconstruct_q4(with_quarters)
    log.info(
        "xbrl normalized",
        extra={
            "raw_candidates": len(candidates),
            "deduped": len(deduped),
            "with_derived_q4": len(reconstructed),
            **calendar.to_dict(),
        },
    )
    return reconstructed


def _entry_to_fact(
    entry: dict[str, Any],
    tag: str,
    concept: str,
    spec: taxonomy.ConceptSpec,
    unit_name: str,
    alias_priority: int,
    calendar: FiscalCalendar,
) -> NormalizedFact | None:
    end = _parse_date(entry.get("end"))
    if end is None or entry.get("val") is None:
        return None
    start = _parse_date(entry.get("start"))
    kind = _classify_duration(start, end)
    if kind is None:
        return None

    if spec.period_type == "instant":
        if kind != "instant":
            return None  # a balance tag reported as a duration: skip, don't guess
        period_type = "instant"
    else:
        if kind == "instant":
            return None
        period_type = "duration"

    # Period labels are derived from the period end via the fiscal calendar -
    # never from the entry's `fy`/`fp`, which describe the filing, not the fact.
    fiscal_period = calendar.fiscal_period(end, is_annual=(kind == "annual"))
    if fiscal_period is None:
        return None
    fiscal_year = calendar.fiscal_year(end)

    ytd_months = 0
    if kind == "ytd":
        # Cash-flow and some income lines are tagged only cumulatively in
        # 10-Qs. Stage them; discrete quarters are differenced out below.
        assert start is not None
        ytd_months = 6 if (end - start).days < 225 else 9

    return NormalizedFact(
        concept=concept,
        raw_concept=tag,
        statement=spec.statement,
        value=float(entry["val"]) * spec.sign,
        unit=unit_name,
        period_type=period_type,
        period_start=start,
        period_end=end,
        fiscal_year=int(fiscal_year),
        fiscal_period=fiscal_period,
        accession=entry.get("accn"),
        filed=_parse_date(entry.get("filed")),
        alias_priority=alias_priority,
        ytd_months=ytd_months,
        source="ytd-staging" if ytd_months else "sec-edgar-xbrl",
    )


def mirror_year_end_instants(facts: Iterable[NormalizedFact]) -> list[NormalizedFact]:
    """Publish fiscal-year-end balances under both Q4 and FY.

    A balance sheet dated at the fiscal year end is simultaneously the closing
    balance of Q4 and of the year. Without this, annual ratio views would have
    income-statement data and no balance sheet.
    """
    import copy

    mirrored: list[NormalizedFact] = []
    for fact in facts:
        if fact.period_type == "instant" and fact.fiscal_period == "Q4":
            twin = copy.copy(fact)
            twin.fiscal_period = "FY"
            mirrored.append(twin)
    return mirrored


def deduplicate(facts: Iterable[NormalizedFact]) -> list[NormalizedFact]:
    """Keep one fact per (concept, fy, fp, period_end).

    Preference order: lower alias priority (the canonical tag), then the most
    recently filed value (restatements supersede originals).
    """
    best: dict[tuple, NormalizedFact] = {}
    for fact in facts:
        current = best.get(fact.key)
        if current is None:
            best[fact.key] = fact
            continue
        if (fact.alias_priority, _filed_sort(fact)) < (
            current.alias_priority, _filed_sort(current)
        ):
            best[fact.key] = fact
    return sorted(best.values(), key=lambda f: (f.concept, f.fiscal_year, f.fiscal_period))


def _filed_sort(fact: NormalizedFact) -> dt.date:
    # Negated so "most recent filed" sorts first under an ascending comparison.
    filed = fact.filed or dt.date(1900, 1, 1)
    return dt.date(9999, 12, 31) - dt.timedelta(days=(filed - dt.date(1900, 1, 1)).days)


def difference_ytd_to_quarters(facts: list[NormalizedFact]) -> list[NormalizedFact]:
    """Turn cumulative year-to-date spans into discrete quarters.

    10-Q cash-flow statements are cumulative: the Q3 filing reports nine
    months, not three. Without differencing, quarterly operating cash flow,
    capex and D&A simply do not exist, which silently removes EBITDA, free
    cash flow and every ratio built on them from the TTM view.

        Q2 = YTD(6m) - Q1
        Q3 = YTD(9m) - YTD(6m)

    Discrete quarters that the filer *did* tag always win; this only fills
    holes, and everything it adds is marked `derived-from-ytd`.
    """
    staged = [f for f in facts if f.ytd_months]
    if not staged:
        return [f for f in facts if not f.ytd_months]

    discrete = {
        (f.concept, f.fiscal_year, f.fiscal_period): f
        for f in facts if not f.ytd_months and f.period_type == "duration"
    }
    ytd_index: dict[tuple[str, int, int], NormalizedFact] = {}
    for fact in staged:
        ytd_index[(fact.concept, fact.fiscal_year, fact.ytd_months)] = fact

    derived: list[NormalizedFact] = []
    for (concept, year, months), fact in ytd_index.items():
        target_period = {6: "Q2", 9: "Q3"}[months]
        if (concept, year, target_period) in discrete:
            continue
        if months == 6:
            prior = discrete.get((concept, year, "Q1"))
            prior_value = prior.value if prior else None
            basis = "Q1"
        else:
            earlier = ytd_index.get((concept, year, 6))
            prior_value = earlier.value if earlier else None
            basis = "YTD 6M"
            if prior_value is None:
                q1 = discrete.get((concept, year, "Q1"))
                q2 = discrete.get((concept, year, "Q2"))
                if q1 and q2:
                    prior_value, basis = q1.value + q2.value, "Q1 + Q2"
        if prior_value is None:
            continue
        derived.append(NormalizedFact(
            concept=concept,
            raw_concept=fact.raw_concept,
            statement=fact.statement,
            value=fact.value - prior_value,
            unit=fact.unit,
            period_type="duration",
            period_start=fact.period_start,
            period_end=fact.period_end,
            fiscal_year=year,
            fiscal_period=target_period,
            accession=fact.accession,
            filed=fact.filed,
            source=f"derived-from-ytd ({months}M less {basis})",
            alias_priority=fact.alias_priority,
        ))

    if derived:
        log.info("differenced YTD spans into quarters", extra={"count": len(derived)})
    return [f for f in facts if not f.ytd_months] + derived


def reconstruct_q4(facts: list[NormalizedFact]) -> list[NormalizedFact]:
    """Fill missing Q4 flow values as FY minus the first three quarters."""
    by_concept_year: dict[tuple[str, int], dict[str, NormalizedFact]] = defaultdict(dict)
    for fact in facts:
        by_concept_year[(fact.concept, fact.fiscal_year)][fact.fiscal_period] = fact

    derived: list[NormalizedFact] = []
    for (concept, year), periods in by_concept_year.items():
        if "Q4" in periods or "FY" not in periods:
            continue
        if not taxonomy.is_flow(concept):
            continue
        if not all(q in periods for q in ("Q1", "Q2", "Q3")):
            continue
        annual = periods["FY"]
        partial = sum(periods[q].value for q in ("Q1", "Q2", "Q3"))
        q3_end = periods["Q3"].period_end
        derived.append(NormalizedFact(
            concept=concept,
            raw_concept=annual.raw_concept,
            statement=annual.statement,
            value=annual.value - partial,
            unit=annual.unit,
            period_type="duration",
            period_start=q3_end + dt.timedelta(days=1),
            period_end=annual.period_end,
            fiscal_year=year,
            fiscal_period="Q4",
            accession=annual.accession,
            filed=annual.filed,
            source="derived-q4",
            alias_priority=annual.alias_priority,
        ))
    if derived:
        log.info("reconstructed Q4 flows", extra={"count": len(derived)})
    return facts + derived


def carry_instants_to_quarters(facts: list[NormalizedFact]) -> list[NormalizedFact]:
    """Ensure each quarter with flow data also has the balance-sheet instants.

    Some filers tag balances only in the annual report. Rather than invent
    values, we copy the *nearest prior reported instant* forward and label it
    `carried-forward` so its staleness is visible.
    """
    instants = [f for f in facts if f.period_type == "instant"]
    if not instants:
        return facts
    by_concept: dict[str, list[NormalizedFact]] = defaultdict(list)
    for fact in instants:
        by_concept[fact.concept].append(fact)
    for series in by_concept.values():
        series.sort(key=lambda f: f.period_end)

    flow_periods = {
        (f.fiscal_year, f.fiscal_period, f.period_end)
        for f in facts if f.period_type == "duration"
    }
    existing = {(f.concept, f.fiscal_year, f.fiscal_period) for f in instants}

    added: list[NormalizedFact] = []
    for concept, series in by_concept.items():
        for year, period, end in flow_periods:
            if (concept, year, period) in existing:
                continue
            prior = [f for f in series if f.period_end <= end]
            if not prior:
                continue
            source_fact = prior[-1]
            if (end - source_fact.period_end).days > 200:
                continue
            added.append(NormalizedFact(
                concept=concept,
                raw_concept=source_fact.raw_concept,
                statement=source_fact.statement,
                value=source_fact.value,
                unit=source_fact.unit,
                period_type="instant",
                period_start=None,
                period_end=end,
                fiscal_year=year,
                fiscal_period=period,
                accession=source_fact.accession,
                filed=source_fact.filed,
                source=f"carried-forward-from-{source_fact.period_end.isoformat()}",
                alias_priority=source_fact.alias_priority,
            ))
    return facts + added
