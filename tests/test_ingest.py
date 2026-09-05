"""Ingestion: XBRL normalization, fiscal calendars, section parsing, chunking.

The cases here are drawn from failure modes observed against real SEC filings,
not invented ones - each one broke the pipeline before it was fixed.
"""
from __future__ import annotations

import datetime as dt

import pytest

from creditlens.ingest import chunker, filing_parser
from creditlens.ingest.xbrl_normalizer import (
    FiscalCalendar,
    infer_calendar,
    normalize_company_facts,
)


def _facts_payload(entries: list[dict], tag: str = "Revenues") -> dict:
    return {"facts": {"us-gaap": {tag: {"units": {"USD": entries}}}}}


def _entry(start, end, val, fy, fp, form="10-Q", accn="a", filed=None):
    payload = {"end": end, "val": val, "fy": fy, "fp": fp, "form": form, "accn": accn}
    if start:
        payload["start"] = start
    payload["filed"] = filed or end
    return payload


class TestFiscalCalendar:
    def test_june_year_end_rolls_quarters_forward(self):
        """A quarter ending September belongs to the *next* fiscal year."""
        calendar = FiscalCalendar(fye_month=6, year_offset=0)
        assert calendar.fiscal_year(dt.date(2025, 9, 30)) == 2026
        assert calendar.fiscal_period(dt.date(2025, 9, 30), False) == "Q1"
        assert calendar.fiscal_year(dt.date(2026, 6, 30)) == 2026
        assert calendar.fiscal_period(dt.date(2026, 6, 30), False) == "Q4"

    def test_calendar_year_end(self):
        calendar = FiscalCalendar(fye_month=12)
        assert calendar.fiscal_year(dt.date(2025, 3, 31)) == 2025
        assert calendar.fiscal_period(dt.date(2025, 3, 31), False) == "Q1"
        assert calendar.fiscal_period(dt.date(2025, 12, 31), False) == "Q4"

    def test_retail_calendar_offset(self):
        """FY ending January 2025 is 'fiscal 2024' at most retailers."""
        calendar = FiscalCalendar(fye_month=1, year_offset=-1)
        assert calendar.fiscal_year(dt.date(2025, 1, 31)) == 2024
        assert calendar.fiscal_year(dt.date(2024, 11, 2)) == 2024

    def test_tolerates_52_53_week_drift(self):
        """Apple-style year ends drift by days; the quarter must still resolve."""
        calendar = FiscalCalendar(fye_month=9)
        assert calendar.fiscal_period(dt.date(2022, 10, 1), False) == "Q4"
        assert calendar.fiscal_period(dt.date(2022, 12, 31), False) == "Q1"

    def test_inference_from_payload(self):
        payload = _facts_payload([
            _entry("2024-07-01", "2025-06-30", 400, 2025, "FY", "10-K", "k1", "2025-08-01"),
            _entry("2023-07-01", "2024-06-30", 350, 2024, "FY", "10-K", "k0", "2024-08-01"),
        ])
        calendar = infer_calendar(payload)
        assert calendar.fye_month == 6
        assert calendar.calibrated is True
        assert calendar.year_offset == 0


class TestNormalization:
    def test_comparatives_are_not_filed_under_the_filing_year(self):
        """`fy` describes the filing, not the period - the classic XBRL trap."""
        payload = _facts_payload([
            _entry("2023-01-01", "2023-12-31", 100, 2025, "FY", "10-K", "k2", "2026-02-10"),
            _entry("2025-01-01", "2025-12-31", 130, 2025, "FY", "10-K", "k2", "2026-02-10"),
        ])
        facts = normalize_company_facts(payload)
        by_year = {f.fiscal_year: f.value for f in facts if f.fiscal_period == "FY"}
        assert by_year == {2023: 100.0, 2025: 130.0}

    def test_year_to_date_spans_are_not_stored_as_quarters(self):
        payload = _facts_payload([
            _entry("2024-01-01", "2024-03-31", 100, 2024, "Q1"),
            _entry("2024-01-01", "2024-06-30", 210, 2024, "Q2"),
        ])
        facts = normalize_company_facts(payload)
        values = {(f.fiscal_period, f.value) for f in facts}
        assert ("Q1", 100.0) in values
        # Q2 must be the differenced 110, never the cumulative 210
        assert ("Q2", 110.0) in values
        assert ("Q2", 210.0) not in values

    def test_q4_is_reconstructed_from_the_annual_figure(self):
        payload = _facts_payload([
            _entry("2024-01-01", "2024-03-31", 100, 2024, "Q1"),
            _entry("2024-04-01", "2024-06-30", 110, 2024, "Q2"),
            _entry("2024-07-01", "2024-09-30", 120, 2024, "Q3"),
            _entry("2024-01-01", "2024-12-31", 470, 2024, "FY", "10-K", "k", "2025-02-01"),
        ])
        facts = normalize_company_facts(payload)
        q4 = [f for f in facts if f.fiscal_period == "Q4"]
        assert len(q4) == 1
        assert q4[0].value == pytest.approx(140.0)
        assert q4[0].source == "derived-q4"

    def test_q4_not_invented_for_balance_sheet_items(self):
        payload = _facts_payload([
            {"end": "2024-12-31", "val": 900, "fy": 2024, "fp": "FY",
             "form": "10-K", "accn": "k", "filed": "2025-02-01"},
        ], tag="Assets")
        facts = normalize_company_facts(payload)
        assert all(f.source != "derived-q4" for f in facts)

    def test_year_end_balance_appears_under_both_q4_and_fy(self):
        payload = _facts_payload([
            {"end": "2024-12-31", "val": 900, "fy": 2024, "fp": "FY",
             "form": "10-K", "accn": "k", "filed": "2025-02-01"},
        ], tag="Assets")
        facts = normalize_company_facts(payload)
        periods = {f.fiscal_period for f in facts}
        assert {"Q4", "FY"} <= periods

    def test_restatement_supersedes_the_original(self):
        payload = _facts_payload([
            _entry("2024-01-01", "2024-12-31", 100, 2024, "FY", "10-K", "old", "2025-02-01"),
            _entry("2024-01-01", "2024-12-31", 105, 2024, "FY", "10-K", "new", "2026-02-01"),
        ])
        facts = [f for f in normalize_company_facts(payload) if f.fiscal_period == "FY"]
        assert facts[0].value == 105.0
        assert facts[0].accession == "new"

    def test_higher_priority_alias_wins(self):
        payload = {"facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [
                _entry("2024-01-01", "2024-12-31", 100, 2024, "FY", "10-K", "k", "2025-02-01")]}},
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                _entry("2024-01-01", "2024-12-31", 101, 2024, "FY", "10-K", "k", "2025-02-01")]}},
        }}}
        facts = [f for f in normalize_company_facts(payload)
                 if f.concept == "revenue" and f.fiscal_period == "FY"]
        assert facts[0].value == 101.0

    def test_unmapped_tags_are_ignored(self):
        payload = _facts_payload([
            _entry("2024-01-01", "2024-12-31", 1, 2024, "FY")], tag="SomeUnknownTag")
        assert normalize_company_facts(payload) == []


class TestFilingParser:
    def test_table_of_contents_entries_are_rejected(self):
        text = (
            "Item 1. Business 3\nItem 1A. Risk Factors 12\nItem 7. MD&A 30\n\n"
            + "Item 1. Business\n\n" + ("Real business prose here. " * 80)
            + "\n\nItem 1A. Risk Factors\n\n" + ("Real risk prose here. " * 80)
        )
        sections = filing_parser.split_sections(text)
        codes = [s.item_code for s in sections]
        assert codes == ["1", "1A"]
        assert len(sections[0].text) > 1000

    def test_running_page_headers_do_not_fragment_a_section(self):
        """Filers print 'Item 7' atop every MD&A page; the section stays whole."""
        page = "Item 7\n\n" + ("Management discussion prose. " * 90) + "\n\n"
        text = "Item 7. MD&A\n\n" + page * 4 + "Item 7A. Market Risk\n\n" + ("Market risk. " * 90)
        sections = filing_parser.split_sections(text)
        item7 = next(s for s in sections if s.item_code == "7")
        assert len(item7.text) > 8000

    def test_stray_high_numbered_reference_does_not_truncate_the_document(self):
        text = (
            "Item 16 is discussed later in this document for reference purposes only.\n\n"
            + ("Filler. " * 120)
            + "\n\nItem 1. Business\n\n" + ("Business prose. " * 90)
            + "\n\nItem 1A. Risk Factors\n\n" + ("Risk prose. " * 90)
            + "\n\nItem 7. MD&A\n\n" + ("MD&A prose. " * 90)
        )
        codes = [s.item_code for s in filing_parser.split_sections(text)]
        assert {"1", "1A", "7"} <= set(codes)

    def test_financial_statements_are_split_out(self):
        text = (
            "Item 15. Exhibits\n\n" + ("Exhibit list. " * 90)
            + "\n\nREPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM\n\n"
            + ("Audited statement prose. " * 200)
        )
        sections = filing_parser.split_sections(text)
        assert any(s.item_code == "8" for s in sections)

    def test_tables_are_flattened_readably(self):
        html = ("<table><tr><td>Revenue</td><td>1,000</td></tr>"
                "<tr><td>Operating income</td><td>200</td></tr></table>")
        text = filing_parser.html_to_text(html)
        assert "Revenue | 1,000" in text
        assert "Operating income | 200" in text

    def test_document_without_items_returns_one_section(self):
        sections = filing_parser.split_sections("Just some prose. " * 200)
        assert len(sections) == 1
        assert sections[0].item_code is None


class TestChunker:
    def test_respects_the_token_target(self):
        text = "\n\n".join(f"Paragraph {i} with several words in it." * 8 for i in range(30))
        chunks = chunker.chunk_text(text, target_tokens=200, overlap_tokens=40)
        assert chunks
        assert all(c.token_count <= 320 for c in chunks)

    def test_overlap_carries_context_across_the_boundary(self):
        text = "\n\n".join(f"Sentence number {i} about leverage and covenants." * 10
                           for i in range(12))
        chunks = chunker.chunk_text(text, target_tokens=120, overlap_tokens=40)
        assert len(chunks) > 2
        tail = " ".join(chunks[0].text.split()[-8:])
        assert tail.split()[0] in chunks[1].text

    def test_oversized_paragraph_is_split_on_sentences(self):
        paragraph = " ".join(f"This is sentence {i}." for i in range(200))
        chunks = chunker.chunk_text(paragraph, target_tokens=100, overlap_tokens=0)
        assert len(chunks) > 1
        assert all(c.text.strip().endswith(".") for c in chunks)

    def test_section_metadata_is_attached(self):
        section = filing_parser.Section("1A", "Risk Factors", "Risk prose. " * 200, 0, 100)
        chunks = chunker.chunk_sections([section])
        assert all(c.item_code == "1A" for c in chunks)
        assert all(c.section_title == "Risk Factors" for c in chunks)

    def test_content_hash_is_stable(self):
        chunks = chunker.chunk_text("Stable text. " * 100, target_tokens=100)
        again = chunker.chunk_text("Stable text. " * 100, target_tokens=100)
        assert [c.content_hash for c in chunks] == [c.content_hash for c in again]
