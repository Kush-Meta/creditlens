"""10-K / 10-Q HTML -> clean, item-tagged sections.

Section structure is what makes retrieval good here: "what risks did
management highlight" should search Item 1A, not the whole document. Two
practical problems have to be solved to get it:

* **Table-of-contents decoys.** Every item heading appears at least twice -
  once in the TOC, once at the real section. We keep the occurrence that is
  followed by the most text.
* **Table soup.** Filing HTML is mostly nested layout tables. Naive
  `get_text()` welds numbers into unreadable runs, so tables are flattened
  cell-by-cell with separators before extraction.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

from creditlens.observability import get_logger

# Modern EDGAR filings are inline-XBRL XHTML. Parsing them with the HTML parser
# is deliberate and correct - XHTML is HTML-compatible, and the XML parser
# discards the presentational markup the section splitter relies on - so the
# advisory warning is silenced rather than worked around.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

log = get_logger(__name__)

ITEM_TITLES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "10": "Directors and Executive Officers",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
}

#: Item headings survive HTML flattening in several shapes: bare on a line,
#: prefixed by a table-cell separator, preceded by a short "PART II" label, or
#: preceded by a page number that shared a table row with the heading
#: ("30 | ITEM 1A. | Risk Factors" - Honeywell renders every heading this way,
#: and without the page-number branch its 10-K yields exactly one section).
ITEM_PATTERN = re.compile(
    r"^[\s|>]{0,6}(?:\d{1,4}\s*[\|\.\-]\s*)?(?:part\s+[ivx]+\s*[\.\|\-]?\s*)?"
    r"item\s+(\d{1,2}[A-C]?)\s*[\.\:\-–—|]?\s*(.{0,120})$",
    re.IGNORECASE | re.MULTILINE,
)

_WS = re.compile(r"[ \t\u00a0\u2007\u202f]+")
_BLANK = re.compile(r"\n{3,}")
_PAGE_NOISE = re.compile(
    r"^\s*(table of contents|page\s*\d+|\d+\s*)$", re.IGNORECASE | re.MULTILINE
)


@dataclass
class Section:
    item_code: str | None
    title: str
    text: str
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_code": self.item_code,
            "title": self.title,
            "chars": len(self.text),
            "char_start": self.char_start,
        }


def html_to_text(html: str) -> str:
    """Flatten filing HTML into readable text, preserving table structure."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()

    for table in soup.find_all("table"):
        rows: list[str] = []
        for tr in table.find_all("tr"):
            cells = [
                _WS.sub(" ", td.get_text(" ", strip=True))
                for td in tr.find_all(["td", "th"])
            ]
            cells = [c for c in cells if c not in ("", "$", ")", "(", "%")]
            if cells:
                rows.append(" | ".join(cells))
        replacement = "\n".join(rows)
        table.replace_with(NavigableString("\n" + replacement + "\n"))

    for tag in soup.find_all(["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4"]):
        tag.append(NavigableString("\n"))

    text = soup.get_text("\n")
    text = text.replace("\u00a0", " ")
    text = _WS.sub(" ", text)
    text = _PAGE_NOISE.sub("", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK.sub("\n\n", text).strip()


#: Fallback headings for filers that introduce sections by title alone and
#: never repeat the item number in the body (GE, Honeywell, Chevron). These are
#: applied ONLY when the item-number pass comes up empty for a document: used
#: as peers they fire on cross-references and contents rows and make good
#: parses worse, which is exactly what happened when they were tried first.
TITLE_HEADINGS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("1", re.compile(r"^[\s|>0-9\.\-]{0,12}business[\s\|\d\-–—]*$", re.IGNORECASE | re.MULTILINE)),
    ("1A", re.compile(r"^[\s|>0-9\.\-]{0,12}risk factors[\s\|\d\-–—\.]*$", re.IGNORECASE | re.MULTILINE)),
    ("7", re.compile(
        r"^[\s|>0-9\.\-]{0,12}management.{0,3}s discussion and analysis.{0,90}$",
        re.IGNORECASE | re.MULTILINE)),
    ("7A", re.compile(
        r"^[\s|>0-9\.\-]{0,12}quantitative and qualitative disclosures.{0,70}$",
        re.IGNORECASE | re.MULTILINE)),
    ("8", re.compile(
        r"^[\s|>0-9\.\-]{0,12}financial statements and supplementary data[\s\|\d\-–—\.]*$",
        re.IGNORECASE | re.MULTILINE)),
)

#: Below this many surviving item headings, the item-number pass is treated as
#: having failed and the title fallback is used instead.
_MIN_ITEM_HEADINGS = 3

#: Canonical 10-K item order, used to reject out-of-sequence matches.
CANONICAL_ORDER: dict[str, int] = {
    code: rank for rank, code in enumerate([
        "1", "1A", "1B", "1C", "2", "3", "4", "4A", "5", "6", "7", "7A", "8",
        "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16",
    ])
}

#: Where the audited financial statements begin. Several filers place them
#: after the last numbered item, which would otherwise swallow them into a
#: meaningless trailing section.
FINANCIAL_STATEMENTS_MARKER = re.compile(
    r"(?im)^\s*(report of independent registered public accounting firm"
    r"|consolidated (income )?statements? of (income|operations)"
    r"|consolidated balance sheets?)\s*$"
)

#: A match followed almost immediately by another match is a table-of-contents
#: line, not a section heading.
_TOC_MAX_GAP = 500


def split_sections(text: str, *, min_section_chars: int = 700) -> list[Section]:
    """Split flattened filing text into item-coded sections.

    The naive approach - treat every "Item N" match as a boundary - fails on
    real filings in two distinct ways, both handled here:

    1. **Table of contents.** Every item appears in the TOC before it appears
       as a heading. TOC lines are identified by their tiny gap to the next
       match and dropped.
    2. **Running page headers.** Microsoft (among others) prints "Item 7" at
       the top of every page of the MD&A, producing dozens of matches inside
       one section. Taking the *first* surviving occurrence per item, and
       ending each section at the *next selected section* rather than the next
       raw match, keeps the section whole instead of slicing it into pages.
    """
    matches: list[tuple[str, int]] = [
        (m.group(1).upper(), m.start()) for m in ITEM_PATTERN.finditer(text)
    ]
    if not matches:
        return [Section(None, "Full document", text, 0, len(text))]

    # 1. remove table-of-contents entries, using two filters that cover each
    #    other's blind spots:
    #      a) drop the contents *block* - a dense run spanning many items.
    #         Per-line filtering misses sparse TOC rows, which then get chosen
    #         as the "first occurrence" of their item so the real heading later
    #         in the document is never considered (Exxon, Chevron, Pfizer).
    #      b) drop individual closely-spaced lines, which catches short
    #         contents lists too small to register as a block.
    #    An item appearing exactly once is exempt from (b): a filer rendering
    #    "ITEM 7." alone on a line produces a short gap but is still the real
    #    heading, and dropping it would lose the entire MD&A.
    toc_start, toc_end = _toc_span(matches)
    toc_range = (
        (matches[toc_start][1], matches[toc_end - 1][1])
        if toc_end > toc_start else None
    )
    after_toc = matches[toc_end:]

    occurrences: dict[str, int] = {}
    for code, _position in after_toc:
        occurrences[code] = occurrences.get(code, 0) + 1

    headings: list[tuple[str, int]] = []
    for i, (code, position) in enumerate(after_toc):
        next_start = after_toc[i + 1][1] if i + 1 < len(after_toc) else len(text)
        if next_start - position >= _TOC_MAX_GAP or occurrences[code] == 1:
            headings.append((code, position))
    # The title fallback is checked before any last-resort restore, or a
    # document whose body carries no item numbers at all would fall back to
    # its own table of contents and produce nothing usable.
    used_titles = False
    if len(headings) < _MIN_ITEM_HEADINGS:
        fallback = _title_headings(text, exclude=toc_range)
        if len(fallback) > len(headings):
            log.info("item numbers absent from body; using title headings",
                     extra={"found": [c for c, _ in fallback]})
            headings, used_titles = fallback, True
    if not headings:
        headings = after_toc or list(matches)

    # 2. first surviving occurrence wins; later ones are page headers
    first_seen: dict[str, int] = {}
    for code, position in headings:
        first_seen.setdefault(code, position)

    ordered = sorted(first_seen.items(), key=lambda kv: kv[1])

    # 3. keep the largest set of items that appear in canonical order.
    #    A greedy left-to-right scan fails badly here: one stray "Item 16"
    #    cross-reference near the top of the document would reject every real
    #    section after it. The longest increasing subsequence by canonical
    #    rank drops the stray instead.
    # Title-derived headings skip the canonical-order filter: filers that
    # introduce sections by title also tend to present them out of the standard
    # order (GE puts MD&A ahead of Risk Factors), and enforcing sequence there
    # would discard most of what the fallback just recovered.
    kept = ordered if used_titles else _longest_ordered_run(ordered)
    if not kept:
        return [Section(None, "Full document", text, 0, len(text))]

    # 4. each section runs to the next selected heading
    boundaries = [position for _, position in kept] + [len(text)]
    sections: list[Section] = []
    for index, (code, position) in enumerate(kept):
        end = boundaries[index + 1]
        body = text[position:end].strip()
        if len(body) < min_section_chars:
            continue
        sections.append(Section(
            item_code=code,
            title=ITEM_TITLES.get(code, f"Item {code}"),
            text=body,
            char_start=position,
            char_end=end,
        ))

    sections = _split_out_financial_statements(text, sections)

    if not sections:
        return [Section(None, "Full document", text, 0, len(text))]

    covered = sum(len(s.text) for s in sections)
    log.info(
        "filing sections extracted",
        extra={
            "sections": len(sections),
            "items": [s.item_code for s in sections],
            "coverage_pct": round(covered / max(len(text), 1) * 100, 1),
        },
    )
    return sections


def _title_headings(
    text: str, *, exclude: tuple[int, int] | None = None
) -> list[tuple[str, int]]:
    """Locate sections by their title, for filers that omit the item number.

    Searches the whole document minus the contents block, because the block is
    not always at the front: GE and Honeywell publish their item cross-index at
    the *end* of the 10-K, so anchoring the search after it finds nothing.
    """
    found: list[tuple[str, int]] = []
    for code, pattern in TITLE_HEADINGS:
        for match in pattern.finditer(text):
            position = match.start()
            if exclude and exclude[0] <= position <= exclude[1]:
                continue
            found.append((code, position))
            break  # first occurrence outside the contents block
    return sorted(found, key=lambda pair: pair[1])


def _toc_span(matches: list[tuple[str, int]], min_items: int = 6) -> tuple[int, int]:
    """Index range of the contents block, as (start, end-exclusive).

    A contents block is a run of closely-spaced matches covering many distinct
    items. Real section headings are separated by thousands of characters of
    prose. The block is usually at the front of the filing but not always -
    several filers publish the item cross-index at the back - so the caller
    gets a span to exclude rather than a point to start after.
    """
    index = 0
    while index < len(matches):
        end = index
        seen = {matches[index][0]}
        while (
            end + 1 < len(matches)
            and matches[end + 1][1] - matches[end][1] < _TOC_MAX_GAP
        ):
            end += 1
            seen.add(matches[end][0])
        if len(seen) >= min_items:
            return index, end + 1
        index = end + 1
    return 0, 0  # no contents block found; every match is a candidate heading


def _longest_ordered_run(candidates: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Longest position-ordered subsequence whose canonical ranks increase."""
    ranked = [
        (code, position, CANONICAL_ORDER[code])
        for code, position in candidates
        if code in CANONICAL_ORDER
    ]
    if not ranked:
        return []
    n = len(ranked)
    best = [1] * n
    previous = [-1] * n
    for i in range(n):
        for j in range(i):
            if ranked[j][2] < ranked[i][2] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
                previous[i] = j
    end = max(range(n), key=lambda i: best[i])
    chain: list[tuple[str, int]] = []
    while end != -1:
        code, position, _rank = ranked[end]
        chain.append((code, position))
        end = previous[end]
    return list(reversed(chain))


def _split_out_financial_statements(
    text: str, sections: list[Section]
) -> list[Section]:
    """Carve the audited statements out of whichever section swallowed them."""
    if not sections:
        return sections
    if any(s.item_code == "8" for s in sections):
        return sections
    marker = FINANCIAL_STATEMENTS_MARKER.search(text)
    if marker is None:
        return sections

    position = marker.start()
    out: list[Section] = []
    for section in sections:
        if not (section.char_start < position < section.char_end):
            out.append(section)
            continue
        head = text[section.char_start:position].strip()
        if len(head) >= 700:
            out.append(Section(section.item_code, section.title, head,
                               section.char_start, position))
        out.append(Section(
            "8", ITEM_TITLES["8"], text[position:section.char_end].strip(),
            position, section.char_end,
        ))
    return sorted(out, key=lambda s: s.char_start)


def parse_filing_html(html: str) -> list[Section]:
    return split_sections(html_to_text(html))
