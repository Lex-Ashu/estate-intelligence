"""
Parse the raw markdown string returned by the LLM into an AdvisoryReport
Pydantic model.

Strategy:
  • Use regex to split on the bold section headers the prompt guarantees
    (e.g. **1. Property Summary**).
  • If a section is missing, fall back to an empty string so validation
    never hard-fails — the downstream PDF/UI will show a placeholder.
"""

from __future__ import annotations

import re

from report.schema import AdvisoryReport

_SECTION_FIELDS = {
    1: "property_summary",
    2: "price_interpretation",
    3: "market_trend_insights",
    4: "recommended_actions",
    5: "sources_and_references",
    6: "legal_disclaimers",
}

_HEADER_RE = re.compile(
    r"\*{1,2}(\d)\.\s+([^\*\n]+?)\*{0,2}\s*\n",
    re.IGNORECASE,
)


def parse_report(raw_text: str) -> AdvisoryReport:
    """
    Convert the LLM's raw markdown output into a validated AdvisoryReport.

    Falls back gracefully:
      • If parsing finds fewer than 6 sections the missing ones become empty strings.
      • If parsing fails entirely, the full raw text lands in `property_summary`
        so nothing is lost.
    """
    if not raw_text or not raw_text.strip():
        return AdvisoryReport(
            property_summary="Report could not be generated.",
            price_interpretation="",
            market_trend_insights="",
            recommended_actions="",
            sources_and_references="",
            legal_disclaimers="",
        )

    sections: dict[str, str] = {f: "" for f in _SECTION_FIELDS.values()}

    matches = list(_HEADER_RE.finditer(raw_text))

    if not matches:
        sections["property_summary"] = raw_text.strip()
        return AdvisoryReport(**sections)

    for i, match in enumerate(matches):
        sec_num = int(match.group(1))
        field = _SECTION_FIELDS.get(sec_num)
        if field is None:
            continue

        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[body_start:body_end].strip()

        sections[field] = body

    return AdvisoryReport(**sections)
