"""
Pydantic model representing the six-section advisory report produced by the
LangGraph agent. Each section maps 1-to-1 with the prompt structure so the
parser can locate them reliably.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AdvisoryReport(BaseModel):
    """Structured 6-section real estate advisory report."""

    property_summary: str = Field(
        description="2–3 sentence description of the property and its key features."
    )
    price_interpretation: str = Field(
        description=(
            "Analysis of whether the ML-predicted price is fair, high, or low "
            "relative to property features and current market."
        )
    )
    market_trend_insights: str = Field(
        description="How the property relates to current Indian real estate market conditions."
    )
    recommended_actions: str = Field(
        description="2–3 specific, actionable recommendations for a buyer or investor."
    )
    sources_and_references: str = Field(
        description="Data sources and model used to generate this report."
    )
    legal_disclaimers: str = Field(
        description="Standard real estate and financial advice disclaimers."
    )

    # ── Convenience helpers ───────────────────────────────────────────────────
    def sections(self) -> list[tuple[str, str]]:
        """Return ordered (title, body) pairs for rendering / PDF export."""
        return [
            ("1. Property Summary",              self.property_summary),
            ("2. Price Prediction Interpretation", self.price_interpretation),
            ("3. Market Trend Insights",          self.market_trend_insights),
            ("4. Recommended Actions",            self.recommended_actions),
            ("5. Supporting Sources & References", self.sources_and_references),
            ("6. Legal & Financial Disclaimers",  self.legal_disclaimers),
        ]

    def to_markdown(self) -> str:
        """Render the report as a markdown string."""
        lines: list[str] = []
        for title, body in self.sections():
            lines.append(f"**{title}**\n")
            lines.append(body.strip())
            lines.append("")
        return "\n".join(lines)
