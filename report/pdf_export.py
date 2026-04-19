"""
PDF export for the AdvisoryReport using fpdf2.

Usage:
    from report.pdf_export import generate_pdf
    pdf_bytes = generate_pdf(report, predicted_price, property_input)
    # pdf_bytes is a bytes object ready for st.download_button
"""

from __future__ import annotations

import io
import re
from datetime import datetime

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from report.schema import AdvisoryReport

BRAND_BLUE  = (26,  90, 255)   
DARK_HEADER = (10,  22,  40)   
BODY_GREY   = (45,  55,  72)   
SECTION_BG  = (235, 242, 255)  
LINE_COLOUR = (180, 200, 230)  

FONT_FAMILY = "Helvetica"      


def _sanitise(text: str) -> str:
    """
    Make text safe for Helvetica (Latin-1 only):
      • Replace ₹ with Rs.
      • Replace em-dash / en-dash with a hyphen
      • Drop any remaining non-Latin-1 characters
    """
    text = text.replace("₹", "Rs.")
    text = text.replace("\u2013", "-").replace("\u2014", "-")  
    text = text.replace("\u2019", "'").replace("\u2018", "'")  
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.encode("latin-1", errors="ignore").decode("latin-1")
    return text


def _strip_markdown(text: str) -> str:
    """Remove basic markdown formatting so plain PDF text looks clean."""
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)   
    text = re.sub(r"`(.+?)`", r"\1", text)                 
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  
    text = re.sub(r"^\s*[-•]\s+", "  - ", text, flags=re.MULTILINE) 
    return _sanitise(text.strip())


class _ReportPDF(FPDF):
    """Custom FPDF subclass with pre-styled header / footer."""

    def __init__(self, predicted_price: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._predicted_price = predicted_price
        self._generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")

    def header(self):
        self.set_fill_color(*BRAND_BLUE)
        self.rect(0, 0, 210, 18, style="F")

        self.set_font(FONT_FAMILY, "B", 11)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 4)
        self.cell(0, 10, "Estate Intelligence · Property Advisory Report",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_font(FONT_FAMILY, "", 8)
        self.set_text_color(200, 220, 255)
        self.set_xy(10, 11)
        self.cell(0, 6, f"Generated: {self._generated_at}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_y(24)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(*LINE_COLOUR)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_font(FONT_FAMILY, "I", 7.5)
        self.set_text_color(*BODY_GREY)
        self.cell(0, 8,
                  "This report is for informational purposes only and does not constitute "
                  "financial or legal advice. | Estate Intelligence",
                  align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font(FONT_FAMILY, "", 7.5)
        self.cell(0, 5, f"Page {self.page_no()}", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _format_inr(amount: float) -> str:
    """Format a number as Indian Rupees (e.g. ₹45,00,000)."""
    amount = int(round(amount))
    s = str(amount)
    if len(s) <= 3:
        return f"Rs. {s}"
    result = s[-3:]
    s = s[:-3]
    while s:
        result = s[-2:] + "," + result
        s = s[:-2]
    return f"Rs. {result}"


def generate_pdf(
    report: AdvisoryReport,
    predicted_price: float,
    property_input: dict,
) -> bytes:
    """
    Build a styled PDF for the advisory report and return it as raw bytes.

    Parameters
    ----------
    report          : parsed AdvisoryReport Pydantic model
    predicted_price : ML price output in rupees
    property_input  : the validated property feature dict
    """
    pdf = _ReportPDF(predicted_price=predicted_price, orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(left=14, top=6, right=14)
    pdf.add_page()

    pdf.set_fill_color(*SECTION_BG)
    pdf.set_draw_color(*BRAND_BLUE)
    pdf.set_line_width(0.5)
    pdf.rect(14, pdf.get_y(), 182, 22, style="DF")

    pdf.set_font(FONT_FAMILY, "B", 10)
    pdf.set_text_color(*DARK_HEADER)
    pdf.set_xy(18, pdf.get_y() + 3)
    pdf.cell(0, 6, "ML-Predicted Property Price",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(18, pdf.get_y())
    pdf.set_font(FONT_FAMILY, "B", 18)
    pdf.set_text_color(*BRAND_BLUE)
    pdf.cell(0, 10, _format_inr(predicted_price),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(8)

    furnish_map = {0: "Unfurnished", 1: "Semi-Furnished", 2: "Furnished"}
    yes_no = lambda v: "Yes" if v else "No"

    features = [
        ("Area",           f"{property_input.get('area', '-')} sq ft"),
        ("Bedrooms",       str(property_input.get("bedrooms", "-"))),
        ("Bathrooms",      str(property_input.get("bathrooms", "-"))),
        ("Stories",        str(property_input.get("stories", "-"))),
        ("Parking",        str(property_input.get("parking", "-"))),
        ("Furnishing",     furnish_map.get(property_input.get("furnishingstatus", 1), "-")),
        ("Air Conditioning", yes_no(property_input.get("airconditioning", 0))),
        ("Preferred Area", yes_no(property_input.get("prefarea", 0))),
        ("Main Road",      yes_no(property_input.get("mainroad", 1))),
        ("Basement",       yes_no(property_input.get("basement", 0))),
        ("Guest Room",     yes_no(property_input.get("guestroom", 0))),
        ("Hot Water",      yes_no(property_input.get("hotwaterheating", 0))),
    ]

    pdf.set_font(FONT_FAMILY, "B", 9)
    pdf.set_text_color(*DARK_HEADER)
    pdf.cell(0, 7, "Property Details", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*LINE_COLOUR)
    pdf.set_line_width(0.3)
    pdf.line(14, pdf.get_y(), 196, pdf.get_y())
    pdf.ln(2)

    col_w = 89
    row_h = 7
    for i, (key, val) in enumerate(features):
        if i % 2 == 0:
            pdf.set_fill_color(245, 249, 255)
        else:
            pdf.set_fill_color(255, 255, 255)

        x = 14 + (col_w + 4) * (i % 2)
        if i % 2 == 0 and i > 0:
            pdf.ln(row_h)
        if i % 2 == 0:
            pdf.set_x(x)
            pdf.set_font(FONT_FAMILY, "B", 8.5)
            pdf.set_text_color(*BODY_GREY)
            pdf.cell(44, row_h, key + ":", fill=True)
            pdf.set_font(FONT_FAMILY, "", 8.5)
            pdf.set_text_color(*DARK_HEADER)
            pdf.cell(col_w - 44, row_h, val, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        else:
            pdf.set_x(x)
            pdf.set_font(FONT_FAMILY, "B", 8.5)
            pdf.set_text_color(*BODY_GREY)
            pdf.cell(44, row_h, key + ":", fill=True)
            pdf.set_font(FONT_FAMILY, "", 8.5)
            pdf.set_text_color(*DARK_HEADER)
            pdf.cell(col_w - 44, row_h, val, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if len(features) % 2 != 0:
        pdf.ln(row_h)

    pdf.ln(8)

    for title, body in report.sections():
        pdf.set_fill_color(*SECTION_BG)
        pdf.set_draw_color(*BRAND_BLUE)
        pdf.set_line_width(0.4)
        pdf.rect(14, pdf.get_y(), 182, 9, style="DF")

        pdf.set_x(17)
        pdf.set_font(FONT_FAMILY, "B", 10)
        pdf.set_text_color(*BRAND_BLUE)
        pdf.cell(0, 9, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(2)
        pdf.set_x(14)
        pdf.set_font(FONT_FAMILY, "", 9.5)
        pdf.set_text_color(*BODY_GREY)
        clean_body = _strip_markdown(body) if body else "No information available."
        pdf.multi_cell(182, 5.5, clean_body, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

    buf = io.BytesIO()
    buf.write(pdf.output())
    return buf.getvalue()
