"""Build report/report.docx from report/report.md (same content as the PDF).

Handles the Markdown the report uses: headings, paragraphs, bold/italic/code, links, bullet and
numbered lists, pipe tables and images. Usage: python tools/build_report_docx.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report" / "report.md"
OUT = ROOT / "report" / "report.docx"
INLINE = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`|\[.+?\]\(.+?\))")
IMAGE = re.compile(r"!\[(.*?)\]\((.+?)\)")


def add_inline(paragraph, text: str) -> None:
    """Add text with **bold**, *italic*, `code` and [links](url) as formatted runs."""
    for part in INLINE.split(text.replace("<br>", "")):
        if not part:
            continue
        if part.startswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`"):
            paragraph.add_run(part[1:-1]).font.name = "Consolas"
        elif part.startswith("["):
            label, url = re.match(r"\[(.+?)\]\((.+?)\)", part).groups()
            paragraph.add_run(label if label == url or "github" in label else f"{label} ({url})")
        elif part.startswith("*"):
            paragraph.add_run(part[1:-1]).italic = True
        else:
            paragraph.add_run(part)


def add_table(doc: Document, rows: list[str]) -> None:
    """Add a pipe table (header row, separator row, body rows)."""
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    cells = [cells[0], *cells[2:]]  # drop the --- separator row
    table = doc.add_table(rows=len(cells), cols=len(cells[0]))
    table.style = "Table Grid"
    for r, row in enumerate(cells):
        for c, text in enumerate(row):
            paragraph = table.cell(r, c).paragraphs[0]
            add_inline(paragraph, text)
            for run in paragraph.runs:
                run.font.size = Pt(8.5)
                run.bold = run.bold or r == 0


def main() -> None:
    """Convert the Markdown report line by line."""
    doc = Document()
    for section in doc.sections:
        section.left_margin = section.right_margin = Cm(2)
        section.top_margin = section.bottom_margin = Cm(1.8)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, block)
            continue
        if image := IMAGE.fullmatch(line.strip()):
            caption, path = image.groups()
            doc.add_picture(str((SOURCE.parent / path).resolve()), width=Cm(13))
            doc.add_paragraph(caption).runs[0].italic = True
        elif heading := re.match(r"(#{1,3}) (.+)", line):
            doc.add_heading(heading.group(2), level=len(heading.group(1)) - 1)
        elif item := re.match(r"(?:- |\d+\. )(.+)", line):
            style = "List Bullet" if line.startswith("- ") else "List Number"
            add_inline(doc.add_paragraph(style=style), item.group(1))
        elif line.strip():
            add_inline(doc.add_paragraph(), line)
        i += 1
    doc.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
