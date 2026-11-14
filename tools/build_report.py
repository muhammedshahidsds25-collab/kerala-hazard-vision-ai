"""Build report/report.pdf from report/report.md with headless Chrome or Edge.

Usage: python tools/build_report.py   (needs the `markdown` package and Chrome or Edge)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report" / "report.md"
HTML = ROOT / "report" / "report.html"
PDF = ROOT / "report" / "report.pdf"
BROWSERS = (
    "chrome",
    "msedge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)
STYLE = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; line-height: 1.4; color: #111; }
h1 { font-size: 16pt; margin: 0 0 4pt; }
h2 { font-size: 12.5pt; margin: 12pt 0 4pt; border-bottom: 1px solid #ccc; }
h3 { font-size: 11pt; margin: 9pt 0 3pt; }
p, li { margin: 3pt 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 5pt 0;
        page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2pt 4pt; text-align: left; }
th { background: #eee; }
img { display: block; width: 72%; margin: 6pt auto 2pt; page-break-inside: avoid; }
code { font-size: 8.5pt; }
a { color: #1a4fa3; }
"""


def find_browser() -> str:
    """Path of the first Chrome or Edge found."""
    for name in BROWSERS:
        path = shutil.which(name) or (name if Path(name).exists() else None)
        if path:
            return path
    raise FileNotFoundError("Chrome or Edge is needed to print the PDF")


def main() -> None:
    """Render Markdown to HTML, then print it to PDF."""
    body = markdown.markdown(SOURCE.read_text(encoding="utf-8"), extensions=["tables"])
    HTML.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><style>{STYLE}</style></head>"
        f"<body>{body}</body></html>",
        encoding="utf-8",
    )
    subprocess.run(
        [find_browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={PDF}", HTML.as_uri()],
        check=True, capture_output=True,
    )
    print(f"Wrote {PDF}")


if __name__ == "__main__":
    main()
