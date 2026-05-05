#!/usr/bin/env python3
"""Convert a Markdown file to PDF with academic styling."""
import sys
import markdown
from weasyprint import HTML

CSS = """
@page {
    size: letter;
    margin: 1in;
}
body {
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 12pt;
    line-height: 1.6;
    color: #000;
}
h1 {
    font-size: 18pt;
    text-align: center;
    margin-bottom: 0.5em;
    border-bottom: none;
}
h2 {
    font-size: 14pt;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}
h3 {
    font-size: 12pt;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
}
p {
    text-align: justify;
    margin-bottom: 0.8em;
}
ul, ol {
    margin-bottom: 0.8em;
}
li {
    margin-bottom: 0.3em;
}
code {
    font-family: monospace;
    font-size: 10pt;
    background: #f5f5f5;
    padding: 1px 4px;
}
"""

def convert(md_path, pdf_path):
    with open(md_path, "r") as f:
        md_text = f.read()
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body></html>"""
    HTML(string=full_html).write_pdf(pdf_path)
    print(f"Created: {pdf_path}")

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
