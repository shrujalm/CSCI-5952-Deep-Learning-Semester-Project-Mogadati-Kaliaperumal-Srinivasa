"""Compile docs/final_report.md into docs/final_report.pdf.

The project report is maintained as Markdown, while the submitted artifact is a
PDF. This script performs a small Markdown-to-LaTeX conversion that is tailored
to the report structure and then runs pdflatex.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
MARKDOWN_PATH = DOCS_DIR / "final_report.md"
TEX_NAME = "final_report_build.tex"
TEX_PATH = DOCS_DIR / TEX_NAME
BUILD_PDF = DOCS_DIR / "final_report_build.pdf"
FINAL_PDF = DOCS_DIR / "final_report.pdf"


LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    return "".join(LATEX_SPECIALS.get(char, char) for char in text)


def render_non_code_inline(text: str) -> str:
    """Render inline Markdown that is outside code spans."""

    parts: list[str] = []
    cursor = 0
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        parts.append(escape_latex(text[cursor : match.start()]))
        parts.append(r"\href{" + match.group(2) + "}{" + escape_latex(match.group(1)) + "}")
        cursor = match.end()
    parts.append(escape_latex(text[cursor:]))
    return "".join(parts)


def render_inline(text: str) -> str:
    """Render simple inline Markdown used by the report."""

    parts: list[str] = []
    cursor = 0
    for match in re.finditer(r"`([^`]+)`", text):
        parts.append(render_non_code_inline(text[cursor : match.start()]))
        parts.append(r"\texttt{" + escape_latex(match.group(1)) + "}")
        cursor = match.end()
    parts.append(render_non_code_inline(text[cursor:]))
    rendered = "".join(parts)
    rendered = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", rendered)
    return rendered


def is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def render_table(lines: list[str]) -> list[str]:
    rows: list[list[str]] = []
    for line in lines:
        if is_table_separator(line):
            continue
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])

    if not rows:
        return []

    column_count = max(len(row) for row in rows)
    column_spec = "|".join(["l"] * column_count)
    output = [
        r"\begin{center}",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{|{column_spec}|}}",
        r"\hline",
    ]
    for index, row in enumerate(rows):
        padded = row + [""] * (column_count - len(row))
        output.append(" & ".join(render_inline(cell) for cell in padded) + r" \\")
        output.append(r"\hline")
        if index == 0:
            output.append(r"\hline")
    output.extend([r"\end{tabular}%", r"}", r"\end{center}", r"\normalsize"])
    return output


def collect_paragraph(lines: list[str], start: int) -> tuple[str, int]:
    paragraph: list[str] = []
    index = start
    while index < len(lines) and lines[index].strip():
        line = lines[index]
        if line.startswith("#") or line.startswith("![") or line.startswith("|"):
            break
        paragraph.append(line.strip())
        index += 1
    return " ".join(paragraph), index


def render_figure(image_line: str, caption: str) -> list[str]:
    match = re.fullmatch(r"!\[[^\]]*\]\(([^)]+)\)", image_line.strip())
    if not match:
        return [render_inline(image_line)]

    image_path = match.group(1).replace("\\", "/")
    return [
        r"\begin{figure}[htbp]",
        r"\centering",
        rf"\includegraphics[width=0.93\linewidth,height=0.62\textheight,keepaspectratio]{{{image_path}}}",
        r"\par\vspace{0.35em}",
        r"{\small " + render_inline(caption) + r"\par}",
        r"\end{figure}",
    ]


def markdown_to_latex(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.85in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{array}",
        r"\usepackage{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.65em}",
        r"\emergencystretch=3em",
        r"\sloppy",
        r"\begin{document}",
    ]

    index = 0
    in_title_block = False
    title_line_seen = False
    title_center_open = False

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped == "# Title":
            in_title_block = True
            index += 1
            continue

        if in_title_block and stripped.startswith("# "):
            if title_center_open:
                output.append(r"\end{center}")
                title_center_open = False
            in_title_block = False

        if in_title_block:
            if not title_line_seen:
                output.extend(
                    [
                        r"\begin{center}",
                        r"{\LARGE\bfseries " + render_inline(stripped) + r"\par}",
                    ]
                )
                title_line_seen = True
                title_center_open = True
            elif stripped.startswith("*Keywords*:"):
                if title_center_open:
                    output.append(r"\end{center}")
                    title_center_open = False
                output.append(render_inline(stripped))
            else:
                output.append(render_inline(stripped) + r"\par")
            index += 1
            continue

        if stripped.startswith("# "):
            output.append(r"\section*{" + render_inline(stripped[2:]) + "}")
            index += 1
            continue

        if stripped.startswith("## "):
            output.append(r"\subsection*{" + render_inline(stripped[3:]) + "}")
            index += 1
            continue

        if stripped.startswith("!["):
            caption = ""
            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            if next_index < len(lines) and lines[next_index].strip().startswith("Figure"):
                caption, next_index = collect_paragraph(lines, next_index)
            output.extend(render_figure(stripped, caption))
            index = next_index
            continue

        if stripped.startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            output.extend(render_table(table_lines))
            continue

        paragraph, index = collect_paragraph(lines, index)
        if paragraph:
            output.append(render_inline(paragraph))

    if in_title_block and title_center_open:
        output.append(r"\end{center}")

    output.append(r"\end{document}")
    return "\n".join(output) + "\n"


def compile_pdf() -> None:
    if not MARKDOWN_PATH.exists():
        raise FileNotFoundError(f"Missing report source: {MARKDOWN_PATH}")

    TEX_PATH.write_text(markdown_to_latex(MARKDOWN_PATH.read_text(encoding="utf-8")), encoding="utf-8")

    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError("pdflatex was not found on PATH")

    command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", TEX_NAME]
    for _ in range(2):
        subprocess.run(command, cwd=DOCS_DIR, check=True)

    BUILD_PDF.replace(FINAL_PDF)

    for suffix in [".aux", ".log", ".out", ".toc", ".tex"]:
        path = DOCS_DIR / f"final_report_build{suffix}"
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    compile_pdf()
    print(f"Updated {FINAL_PDF.relative_to(ROOT)}")
