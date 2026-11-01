#!/usr/bin/env python3
"""Build ``paper/main.md`` -> ``paper/main.html`` -> ``paper/main.pdf``.

The pipeline is deliberately dependency-light: Python-Markdown converts the
manuscript, KaTeX (loaded from a CDN) typesets the mathematics in the browser,
and Microsoft Edge in headless mode prints the result to PDF.  No LaTeX or
pandoc installation is required.

Manuscript features understood by this builder
----------------------------------------------
Placeholders (all resolved before rendering; unknown ones survive as visible
``[[...]]`` slots so a draft build is always possible)::

    [[FIG:name]]            numbered figure float, image file paper/figures/<name>.{png,jpg,jpeg,svg}
    [[TABLE:name]]          numbered table float, data in paper/tables/<name>.{md,csv,tex}
    [[REF:fig:name]]        "Figure 3"
    [[REF:tab:name]]        "Table 2"        ([[REF:table:name]] is accepted too)
    [[REF:sec:id]]          "4.2"            (id given by a heading attribute {#sec:id})
    [[REF:eq:id]]           "(7)"
    [[@key]]                "[4]"  bibliography key, see paper/bibliography.md
    [[CITE:key1,key2]]      "[4, 9]"
    [[RESULTS:B2]]          unresolvable by design: rendered as a marked slot

Numbered display mathematics is written as::

    $$
    z^{k+1} - z^k + g = 0
    $$ {#eq:characteristic}

Headings are numbered automatically (``## Title {#sec:id}``); a heading carrying
``{.appendix}`` is lettered A, B, ... and one carrying ``{.unnumbered}`` (or the
``no-number`` class) is left unnumbered.

Usage
-----
    .venv/bin/python scripts/build_paper.py                 # html + pdf
    .venv/bin/python scripts/build_paper.py --no-pdf        # html only (fast)
    .venv/bin/python scripts/build_paper.py --strict        # fail on any missing slot
    .venv/bin/python scripts/build_paper.py --stats-only    # just report counts
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FIG_DIR = PAPER / "figures"
TAB_DIR = PAPER / "tables"
BUILD = PAPER / ".build"
MAIN_MD = PAPER / "main.md"
MAIN_HTML = PAPER / "main.html"
MAIN_PDF = PAPER / "main.pdf"
BIB_MD = PAPER / "bibliography.md"
REPORT = PAPER / "build_report.json"

EDGE_CANDIDATES = [
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Microsoft Edge Beta.app/Contents/MacOS/Microsoft Edge",
    "microsoft-edge",
    "msedge",
]

FIG_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif")

#: Opaque sentinel tokens.  They must be made of plain alphanumerics: any
#: markdown-significant character (or an HTML comment, which Python-Markdown
#: lifts into a raw-HTML *block*) would split the surrounding paragraph.
TOKEN_TMPL = "PLB{}ENDPLB"
TOKEN_RE = re.compile(r"PLB(\d+)ENDPLB")


class Tokens:
    """Registry of deferred renderings, keyed by a plain-text placeholder."""

    def __init__(self) -> None:
        self.payloads: list[str] = []

    def add(self, payload: str) -> str:
        self.payloads.append(payload)
        return TOKEN_TMPL.format(len(self.payloads) - 1)

    def get(self, idx: int) -> str:
        return self.payloads[idx]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def warn(msg: str) -> None:
    print(f"[build_paper] WARNING {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[build_paper] {msg}", file=sys.stderr)


def slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text or "x"


def strip_tex(s: str) -> str:
    """Remove TeX markup so that word counts and slugs are meaningful."""
    s = re.sub(r"\$[^$]*\$", " math ", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"[{}\\^_]", " ", s)
    return s


# --------------------------------------------------------------------------- #
# Front matter and bibliography
# --------------------------------------------------------------------------- #


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Parse a leading ``---`` delimited ``key: value`` block."""
    m = re.match(r"\A\s*---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    key = None
    for raw in m.group(1).splitlines():
        if not raw.strip():
            continue
        if raw.lstrip().startswith("#"):
            continue
        mm = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", raw)
        if mm:
            key = mm.group(1).lower()
            val = mm.group(2).strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            meta[key] = val
        elif key is not None:  # continuation line
            meta[key] = (meta[key] + " " + raw.strip()).strip()
    return meta, text[m.end():]


BIB_ENTRY_RE = re.compile(r"^###\s+@?([A-Za-z0-9_:\-]+)\s*$", re.M)


def parse_bibliography(path: Path) -> dict[str, dict[str, str]]:
    """Parse ``paper/bibliography.md``.

    Each entry starts with a level-3 heading ``### key`` and its body is the
    formatted reference, optionally carrying a status marker in a trailing
    line such as ``[UNVERIFIED]``.
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    text = split_front_matter(text)[1]
    hits = list(BIB_ENTRY_RE.finditer(text))
    entries: dict[str, dict[str, str]] = {}
    for i, m in enumerate(hits):
        body = text[m.end(): hits[i + 1].start() if i + 1 < len(hits) else len(text)]
        body = body.strip()
        # drop markdown rules / comments between entries
        body = re.sub(r"\n+-{3,}\s*$", "", body).strip()
        body_clean = re.sub(r"\s*\[UNVERIFIED\]", "", body).strip()
        entries[m.group(1)] = {
            "key": m.group(1),
            "html": render_inline(body_clean, bib_mode=True),
            "raw": body,
            "unverified": "[UNVERIFIED]" in body,
        }
    return entries


# --------------------------------------------------------------------------- #
# Inline markdown rendering for bibliography bodies
# --------------------------------------------------------------------------- #


def render_inline(text: str, *, bib_mode: bool = False) -> str:
    """Render a small subset of markdown inline syntax to HTML."""
    out = text
    out = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{html.escape(m.group(1))}</a>',
        out,
    )
    out = re.sub(r"(?<!\*)\*\*([^*]+)\*\*(?!\*)", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = out.replace("\u00a0", "&nbsp;")
    if bib_mode:
        out = out.replace("\n", " ")
        out = re.sub(r"\s{2,}", " ", out)
    return out


# --------------------------------------------------------------------------- #
# Placeholder machinery
# --------------------------------------------------------------------------- #


@dataclass
class Slot:
    kind: str          # FIG, TABLE, REF, RESULTS, ...
    name: str
    line: int
    resolved: bool = False
    detail: str = ""


@dataclass
class Counters:
    fig: int = 0
    tab: int = 0
    eq: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"figures": self.fig, "tables": self.tab, "equations": self.eq}


@dataclass
class Figure:
    name: str
    number: int
    path: Path | None
    caption: str


@dataclass
class Table:
    name: str
    number: int
    path: Path | None
    caption: str


@dataclass
class BuildState:
    slots: list[Slot] = field(default_factory=list)
    figures: dict[str, Figure] = field(default_factory=dict)
    tables: dict[str, Table] = field(default_factory=dict)
    equations: dict[str, tuple[int, str]] = field(default_factory=dict)
    eq_tex: dict[int, str] = field(default_factory=dict)
    section_ids: dict[str, str] = field(default_factory=dict)
    section_text: dict[str, str] = field(default_factory=dict)
    sections: dict[str, tuple[str, str]] = field(default_factory=dict)
    cites: dict[str, int] = field(default_factory=dict)     # key -> number
    cite_order: list[str] = field(default_factory=list)
    missing_figures: list[str] = field(default_factory=list)
    missing_captions: list[str] = field(default_factory=list)
    missing_tables: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    labels_seen: dict[str, str] = field(default_factory=dict)
    toks: "Tokens" = field(default_factory=lambda: Tokens())


CAPTION_RE = re.compile(r"^\s*(?:<!--\s*)?cap(?:tion)?\s*:\s*(.+)$", re.I | re.M)


def figure_caption(name: str) -> tuple[str, bool]:
    """Caption for a figure: *``paper/figures/<name>.cap``* if present."""
    for ext in (".cap", ".caption", ".txt"):
        p = FIG_DIR / f"{name}{ext}"
        if p.exists():
            return p.read_text(encoding="utf-8").strip(), True
    return f"caption for figure {name} not yet written", False


def find_figure(name: str) -> Path | None:
    for ext in FIG_SUFFIXES:
        p = FIG_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None


def find_table(name: str) -> Path | None:
    for ext in (".md", ".csv", ".tex"):
        p = TAB_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None


def table_html(path: Path) -> str:
    """Render a table source file to HTML (markdown passed through later)."""
    if path.suffix == ".csv":
        rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
        rows = [r for r in rows if any(c.strip() for c in r)]
        if not rows:
            return "<p><em>empty table</em></p>"
        head, *body = rows
        out = ["<table>", "<thead><tr>"]
        out += [f"<th>{render_inline(c.strip())}</th>" for c in head]
        out += ["</tr></thead>", "<tbody>"]
        for r in body:
            out.append("<tr>" + "".join(f"<td>{render_inline(c.strip())}</td>" for c in r) + "</tr>")
        out += ["</tbody>", "</table>"]
        return "".join(out)
    if path.suffix == ".md":
        return make_markdown().convert(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Stage 1: extract mathematics, floats, and refs from the markdown source
# --------------------------------------------------------------------------- #

EQ_BLOCK_RE = re.compile(
    r"(?m)^[ \t]*\$\$[ \t]*\n(?P<tex>.*?)\n[ \t]*\$\$[ \t]*(?:\{\#(?P<label>[\w:\-]+)\})?[ \t]*$",
    re.S,
)
EQ_INLINE_RE = re.compile(
    r"(?m)^[ \t]*\$\$[ \t]*(?P<tex>[^\n$]+?)[ \t]*\$\$[ \t]*(?:\{\#(?P<label>[\w:\-]+)\})?[ \t]*$"
)


def extract_equations(src: str, st: BuildState) -> str:
    """Replace numbered display math by sentinels, in document order."""

    def repl(m: re.Match) -> str:
        tex = m.group("tex").strip()
        label = m.group("label")
        st.eq_counter += 1  # type: ignore[attr-defined]
        n = st.eq_counter  # type: ignore[attr-defined]
        st.eq_tex[n] = tex
        if label:
            label = re.sub(r"^eq:", "", label)
            if label in st.equations:
                warn(f"duplicate equation label {label!r}")
            st.equations[label] = (n, tex)
        return "\n\n" + st.toks.add(f"EQ:{n}:{label or ''}") + "\n\n"

    src = EQ_BLOCK_RE.sub(repl, src)
    if EQ_INLINE_RE.search(src):
        src = EQ_INLINE_RE.sub(repl, src)
    return src


FLOAT_RE = re.compile(r"\[\[(FIG|FIGURE|TABLE|TAB):([A-Za-z0-9_:\-]+)\]\]")


DIV_OPEN_RE = re.compile(r"(?m)^[ \t]*:::[ \t]*\{(?P<attrs>[^}]*)\}[ \t]*$")
DIV_CLOSE_RE = re.compile(r"(?m)^[ \t]*:::[ \t]*$")
THEOREM_KINDS = ("prop", "cor", "thm", "lem", "def")


def extract_fenced_divs(src: str, st: BuildState) -> str:
    """Turn ``::: {.prop #prop:x}`` fenced divs into ``<div markdown="1">``.

    Theorem-like blocks (class ``prop``, ``cor``, ...) are also scanned for the
    bold label at their head, so that ``[[REF:prop:delay]]`` can print
    "Proposition 1" without the number being maintained in two places.
    """
    for m in re.finditer(
        r"(?ms)^[ \t]*:::[ \t]*\{(?P<attrs>[^}]*)\}[ \t]*$(?P<body>.*?)^[ \t]*:::[ \t]*$", src
    ):
        attrs = m.group("attrs")
        idm = re.search(r"#([\w:\-]+)", attrs)
        if not idm:
            continue
        head = re.search(r"\*\*([A-Z][A-Za-z]+\s+(?:\d+|[IVX]+|[A-Z]))\b[^*]*\*\*", m.group("body"))
        if head:
            st.labels_seen[idm.group(1)] = re.sub(r"\s+", " ", head.group(1)).strip()

    def open_repl(m: re.Match) -> str:
        attrs = m.group("attrs")
        classes = " ".join(re.findall(r"\.([\w\-]+)", attrs))
        ident = re.search(r"#([\w:\-]+)", attrs)
        out = f'<div class="{classes}"' if classes else "<div"
        if ident:
            out += f' id="{ident.group(1)}"'
        return "\n\n" + out + ' markdown="1">\n'

    src = DIV_OPEN_RE.sub(open_repl, src)
    src = DIV_CLOSE_RE.sub(lambda _m: "\n</div>\n\n", src)
    return src


def extract_floats(src: str, st: BuildState) -> str:
    """Number figures and tables in order of appearance and hide their bodies."""

    def repl(m: re.Match) -> str:
        kind = m.group(1).upper()
        name = m.group(2)
        if kind.startswith("FIG"):
            if name in st.figures:
                return st.toks.add(f"FIG:{name}")
            st.fig_counter += 1  # type: ignore[attr-defined]
            path = find_figure(name)
            cap, ok = figure_caption(name)
            cap = extract_refs_and_cites(cap, st) if ok else cap
            if path is None:
                st.missing_figures.append(name)
                warn(f"figure {name!r} has no image in paper/figures/")
            if not ok:
                st.missing_captions.append(name)
            st.figures[name] = Figure(name, st.fig_counter, path, cap)  # type: ignore[attr-defined]
            return st.toks.add(f"FIG:{name}")
        if name in st.tables:
            return st.toks.add(f"TAB:{name}")
        st.tab_counter += 1  # type: ignore[attr-defined]
        path = find_table(name)
        cap_file = TAB_DIR / f"{name}.cap"
        cap = extract_refs_and_cites(cap_file.read_text(encoding="utf-8").strip(), st) if cap_file.exists() else (
            f"caption for table {name} not yet written"
        )
        if path is None:
            st.missing_tables.append(name)
            warn(f"table {name!r} has no data file in paper/tables/")
        st.tables[name] = Table(name, st.tab_counter, path, cap)  # type: ignore[attr-defined]
        return st.toks.add(f"TAB:{name}")

    return FLOAT_RE.sub(repl, src)


REF_RE = re.compile(
    r"\[\[REF:(fig|tab|table|sec|eq|prop|cor|thm|lem|def):([A-Za-z0-9_:\-]+)\]\]"
)
CITE_RE = re.compile(r"\[\[@([A-Za-z0-9_:\-]+)\]\]")
CITE_LIST_RE = re.compile(r"\[\[CITE:([A-Za-z0-9_:\-,\s]+)\]\]")


def extract_refs_and_cites(src: str, st: BuildState) -> str:
    # the reference itself prints "Section", "Appendix", ... so a preceding noun
    # in the prose would be duplicated ("Section Section 4")
    src = re.sub(
        r"\b(?:Sections?|Appendices|Appendix|Figures?|Tables?|Propositions?|Corollary|"
        r"Corollaries|Lemmas?|Theorems?)\s+"
        r"(?=\[\[REF:(?:sec|app|fig|tab|table|prop|cor|lem|thm):)", "", src)

    def do_ref(m: re.Match) -> str:
        kind, label = m.group(1).lower(), m.group(2)
        # tolerate both [[REF:sec:theory]] and [[REF:sec:sec:theory]]
        label = re.sub(
            rf"^(?:{'|'.join(('fig', 'tab', 'table', 'sec', 'eq', 'prop', 'cor', 'thm', 'lem', 'def'))}):",
            "", label,
        )
        return st.toks.add(f"REF:{kind}:{label}")

    def do_cite(m: re.Match) -> str:
        key = m.group(1)
        return st.toks.add(f"CITE:{key}")

    def do_cite_list(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        return st.toks.add("CITE:" + ",".join(keys))

    src = REF_RE.sub(do_ref, src)
    src = CITE_LIST_RE.sub(do_cite_list, src)
    src = CITE_RE.sub(do_cite, src)
    return src


GENERIC_SLOT_RE = re.compile(r"\[\[([A-Za-z][A-Za-z0-9_]*:[^\[\]]+?)\]\]")


def extract_page_breaks(src: str, st: BuildState) -> str:
    """``\\newpage`` on a line of its own starts a new printed page."""
    return re.sub(r"(?m)^[ \t]*\\newpage[ \t]*$", lambda _m: "\n\n" + st.toks.add("PB") + "\n\n", src)


def collect_generic_slots(src: str, st: BuildState) -> str:
    """Anything still of the form ``[[KIND:name]]`` is a declared slot."""

    def repl(m: re.Match) -> str:
        slot = m.group(1)
        st.unresolved.append(f"[[{slot}]]")
        return st.toks.add("SLOT:" + slot)

    return GENERIC_SLOT_RE.sub(repl, src)


# --------------------------------------------------------------------------- #
# Stage 2: markdown -> html, heading numbering, sentinel resolution
# --------------------------------------------------------------------------- #

HEAD_RE = re.compile(r"<h([1-6])([^>]*)>(.*?)</h\1>", re.S | re.I)
ATTR_ID_RE = re.compile(r'id="([^"]*)"')
ATTR_CLASS_RE = re.compile(r'class="([^"]*)"')


def make_markdown():
    import markdown  # imported late so --help works without deps

    return markdown.Markdown(
        extensions=[
            "extra",
            "sane_lists",
            "attr_list",
            "tables",
            "fenced_code",
            "footnotes",
            "def_list",
            "md_in_html",
        ],
        extension_configs={"footnotes": {"UNIQUE_IDS": True}},
        output_format="html5",
    )


def _record_section(st: BuildState, anchor: str, label: str, what: str,
                    text: str = "") -> None:
    st.sections[anchor] = (label, what)
    st.section_ids[anchor] = anchor
    if text:
        st.section_text[anchor] = text
    for pre in ("sec", "app", "fig", "tab", "table", "eq"):
        if anchor.startswith(pre + ":"):
            short = anchor[len(pre) + 1:]
            st.sections[short] = (label, what)
            st.section_ids[short] = anchor
            if text:
                st.section_text[short] = text
            break


def number_headings(body: str, st: BuildState) -> str:
    """Number h2..h5 in place and record ``id -> label`` in ``st.sections``.

    Body sections are numbered 1, 2, ...; a heading carrying ``class="appendix"``
    starts (or continues) a lettered appendix series A, B, ...; sub-headings take
    the enclosing label as prefix (``4.2``, ``A.1``).
    """
    state = {"num": 0, "app": 0, "top": None, "cur": {}}

    def repl(m: re.Match) -> str:
        level = int(m.group(1))
        attrs, inner = m.group(2), m.group(3)
        if level < 2:                       # the paper title: never numbered
            return m.group(0)
        idm = ATTR_ID_RE.search(attrs)
        cm = ATTR_CLASS_RE.search(attrs)
        classes = cm.group(1).split() if cm else []
        anchor = idm.group(1) if idm else slug(re.sub(r"<[^>]+>", "", inner))
        plain = re.sub(r"<[^>]+>", "", inner).strip()
        if "unnumbered" in classes or "no-number" in classes:
            _record_section(st, anchor, plain, "unnumbered", plain)
            return m.group(0)

        if level == 2:
            state["cur"].clear()
            if "appendix" in classes:
                state["app"] += 1
                label = chr(ord("A") + state["app"] - 1)
                state["top"] = label
                state["cur"][2] = label
                _record_section(st, anchor, f"Appendix {label}", "appendix", plain)
                return (f'<h2{attrs}><span class="secno">Appendix {label}</span> '
                        f'{inner}</h2>')
            state["num"] += 1
            label = str(state["num"])
            state["top"] = label
            state["cur"][2] = label
            _record_section(st, anchor, label, "section", plain)
            return f'<h2{attrs}><span class="secno">{label}</span> {inner}</h2>'

        parent = None
        for lv in range(level - 1, 1, -1):
            if lv in state["cur"]:
                parent = state["cur"][lv]
                break
        if parent is None:
            if state["top"] is None:
                return m.group(0)
            parent = state["top"]
        state["cur"][level] = state["cur"].get(level, 0) + 1
        for lv in [k for k in state["cur"] if k > level]:
            del state["cur"][lv]
        label = f"{parent}.{state['cur'][level]}"
        _record_section(st, anchor, label, "section", plain)
        tag = f"h{level}"
        return f'<{tag}{attrs}><span class="secno">{label}</span> {inner}</{tag}>'

    return HEAD_RE.sub(repl, body)


BLOCK_KINDS = {"EQ", "FIG", "TAB", "PB"}


def render_token(payload: str, st: BuildState, entries: dict[str, dict[str, str]]) -> tuple[str, bool]:
    """Render one deferred token; returns (html, is_block)."""
    parts = payload.split(":", 2)
    head = parts[0]

    if head == "PB":
        return '<div class="pagebreak"></div>', True

    if head == "EQ":
        n = int(parts[1])
        label = parts[2] if len(parts) > 2 else ""
        tex = st.equations.get(label, (n, ""))[1] if label else st.eq_tex.get(n, "")  # type: ignore[attr-defined]
        return _equation_html(tex, n, label), True

    if head == "FIG":
        f = st.figures.get(parts[1])
        name = parts[1]
        if f is None:
            return f'<div class="float missing">[[FIG:{name}]]</div>', True
        if f.path is None:
            return (
                f'<figure class="float missing" id="fig-{f.name}">'
                f'<div class="figph">figure <code>{html.escape(f.name)}</code> pending</div>'
                f'<figcaption><span class="floatlabel">Figure {f.number}.</span> '
                f'{render_inline(f.caption)}</figcaption></figure>'
            ), True
        try:
            src = f.path.relative_to(PAPER).as_posix()
        except ValueError:
            src = f.path.as_uri()
        return (
            f'<figure class="float" id="fig-{f.name}">'
            f'<img src="{html.escape(src, quote=True)}" alt="{html.escape(f.name, quote=True)}">'
            f'<figcaption><span class="floatlabel">Figure {f.number}.</span> '
            f'{render_inline(f.caption)}</figcaption></figure>'
        ), True

    if head == "TAB":
        t = st.tables.get(parts[1])
        name = parts[1]
        if t is None:
            return f'<div class="float missing">[[TABLE:{name}]]</div>', True
        inner = table_html(t.path) if t.path else (
            f'<div class="figph">table <code>{html.escape(t.name)}</code> pending</div>'
        )
        return (
            f'<div class="tablefloat" id="tab-{t.name}">'
            f'<figcaption class="tabcap"><span class="floatlabel">Table {t.number}.</span> '
            f'{render_inline(t.caption)}</figcaption>'
            f'<div class="tabbody">{inner}</div></div>'
        ), True

    if head == "REF" and len(parts) >= 3:
        return fmt_ref(parts[1], parts[2], st)

    if head == "CITE":
        keys = [k for k in parts[1].split(",") if k]
        nums = [cite_number(k, st, entries) for k in keys]
        nums = [n for n in nums if n is not None]
        if not nums:
            return '<span class="xref-broken">[?]</span>', False
        return (
            '<a class="cite" href="#ref-' + keys[0] + '">[' + ", ".join(str(n) for n in nums) + "]</a>"
        ), False

    if head == "SLOT":
        return f'<span class="slot">[[{html.escape(":".join(parts[1:]))}]]</span>', False

    return TOKEN_TMPL.format(-1), False


def fmt_ref(kind: str, label: str, st: BuildState) -> tuple[str, bool]:
    if kind == "sec":
        if label in st.sections:
            num, what = st.sections[label]
            dom = st.section_ids.get(label, label)
            if what == "appendix":
                return f'<a class="xref" href="#{dom}">{num}</a>', False
            if what == "unnumbered":
                title = st.section_text.get(label, num)
                return f'<a class="xref" href="#{dom}">{title}</a>', False
            return f'<a class="xref" href="#{dom}">Section&nbsp;{num}</a>', False
        st.unresolved.append(f"[[REF:sec:{label}]]")
        return '<span class="xref-broken">Section&nbsp;??</span>', False
    if kind == "fig":
        f = st.figures.get(label)
        if f:
            return f'<a class="xref" href="#fig-{label}">Figure&nbsp;{f.number}</a>', False
        st.unresolved.append(f"[[REF:fig:{label}]]")
        return '<span class="xref-broken">Figure&nbsp;??</span>', False
    if kind in ("tab", "table"):
        t = st.tables.get(label)
        if t:
            return f'<a class="xref" href="#tab-{label}">Table&nbsp;{t.number}</a>', False
        st.unresolved.append(f"[[REF:tab:{label}]]")
        return '<span class="xref-broken">Table&nbsp;??</span>', False
    if kind in THEOREM_KINDS:
        ident = f"{kind}:{label}" if label not in st.labels_seen else label
        text = st.labels_seen.get(ident) or st.labels_seen.get(label)
        if text:
            anchor = ident if ident in st.labels_seen else label
            return f'<a class="xref" href="#{anchor}">{text}</a>', False
        st.unresolved.append(f"[[REF:{kind}:{label}]]")
        return f'<span class="xref-broken">{kind.title()}&nbsp;??</span>', False

    if kind == "eq":
        if label in st.equations:
            n = st.equations[label][0]
            return f'<a class="xref" href="#eq-{label}">({n})</a>', False
        st.unresolved.append(f"[[REF:eq:{label}]]")
        return '<span class="xref-broken">(??)</span>', False
    st.unresolved.append(f"[[REF:{kind}:{label}]]")
    return f'<span class="xref-broken">[[REF:{kind}:{label}]]</span>', False


def cite_number(key: str, st: BuildState, entries: dict[str, dict[str, str]]) -> int | None:
    if key in st.cites:
        return st.cites[key]
    if key in entries:
        st.cite_order.append(key)
        n = len(st.cite_order)
        st.cites[key] = n
        return n
    warn(f"citation key {key!r} not in paper/bibliography.md")
    st.unresolved.append(f"[[@{key}]]")
    return None


def resolve_sentinels(body: str, st: BuildState, entries: dict[str, dict[str, str]]) -> str:
    """Substitute tokens until none remain.

    Iteration is necessary because a figure or table caption may itself contain a
    cross-reference: resolving the float emits the caption, which contains new
    tokens, which must then be resolved in turn.
    """
    for _ in range(6):
        before = body
        body = _resolve_once(body, st, entries)
        if body == before:
            break
    return _split_blocks_out_of_paragraphs(body)


def _resolve_once(body: str, st: BuildState, entries: dict[str, dict[str, str]]) -> str:
    """One pass: block tokens that own a paragraph replace it entirely."""

    # 1. block tokens that occupy a whole paragraph -> replace the paragraph
    def block_repl(m: re.Match) -> str:
        payload = st.toks.get(int(m.group(1)))
        out, is_block = render_token(payload, st, entries)
        return out if is_block else m.group(0)

    body = re.sub(r"<p>\s*PLB(\d+)ENDPLB\s*</p>", block_repl, body)
    body = re.sub(r"(?m)^[ \t]*PLB(\d+)ENDPLB[ \t]*$", block_repl, body)

    # 2. everything left is inline; block tokens are marked so that step 3 can
    #    lift them out of the surrounding paragraph (a <div> may not nest in <p>)
    def inline_repl(m: re.Match) -> str:
        payload = st.toks.get(int(m.group(1)))
        out, is_block = render_token(payload, st, entries)
        return f"\x00BLK{out}\x00END" if is_block else out

    return TOKEN_RE.sub(inline_repl, body)


BLK_RE = re.compile(r"\x00BLK(.*?)\x00END", re.S)


def _split_blocks_out_of_paragraphs(body: str) -> str:
    """Turn ``<p>prose <div/> more</p>`` into ``<p>prose</p><div/><p>more</p>``."""
    if "\x00BLK" not in body:
        return body

    def fix(m: re.Match) -> str:
        inner = m.group(1)
        if "\x00BLK" not in inner:
            return f"<p>{inner}</p>" if inner.strip() else ""
        pieces = BLK_RE.split(inner)
        out = []
        for i, piece in enumerate(pieces):
            if i % 2 == 1:                       # the block element itself
                out.append(piece)
            elif piece.strip():
                out.append(f"<p>{piece.strip()}</p>")
        return "\n".join(out)

    body = re.sub(r"<p>(.*?)</p>", fix, body, flags=re.S)
    return body.replace("\x00BLK", "").replace("\x00END", "")


def _equation_html(tex: str, number: int, label: str) -> str:
    anchor = f' id="eq-{label}"' if label else ""
    return (
        f'<div class="equation"{anchor}>'
        f'<div class="eqbody">$$\n{tex}\n$$</div>'
        f'<div class="eqno">({number})</div>'
        f"</div>"
    )


# --------------------------------------------------------------------------- #
# Stage 3: assemble the standalone HTML document
# --------------------------------------------------------------------------- #

CSS = r"""
:root {
  --ink: #14171c; --muted: #5b6470; --rule: #d6dae0; --accent: #7a1f2b;
  --bg: #ffffff; --slotbg: #fdf3d8; --slotbd: #b98a12;
  --mono: "SF Mono", "Menlo", "Consolas", monospace;
  --serif: "Charter", "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --width: 46rem;
}
* { box-sizing: border-box; }
html { font-size: 11pt; -webkit-text-size-adjust: 100%; }
body {
  margin: 0 auto; padding: 2.6rem 1.6rem 4rem; max-width: calc(var(--width) + 3.2rem);
  background: var(--bg); color: var(--ink); font-family: var(--serif);
  line-height: 1.52; text-rendering: optimizeLegibility;
  font-kerning: normal; font-variant-ligatures: common-ligatures;
}
p { margin: 0 0 0.62rem; text-align: justify; hyphens: auto; -webkit-hyphens: auto;
    orphans: 3; widows: 3; }
h1, h2, h3, h4, h5 { font-family: var(--serif); font-weight: 700; line-height: 1.22;
    break-after: avoid-page; page-break-after: avoid; }
h1 { font-size: 1.62rem; margin: 0 0 .35rem; letter-spacing: -.01em; text-align: left; }
h2 { font-size: 1.16rem; margin: 1.9rem 0 .55rem; padding-bottom: .18rem;
     border-bottom: 1px solid var(--rule); }
h3 { font-size: 1.02rem; margin: 1.35rem 0 .4rem; }
h4 { font-size: .96rem; margin: 1.1rem 0 .32rem; font-style: italic; }
h5 { font-size: .92rem; margin: .9rem 0 .25rem; }
.secno { color: var(--accent); font-variant-numeric: tabular-nums; margin-right: .28em; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
a.xref, a.cite { font-variant-numeric: tabular-nums; }
code, pre, .eqno, .floatlabel { font-family: var(--mono); }
code { font-size: .88em; background: #f2f4f7; padding: .06em .28em; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid var(--rule); border-radius: 4px;
      padding: .7rem .85rem; overflow-x: auto; font-size: .80rem; line-height: 1.45; }
pre code { background: none; padding: 0; }
blockquote { margin: .8rem 0 .8rem 1.1rem; padding-left: .9rem;
             border-left: 2px solid var(--rule); color: #33383f; }
blockquote p:last-child { margin-bottom: 0; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 1.6rem 0; }
.pagebreak { break-after: page; page-break-after: always; height: 0; margin: 0; }
ul, ol { margin: 0 0 .7rem 1.25rem; padding: 0; }
li { margin: .12rem 0; }
li > p { margin-bottom: .25rem; }

/* ---------- title block ---------- */
.titleblock { border-bottom: 2px solid var(--ink); padding-bottom: 1.05rem; margin-bottom: 1.5rem; }
.kicker { font-family: var(--mono); font-size: .70rem; letter-spacing: .13em;
          text-transform: uppercase; color: var(--muted); margin-bottom: .7rem; }
.subtitle { font-size: 1.02rem; font-style: italic; color: #2c3138; margin: .1rem 0 .9rem; }
.authors { font-size: .95rem; margin: .1rem 0 .18rem; }
.affil { font-size: .84rem; color: var(--muted); margin: 0 0 .5rem; }
.meta { font-family: var(--mono); font-size: .74rem; color: var(--muted); }
.abstract { background: #fafbfc; border: 1px solid var(--rule); border-left: 3px solid var(--accent);
            padding: .85rem 1.05rem .55rem; margin: 0 0 1.35rem; }
.abstract h2, .abstract .abshead { font-size: .82rem; text-transform: uppercase; letter-spacing: .10em;
            border: 0; margin: 0 0 .45rem; color: var(--muted); font-weight: 700; }
.abstract p { font-size: .94rem; }

/* ---------- floats ---------- */
figure.float, .tablefloat {
  margin: 1.15rem 0 1.3rem; padding: 0; break-inside: avoid; page-break-inside: avoid;
  text-align: center;
}
figure.float img { max-width: 100%; height: auto; display: block; margin: 0 auto .45rem; }
figcaption { font-size: .845rem; line-height: 1.40; text-align: left; color: #2b3037;
             margin: 0 auto; max-width: 100%; }
.floatlabel { font-weight: 700; color: var(--ink); }
figure.float.missing { border: 1px dashed var(--slotbd); background: var(--slotbg);
             padding: .8rem; border-radius: 4px; }
.figph { font-family: var(--mono); font-size: .76rem; color: #7a5c00; padding: 1.1rem .5rem;
         border-bottom: 1px dashed var(--slotbd); margin-bottom: .5rem; }
.tablefloat .tabbody { overflow-x: auto; }
table { border-collapse: collapse; margin: .35rem auto .45rem; font-size: .85rem;
        font-variant-numeric: tabular-nums; width: 100%; }
thead th { border-top: 1.6px solid var(--ink); border-bottom: .8px solid var(--ink);
           padding: .34rem .5rem; text-align: left; font-weight: 700; }
tbody td { padding: .27rem .5rem; border-bottom: .5px solid #eceff3; }
tbody tr:last-child td { border-bottom: 1.6px solid var(--ink); }
tbody tr:nth-child(even) { background: #fafbfc; }
.tabcap { caption-side: top; margin-bottom: .3rem; }
.tablefloat figcaption { text-align: left; }

/* ---------- maths ---------- */
.equation { display: flex; align-items: center; gap: .4rem; margin: .95rem 0;
            break-inside: avoid; page-break-inside: avoid; }
.equation .eqbody { flex: 1 1 auto; min-width: 0; overflow-x: auto; overflow-y: hidden;
                    text-align: center; }
.equation .eqno { flex: 0 0 auto; font-size: .84rem; color: var(--ink); }
.katex-display { margin: .15rem 0; }
.katex { font-size: 1.02em; }

/* ---------- boxes ---------- */
.box { border: 1px solid var(--rule); border-radius: 4px; padding: .75rem .95rem .45rem;
       margin: 1rem 0 1.15rem; background: #fbfcfd; break-inside: avoid; }
.box .boxhead { font-family: var(--mono); font-size: .72rem; letter-spacing: .1em;
       text-transform: uppercase; color: var(--muted); margin-bottom: .4rem; }
.prop { border-left: 3px solid var(--ink); padding-left: .95rem; margin: 1.1rem 0; }
.prop .prophead { font-weight: 700; }
.proof { margin: .4rem 0 .9rem; }
.proof .proofhead { font-style: italic; }
.qed { float: right; }
.proof::after { content: ""; display: block; clear: both; }

/* ---------- slots ---------- */
.slot { background: var(--slotbg); border: 1px dashed var(--slotbd); border-radius: 3px;
        padding: 0 .26em; font-family: var(--mono); font-size: .84em; color: #6b4f00;
        white-space: nowrap; }
.slot::before { content: "◻ "; color: var(--slotbd); }
.xref-broken { color: #b3261e; font-weight: 700; }

/* ---------- references ---------- */
.references { font-size: .90rem; }
.references p { text-align: left; padding-left: 2.1rem; text-indent: -2.1rem;
                margin-bottom: .34rem; }
.references p:target { background: #fff6d6; }
.unverified { font-family: var(--mono); font-size: .74em; color: #8a4b00;
              border: 1px solid #d9a441; border-radius: 2px; padding: 0 .22em; }

/* ---------- screen-only nav ---------- */
nav.toc { border: 1px solid var(--rule); border-radius: 4px; padding: .7rem .9rem;
          margin: 0 0 1.6rem; font-size: .84rem; background: #fafbfc; }
nav.toc .tochead { font-family: var(--mono); font-size: .70rem; letter-spacing: .11em;
          text-transform: uppercase; color: var(--muted); margin-bottom: .35rem; }
nav.toc ol { list-style: none; margin: 0; padding: 0; }
nav.toc li { margin: .08rem 0; }
nav.toc .tocnum { display: inline-block; min-width: 2.4em; color: var(--accent); }

@media print {
  :root { --width: 44rem; }
  html { font-size: 10.2pt; }
  @page { size: A4; margin: 20mm 19mm 18mm; }
  @page :first { margin-top: 16mm; }
  body { padding: 0; max-width: none; line-height: 1.42; }
  nav.toc { display: none; }
  a { color: inherit; text-decoration: none; }
  .slot { background: none; border-style: dotted; }
  .abstract { background: none; }
  h2 { break-after: avoid; page-break-after: avoid; }
  figure.float, .tablefloat, .equation, .box { break-inside: avoid; page-break-inside: avoid; }
  table { font-size: .76rem; }
  .references { font-size: .84rem; }
}
"""

MATH_SCRIPT = r"""
<script>
  window.__mathRendered = false;
  document.addEventListener("DOMContentLoaded", function () {
    if (typeof renderMathInElement !== "function") { return; }
    renderMathInElement(document.body, {
      delimiters: [
        {left: "$$", right: "$$", display: true},
        {left: "\\[", right: "\\]", display: true},
        {left: "$",  right: "$",  display: false},
        {left: "\\(", right: "\\)", display: false}
      ],
      throwOnError: false,
      strict: false,
      trust: false,
      macros: {"\\R": "\\mathbb{R}", "\\N": "\\mathbb{N}", "\\E": "\\mathbb{E}"},
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
    });
    window.__mathRendered = true;
    document.body.setAttribute("data-math", "done");
  });
</script>
"""

HEAD_TMPL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="planlab build_paper.py">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.12/dist/katex.min.css">
<style>{css}</style>
</head>
<body data-built="{stamp}">
"""

MATH_DEPS = """
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.12/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.12/dist/contrib/auto-render.min.js"></script>
"""


def build_titleblock(meta: dict[str, str]) -> str:
    bits = ['<header class="titleblock">']
    if meta.get("kicker"):
        bits.append(f'<div class="kicker">{render_inline(meta["kicker"])}</div>')
    title = meta.get("title", "Untitled")
    bits.append(f"<h1>{render_inline(title)}</h1>")
    if meta.get("subtitle"):
        bits.append(f'<div class="subtitle">{render_inline(meta["subtitle"])}</div>')
    if meta.get("authors"):
        bits.append(f'<div class="authors">{render_inline(meta["authors"])}</div>')
    if meta.get("affiliation"):
        bits.append(f'<div class="affil">{render_inline(meta["affiliation"])}</div>')
    line = []
    for k, lbl in (("version", "version"), ("date", "date"), ("status", "status")):
        if meta.get(k):
            line.append(f"{lbl}: {html.escape(meta[k])}")
    if line:
        bits.append(f'<div class="meta">{" &nbsp;·&nbsp; ".join(line)}</div>')
    bits.append("</header>")
    return "\n".join(bits)


def build_toc(st: BuildState) -> str:
    if not st.sections:
        return ""
    items = []
    for label, anchor in st.sections_order:  # type: ignore[attr-defined]
        num = st.sections.get(anchor, ("", "section"))[0]
        items.append(f'<li><a href="#{anchor}"><span class="tocnum">{num}</span>{label}</a></li>')
    return (
        '<nav class="toc"><div class="tochead">Contents</div><ol>' + "".join(items) + "</ol></nav>"
    )


def build_references(st: BuildState, entries: dict[str, dict[str, str]]) -> str:
    if not entries:
        return ""
    order = st.cite_order + [k for k in entries if k not in st.cite_order]
    out = ['<div class="references">']
    for i, key in enumerate(order, 1):
        e = entries[key]
        tag = ' <span class="unverified">[UNVERIFIED]</span>' if e["unverified"] else ""
        out.append(f'<p id="ref-{key}"><span class="refnum">[{i}]</span> {e["html"]}{tag}</p>')
    out.append("</div>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Stage 4: Edge print-to-pdf
# --------------------------------------------------------------------------- #


def find_edge() -> str | None:
    env = os.environ.get("PLANLAB_EDGE")
    if env and Path(env).exists():
        return env
    for cand in EDGE_CANDIDATES:
        p = Path(cand)
        if p.exists():
            return str(p)
        which = shutil.which(cand)
        if which:
            return which
    return None


def html_to_pdf(html_path: Path, pdf_path: Path, *, timeout: int = 180, budget: int = 20000) -> bool:
    edge = find_edge()
    if edge is None:
        warn("Microsoft Edge not found; skipping PDF (set PLANLAB_EDGE)")
        return False
    profile = BUILD / "edge-profile"
    profile.mkdir(parents=True, exist_ok=True)
    cmd = [
        edge,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-sync",
        "--run-all-compositor-stages-before-draw",
        "--disable-breakpad",
        "--disable-crash-reporter",
        f"--crash-dumps-dir={BUILD / 'crash'}",
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        f"--user-data-dir={profile}",
        f"--virtual-time-budget={budget}",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    info(f"Edge: {' '.join(cmd[:9])} … {pdf_path.name}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        warn(f"Edge timed out after {timeout}s")
        return False
    dt = time.time() - t0
    produced = pdf_path.exists() and pdf_path.stat().st_size > 0
    if not produced:
        warn(f"no PDF produced (Edge exit {proc.returncode}): {proc.stderr.strip()[:600]}")
        return False
    if proc.returncode != 0:
        # headless Edge often exits non-zero after writing the file, from crashpad
        # noise it cannot write outside the workspace; the artefact is what matters
        print(f"[build_paper] note: Edge exit {proc.returncode} after writing the PDF",
              file=sys.stderr)
    info(f"PDF written in {dt:.1f}s ({pdf_path.stat().st_size/1024:.0f} KiB)")
    return True


def pdf_pages(path: Path) -> int:
    data = path.read_bytes()
    counts = [int(x) for x in re.findall(rb"/Type\s*/Pages[^>]*?/Count\s+(\d+)", data)]
    counts += [int(x) for x in re.findall(rb"/Count\s+(\d+)\s*/Type\s*/Pages", data)]
    if counts:
        return max(counts)
    return len(re.findall(rb"/Type\s*/Page[^s]", data))


# --------------------------------------------------------------------------- #
# Word counting
# --------------------------------------------------------------------------- #


def word_count(md: str) -> dict[str, int]:
    text = re.sub(r"(?ms)^```.*?```", " ", md)
    text = re.sub(r"(?ms)^\s*\|.*?\|.*?$", " ", text)          # tables
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"\[\[[^\]]+\]\]", " ", text)                 # slots/refs
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)        # display math
    text = re.sub(r"\$[^$\n]*\$", " ", text)                    # inline math
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+|^\s*\d+\.\s+", "", text, flags=re.M)
    words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", text)
    return {"words": len(words), "chars": len(text)}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def build(args: argparse.Namespace) -> int:
    if not MAIN_MD.exists():
        warn(f"missing {MAIN_MD}")
        return 2
    start = time.time()
    raw = MAIN_MD.read_text(encoding="utf-8")
    meta, body_md = split_front_matter(raw)
    entries = parse_bibliography(BIB_MD)

    st = BuildState()
    st.eq_counter = 0            # type: ignore[attr-defined]
    st.fig_counter = 0           # type: ignore[attr-defined]
    st.tab_counter = 0           # type: ignore[attr-defined]
    st.sections_order = []       # type: ignore[attr-defined]

    # --- Stage 1: lift non-markdown constructs out of the source -------------
    src = extract_page_breaks(body_md, st)
    src = extract_fenced_divs(src, st)
    src = extract_equations(src, st)
    src = extract_floats(src, st)
    src = extract_refs_and_cites(src, st)
    src = collect_generic_slots(src, st)

    # --- Stage 2: markdown -> html ------------------------------------------
    md = make_markdown()
    body = md.convert(src)
    body = number_headings(body, st)
    body = resolve_sentinels(body, st, entries)

    # record section order for the (screen only) table of contents
    for m in HEAD_RE.finditer(body):
        if int(m.group(1)) != 2:
            continue
        am = ATTR_ID_RE.search(m.group(2))
        if am:
            txt = re.sub(r"<[^>]+>", "", m.group(3))
            txt = re.sub(r"^(Appendix\s+[A-Z])\s*", "", txt.strip())
            st.sections_order.append((txt.strip(), am.group(1)))   # type: ignore[attr-defined]

    # --- Stage 3: assemble ---------------------------------------------------
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    abstract = ""
    if meta.get("abstract"):
        amd = make_markdown()
        abs_src = extract_refs_and_cites(meta["abstract"], st)
        abs_src = collect_generic_slots(abs_src, st)
        abstract = resolve_sentinels(amd.convert(abs_src), st, entries)
    parts = [
        HEAD_TMPL.format(title=html.escape(meta.get("title", "PlanLab paper")), css=CSS, stamp=stamp),
        build_titleblock(meta),
    ]
    if abstract:
        parts.append(f'<section class="abstract"><div class="abshead">Abstract</div>{abstract}</section>')
    if args.toc:
        parts.append(build_toc(st))
    parts.append(body)
    if entries:
        refs = build_references(st, entries)
        parts.append(f'<h2 class="unnumbered" id="sec:references">References</h2>\n{refs}')
    parts.append(MATH_DEPS)
    parts.append(MATH_SCRIPT)
    parts.append("</body></html>")
    doc = "\n".join(parts)
    MAIN_HTML.write_text(doc, encoding="utf-8")
    info(f"HTML written: {MAIN_HTML.relative_to(ROOT)} ({len(doc)/1024:.0f} KiB)")

    # --- Stage 4: pdf ---------------------------------------------------------
    pages = None
    pdf_ok = False
    if not args.no_pdf:
        pdf_ok = html_to_pdf(MAIN_HTML, MAIN_PDF)
        if pdf_ok:
            pages = pdf_pages(MAIN_PDF)

    # --- report --------------------------------------------------------------
    dup = sorted(set(re.findall(
        r"\b(Section|Figure|Table|Appendix|Proposition|Corollary|Lemma)\s+\1\b",
        re.sub(r"<[^>]+>", " ", body),
    )))
    if dup:
        warn(f"duplicated nouns after reference resolution: {dup}")
    wc = word_count(body_md)
    ref_words = word_count("\n".join(e["raw"] for e in entries.values()))["words"] if entries else 0
    unresolved = sorted(set(st.unresolved))
    report = {
        "built": stamp,
        "source": str(MAIN_MD.relative_to(ROOT)),
        "html_bytes": MAIN_HTML.stat().st_size,
        "pdf": str(MAIN_PDF.relative_to(ROOT)) if pdf_ok else None,
        "pdf_bytes": MAIN_PDF.stat().st_size if pdf_ok else None,
        "pdf_pages": pages,
        "words_body": wc["words"],
        "words_bibliography": ref_words,
        "civil_words_body": round(wc["words"] / 1000, 1),
        "figures": len(st.figures),
        "tables": len(st.tables),
        "equations": st.eq_counter,          # type: ignore[attr-defined]
        "citations_used": len(st.cite_order),
        "bibliography_entries": len(entries),
        "bibliography_uncited": sorted(set(entries) - set(st.cite_order)),
        "missing_figure_files": sorted(set(st.missing_figures)),
        "missing_captions": sorted(set(st.missing_captions)),
        "missing_table_files": sorted(set(st.missing_tables)),
        "unresolved_slots": unresolved,
        "duplicated_nouns": dup,
        "sections": {k: v[0] for k, v in st.sections.items()},
        "python": platform.python_version(),
        "markdown": __import__("markdown").__version__,
        "elapsed_s": round(time.time() - start, 2),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("")
    print("  PlanLab paper build")
    print("  -------------------")
    print(f"  body words ............ {report['words_body']:>7,}  ({report['civil_words_body']}k)")
    print(f"  bibliography .......... {report['bibliography_entries']:>7}  entries "
          f"({report['citations_used']} cited)")
    print(f"  floats ................ {report['figures']:>7} figures, {report['tables']} tables, "
          f"{report['equations']} equations")
    print(f"  html .................. {report['html_bytes']/1024:>7.0f} KiB")
    if pdf_ok:
        print(f"  pdf ................... {report['pdf_bytes']/1024:>7.0f} KiB, "
              f"{report['pdf_pages']} pages")
    else:
        print("  pdf ................... SKIPPED")
    if unresolved:
        print(f"  open slots ............ {len(unresolved)}")
        for s in unresolved[:40]:
            print(f"      {s}")
    if report["missing_figure_files"]:
        print(f"  missing figure files .. {report['missing_figure_files']}")
    if report["missing_table_files"]:
        print(f"  missing table files ... {report['missing_table_files']}")
    print("")

    if args.strict:
        problems = []
        if unresolved:
            problems.append(f"{len(unresolved)} unresolved slots")
        if report["missing_figure_files"]:
            problems.append("missing figure files")
        if report["missing_table_files"]:
            problems.append("missing table files")
        if report["bibliography_uncited"]:
            problems.append(f"{len(report['bibliography_uncited'])} uncited bib entries")
        if not args.no_pdf and not pdf_ok:
            problems.append("PDF not produced")
        if problems:
            warn("--strict failed: " + "; ".join(problems))
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pdf", action="store_true", help="stop after the HTML build")
    ap.add_argument("--strict", action="store_true", help="fail on unresolved slots or missing assets")
    ap.add_argument("--toc", action="store_true", help="insert a screen-only table of contents")
    ap.add_argument("--stats-only", action="store_true", help="print word counts and exit")
    ap.add_argument("--timeout", type=int, default=180, help="Edge timeout in seconds")
    args = ap.parse_args(argv)
    if args.stats_only:
        raw = MAIN_MD.read_text(encoding="utf-8")
        meta, body = split_front_matter(raw)
        print(json.dumps({"front_matter": meta, **word_count(body)}, indent=2))
        return 0
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
