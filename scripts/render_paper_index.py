#!/usr/bin/env python3
"""Generate PAPERS.md, a complete index of the literature bot's paper database.

Standard library only. Called by `update-repo-status.yml` so that every paper
ever discovered stays visible even when weekly PRs have not been merged. This
is a discovery archive, not the human-curated reading list in README.md.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

TIERS = [
    "Data Substrates",
    "Data Creation and Selection",
    "Data Ingestion Strategies",
]
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def clean_title(text: str) -> str:
    text = " ".join(str(text or "").split())
    # Square brackets would break the Markdown link syntax.
    return re.sub(r"[\[\]]", " ", text).strip()


def author_short(authors: list) -> str:
    if not authors:
        return "Unknown authors"
    return authors[0] if len(authors) == 1 else f"{authors[0]} et al."


def year_of(paper: dict) -> str:
    published = paper.get("published") or ""
    return published[:4] if re.match(r"^\d{4}", published) else "n.d."


def month_of(paper: dict) -> str | None:
    published = paper.get("published") or ""
    month = published[:7]
    return month if MONTH_RE.match(month) else None


def entry_line(paper: dict) -> str:
    title = clean_title(paper.get("title", "Untitled"))
    url = paper.get("url") or ""
    line = (
        f"- **[{title}]({url})** — {author_short(paper.get('authors') or [])} "
        f"({year_of(paper)}) · relevance {paper.get('relevance_score', '-')}"
    )
    if paper.get("quality_score") is not None:
        line += f" · quality {paper.get('quality_score')} ({paper.get('quality_grade', '-')})"
    cross = paper.get("cross_tiers") or []
    if cross:
        line += f" · _cross-tier: {', '.join(cross)}_"
    idea = " ".join(str(paper.get("idea") or "").split())
    if idea:
        line += f"\n  _Idea: {idea}_"
    return line


def render(db: dict) -> str:
    papers = db.get("papers", [])
    now = datetime.now(timezone.utc)
    months = defaultdict(int)
    by_tier: dict[str, list] = defaultdict(list)
    for p in papers:
        by_tier[p.get("primary_tier", "Unclassified")].append(p)
        m = month_of(p)
        if m:
            months[m] += 1

    tier_order = TIERS + sorted(t for t in by_tier if t not in TIERS)
    ordered_months = sorted(months)
    span = (
        f"**{ordered_months[0]}** – **{ordered_months[-1]}**"
        if ordered_months
        else "—"
    )

    lines = [
        "# Automatically Discovered Paper Index",
        "",
        "> Auto-generated from `data/papers.json` by `scripts/render_paper_index.py` and kept current by `update-repo-status.yml`.",
        f"> Last refreshed: **{now.strftime('%Y-%m-%d %H:%M UTC')}**.",
        "",
        f"**{len(papers):,} unique papers** · published {span} · grouped by the repository's primary tier and subcategory.",
        "",
        "This is a **discovery archive**: entries were proposed by the weekly literature bot, grouped by its deterministic taxonomy, and have **not** been human-reviewed. The curated reading list lives in [README.md](README.md); the newest entries also appear there in the recent-paper block.",
        "",
        "Each entry carries a one-sentence **Idea** extracted deterministically from its arXiv abstract by `scripts/extract_ideas.py` (the contribution sentence, not a model-generated summary). Chinese one-sentence summaries for the highest-relevance papers are in [`PAPER_IDEAS_ZH.md`](PAPER_IDEAS_ZH.md).",
        "",
        "Within every subcategory, papers are **ranked by the quality proxy score** (ties broken by publication date); the rules and the global ranking are in [`PAPER_QUALITY_RANKING.md`](PAPER_QUALITY_RANKING.md). `relevance` measures topical keyword fit, `quality` measures how much evidence the paper reports — they are different things.",
        "",
        "## Contents",
        "",
        "| Tier | Papers | Subcategories |",
        "| --- | --- | --- |",
    ]
    for tier in tier_order:
        group = by_tier.get(tier, [])
        subcats = len({p.get("subcategory", "Other") for p in group})
        lines.append(f"| [{tier}](#{tier.lower().replace(' ', '-')}) | {len(group):,} | {subcats} |")

    lines += ["", "## Papers by publication month", "", "| Month | Papers |", "| --- | --- |"]
    for month in ordered_months:
        lines.append(f"| {month} | {months[month]:,} |")
    if not ordered_months:
        lines.append("| _No dated papers._ | — |")

    for tier in tier_order:
        group = by_tier.get(tier, [])
        if not group:
            continue
        lines += ["", f"## {tier}", ""]
        buckets: dict[str, list] = defaultdict(list)
        for p in group:
            buckets[p.get("subcategory") or "Other"].append(p)
        for subcat in sorted(buckets, key=lambda s: (-len(buckets[s]), s)):
            # Rank by the quality proxy first, then by recency within a tie.
            items = sorted(
                buckets[subcat],
                key=lambda p: (p.get("quality_score") or 0, p.get("published") or ""),
                reverse=True,
            )
            lines += [f"### {subcat} ({len(items)})", ""]
            lines.extend(entry_line(p) for p in items)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def without_stamp(text: str) -> str:
    """Ignore the refresh timestamp when deciding whether a commit is needed."""
    return re.sub(r"> Last refreshed: \*\*[^*]+\*\*", "> Last refreshed: **<stamp>**", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/papers.json"))
    parser.add_argument("--output", type=Path, default=Path("PAPERS.md"))
    args = parser.parse_args()

    db = json.loads(args.database.read_text(encoding="utf-8"))
    content = render(db)
    existing = args.output.read_text(encoding="utf-8") if args.output.exists() else None
    if existing is not None and without_stamp(existing) == without_stamp(content):
        print(f"{args.output} already up to date ({len(db.get('papers', [])):,} papers).")
        return 0
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {args.output}: {len(db.get('papers', [])):,} papers, {content.count(chr(10)):,} lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
