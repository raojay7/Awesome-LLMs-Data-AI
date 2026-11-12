#!/usr/bin/env python3
"""Rank discovered papers with a transparent, configurable quality proxy.

The rules live in config/quality_ranking.yaml. Every point in the score comes
from a boolean metadata signal or a bounded normalisation, and the hits are
stored in `quality_evidence` so any score can be audited. This is a reading /
curation priority, NOT a judgement of scientific quality: the corpus has no
citation counts, review decisions, or full texts.

Outputs:
  * `quality_score`, `quality_grade`, `quality_breakdown`, `quality_evidence`
    written back into data/papers.json
  * PAPER_QUALITY_RANKING.md with the rules, distribution and ranked lists
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

DIMENSION_ORDER = ["evidence", "reproducibility", "maturity", "fit", "clarity"]
GRADE_ORDER = ["A", "B", "C", "D"]


def clean_title(text: str) -> str:
    text = " ".join(str(text or "").split())
    return re.sub(r"[\[\]]", " ", text).strip()


def compile_patterns(cfg: dict) -> dict:
    return {k: re.compile(v, re.I) for k, v in cfg.get("patterns", {}).items()}


def score_paper(paper: dict, cfg: dict, patterns: dict) -> dict:
    abstract = paper.get("abstract") or ""
    title = paper.get("title") or ""
    idea = paper.get("idea") or ""
    settings = cfg.get("settings", {})

    hits: dict[str, bool] = {}
    for name, rx in patterns.items():
        text = idea if name == "contribution_sentence" else title if name == "named_method" else abstract
        hits[name] = bool(rx.search(text))

    relevance = paper.get("relevance_score") or 0
    cap = settings.get("relevance_cap", 20)
    families = len(paper.get("source_families") or [])
    categories = len(paper.get("categories") or [])
    authors = len(paper.get("authors") or [])
    arxiv_id = str(paper.get("arxiv_id") or "")
    published = (paper.get("published") or "")[:10]
    updated = (paper.get("updated") or "")[:10]
    abs_len = len(abstract)

    hits["peer_reviewed"] = bool(paper.get("doi") or paper.get("journal_ref"))
    hits["revised"] = (updated > published) or bool(re.search(r"v[2-9]$", arxiv_id))
    hits["multi_author"] = authors >= 3
    hits["families"] = families >= settings.get("families_threshold", 2)
    hits["cross_tier"] = bool(paper.get("cross_tiers"))
    hits["categories"] = categories >= settings.get("categories_threshold", 3)
    hits["abstract_length"] = settings.get("abstract_min", 1000) <= abs_len <= settings.get("abstract_max", 2200)

    breakdown: dict[str, float] = {}
    for dim in DIMENSION_ORDER:
        signals = cfg["dimensions"][dim]["signals"]
        value = 0.0
        for signal, weight in signals.items():
            if signal == "relevance_norm":
                value += round(min(relevance, cap) / cap * weight, 1)
            elif hits.get(signal):
                value += weight
        breakdown[dim] = round(value, 1)

    total = sum(breakdown.values())
    penalty = cfg.get("penalties", {}).get("borderline_other", {})
    penalised = (
        paper.get("subcategory") == penalty.get("subcategory")
        and relevance < penalty.get("relevance_below", 0)
    )
    if penalised:
        total += penalty.get("score", 0)
    total = max(0.0, min(100.0, round(total, 1)))

    grades = cfg.get("grades", {})
    grade = "D"
    for g in GRADE_ORDER:
        if g in grades and total >= grades[g]:
            grade = g
            break

    return {
        "quality_score": total,
        "quality_grade": grade,
        "quality_breakdown": breakdown,
        "quality_evidence": sorted(k for k, v in hits.items() if v),
        "quality_penalised": penalised,
    }


def evidence_short(paper: dict) -> str:
    preferred = [
        "peer_reviewed", "code_link", "release_claim", "ablation",
        "numeric_result", "baseline", "revised",
    ]
    hits = set(paper.get("quality_evidence") or [])
    picked = [k for k in preferred if k in hits][:3]
    return ", ".join(picked) if picked else "—"


def link(paper: dict) -> str:
    return f"[{clean_title(paper.get('title'))}]({paper.get('url') or '#'})"


def render_report(papers: list, cfg: dict, top_n: int) -> str:
    now = datetime.now(timezone.utc)
    ranked = sorted(
        papers,
        key=lambda p: (-(p.get("quality_score") or 0), -(p.get("relevance_score") or 0), p.get("title", "")),
    )
    grades = Counter(p.get("quality_grade") for p in ranked)
    scores = sorted(p.get("quality_score") or 0 for p in ranked)
    n = len(scores)
    quantile = lambda q: scores[min(n - 1, int(n * q))]
    total = len(ranked)

    lines = [
        "# Paper Quality Ranking",
        "",
        "> Auto-generated from `data/papers.json` by `scripts/rank_papers.py` using [`config/quality_ranking.yaml`](config/quality_ranking.yaml).",
        f"> Last refreshed: **{now.strftime('%Y-%m-%d %H:%M UTC')}** · {total:,} papers.",
        "",
        "**This is a reproducible proxy score, not a judgement of scientific quality.** The corpus has no citation counts, review decisions, or full texts; every point comes from a boolean metadata signal or a bounded normalisation. Use it to prioritise reading and curation, then apply human judgement.",
        "",
        "## The rule",
        "",
        "`score = evidence + reproducibility + maturity + fit + clarity − penalty`, clipped to 0–100.",
        "",
        "| Dimension | Max | Signals (points) |",
        "| --- | --- | --- |",
    ]
    for dim in DIMENSION_ORDER:
        d = cfg["dimensions"][dim]
        sig = ", ".join(f"`{k}` +{v}" for k, v in d["signals"].items())
        cap = sum(d["signals"].values())
        lines.append(f"| {dim} · {d['label']} | {cap} | {sig} |")
    penalty = cfg.get("penalties", {}).get("borderline_other", {})
    lines += [
        "",
        f"Penalty: subcategory `{penalty.get('subcategory')}` with relevance < {penalty.get('relevance_below')} gets **{penalty.get('score')}** (classification-boundary hits).",
        f"Grades: **A** ≥ {cfg['grades']['A']} · **B** ≥ {cfg['grades']['B']} · **C** ≥ {cfg['grades']['C']} · **D** below that.",
        "",
        "Signal definitions are regexes in the config file; `relevance_norm` is `min(relevance_score, cap) / cap × 12`. `quality_evidence` in `data/papers.json` records exactly which signals fired for each paper.",
        "",
        "## Distribution",
        "",
        "| Grade | Threshold | Papers | Share |",
        "| --- | --- | --- | --- |",
    ]
    for g in GRADE_ORDER:
        c = grades.get(g, 0)
        threshold = cfg["grades"].get(g, "—")
        lines.append(f"| {g} | {threshold} | {c:,} | {100.0 * c / total:.1f}% |")
    lines += [
        "",
        f"Score quantiles: p10 **{quantile(0.1):.1f}** · p50 **{quantile(0.5):.1f}** · p90 **{quantile(0.9):.1f}** · max **{max(scores):.1f}**.",
        "",
        "| Primary tier | Papers | Mean quality |",
        "| --- | --- | --- |",
    ]
    by_tier: dict[str, list] = {}
    for p in ranked:
        by_tier.setdefault(p.get("primary_tier", "Unclassified"), []).append(p.get("quality_score") or 0)
    for tier, vals in sorted(by_tier.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {tier} | {len(vals):,} | {sum(vals) / len(vals):.1f} |")

    lines += [
        "",
        f"## Top {top_n}",
        "",
        "| # | Score | Grade | Paper | Tier / subcategory | Key evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i, p in enumerate(ranked[:top_n], 1):
        lines.append(
            f"| {i} | {p.get('quality_score')} | {p.get('quality_grade')} | {link(p)} | "
            f"{p.get('primary_tier')} / {p.get('subcategory')} | {evidence_short(p)} |"
        )

    lines += ["", "## Top 15 by primary tier", ""]
    for tier in ["Data Substrates", "Data Creation and Selection", "Data Ingestion Strategies"]:
        group = [p for p in ranked if p.get("primary_tier") == tier][:15]
        lines += [f"### {tier}", "", "| Score | Paper | Idea |", "| --- | --- | --- |"]
        for p in group:
            idea = " ".join(str(p.get("idea") or "").split())[:150]
            lines.append(f"| {p.get('quality_score')} | {link(p)} | {idea}… |")
        lines.append("")

    lines += [
        "## Where quality disagrees with keyword relevance",
        "",
        "High relevance but low quality (keyword-heavy, weak evidence):",
        "",
        "| Paper | relevance | quality |",
        "| --- | --- | --- |",
    ]
    for p in sorted(ranked, key=lambda x: -((x.get("relevance_score") or 0) - (x.get("quality_score") or 0) / 5))[:8]:
        lines.append(f"| {link(p)} | {p.get('relevance_score')} | {p.get('quality_score')} |")
    lines += [
        "",
        "High quality but modest relevance (strong empirical work, weaker topical keyword match):",
        "",
        "| Paper | relevance | quality |",
        "| --- | --- | --- |",
    ]
    for p in [x for x in ranked if (x.get("relevance_score") or 0) <= 8][:8]:
        lines.append(f"| {link(p)} | {p.get('relevance_score')} | {p.get('quality_score')} |")

    lines += [
        "",
        "## Limitations",
        "",
        "- No citations, venue reputation, reviewer scores, or full-text reading; a paper with a polished abstract can score well regardless of correctness.",
        "- Regex hits are coarse: mentioning a benchmark is not the same as releasing one, and `sota_claim` is a self-report.",
        "- `peer_reviewed` / `revised` are sparse in this corpus (about 2% and 9%), so maturity mostly separates a small minority.",
        "- The rule is deliberately configurable: edit `config/quality_ranking.yaml` and re-run `python scripts/rank_papers.py` to change weights or thresholds.",
        "",
        "Full per-paper scores and signal hits are in `data/papers.json`; the complete index is [`PAPERS.md`](PAPERS.md), and the Chinese review is [`PAPER_IDEAS_ZH.md`](PAPER_IDEAS_ZH.md).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/quality_ranking.yaml"))
    parser.add_argument("--database", type=Path, default=Path("data/papers.json"))
    parser.add_argument("--output", type=Path, default=Path("PAPER_QUALITY_RANKING.md"))
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--sample", type=int, default=0, help="print N scored examples without writing")
    parser.add_argument("--force", action="store_true", help="recompute scores already present")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    patterns = compile_patterns(cfg)
    db = json.loads(args.database.read_text(encoding="utf-8"))
    papers = db.get("papers", [])

    rule_version = cfg.get("version", 1)
    # Bumping `version` in the config invalidates every stored score.
    force = args.force or db.get("quality_rule_version") != rule_version
    computed = 0
    for p in papers:
        if not force and p.get("quality_score") is not None:
            continue
        p.update(score_paper(p, cfg, patterns))
        computed += 1

    if args.sample:
        for p in papers[: args.sample]:
            r = score_paper(p, cfg, patterns)
            print(f"- {p.get('title')}\n  score={r['quality_score']} grade={r['quality_grade']} "
                  f"breakdown={r['quality_breakdown']}\n  evidence={r['quality_evidence']}\n")
        print(f"[sample] {min(args.sample, len(papers))} shown; database unchanged.")
        return 0

    report = render_report(papers, cfg, args.top)
    existing = args.output.read_text(encoding="utf-8") if args.output.exists() else None
    stamp = re.compile(r"> Last refreshed: \*\*[^*]+\*\*")
    report_changed = existing is None or stamp.sub("> Last refreshed: **<stamp>**", existing) != stamp.sub(
        "> Last refreshed: **<stamp>**", report
    )
    if report_changed:
        args.output.write_text(report, encoding="utf-8")

    if computed or db.get("quality_rule_version") != rule_version:
        db["quality_rule_version"] = rule_version
        args.database.write_text(json.dumps(db, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Scored {computed:,} new papers ({len(papers):,} total); "
        f"report {'updated' if report_changed else 'unchanged'}: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
