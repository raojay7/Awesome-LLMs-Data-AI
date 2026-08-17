#!/usr/bin/env python3
"""Automatic literature discovery for Awesome-LLMs-Data-AI.

The bot fetches recent arXiv records, filters and classifies them using the
repository's operation-centered taxonomy, updates a bounded README block, and
writes a pull-request summary. Classification is deliberately deterministic so
that every placement can be audited and tuned in config/paper_bot.yaml.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import feedparser
import requests
import yaml

ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", re.I)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]{8,250})\]\((https?://[^)]+)\)")
SPACE_RE = re.compile(r"\s+")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_ws(text: str) -> str:
    return SPACE_RE.sub(" ", text or "").strip()


def normalize_title(title: str) -> str:
    title = html.unescape(title or "").lower()
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    return SPACE_RE.sub(" ", title).strip()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def arxiv_id_from_url(url: str) -> str | None:
    m = ARXIV_ID_RE.search(url or "")
    return m.group(1).replace(".pdf", "") if m else None


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_database(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "updated_at": None, "papers": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("schema_version", 1)
    data.setdefault("papers", [])
    return data


def read_readme_titles(readme_text: str) -> list[str]:
    return [normalize_ws(m.group(1)) for m in MARKDOWN_LINK_RE.finditer(readme_text)]


def read_readme_arxiv_ids(readme_text: str) -> set[str]:
    return {m.group(1).replace(".pdf", "") for m in ARXIV_ID_RE.finditer(readme_text)}


def request_arxiv(endpoint: str, query: str, max_results: int, timeout: int, user_agent: str) -> list[dict[str, Any]]:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(endpoint, params=params, headers={"User-Agent": user_agent}, timeout=timeout)
    response.raise_for_status()
    feed = feedparser.parse(response.text)

    papers: list[dict[str, Any]] = []
    for entry in feed.entries:
        entry_id = normalize_ws(getattr(entry, "id", ""))
        arxiv_id = arxiv_id_from_url(entry_id) or entry_id.rsplit("/", 1)[-1]
        links = getattr(entry, "links", []) or []
        pdf_url = next((getattr(x, "href", None) for x in links if getattr(x, "type", "") == "application/pdf"), None)
        categories = [getattr(tag, "term", "") for tag in (getattr(entry, "tags", []) or []) if getattr(tag, "term", None)]
        authors = [normalize_ws(getattr(a, "name", "")) for a in (getattr(entry, "authors", []) or []) if getattr(a, "name", None)]
        papers.append({
            "arxiv_id": arxiv_id,
            "title": normalize_ws(getattr(entry, "title", "")),
            "abstract": normalize_ws(getattr(entry, "summary", "")),
            "authors": authors,
            "published": getattr(entry, "published", None),
            "updated": getattr(entry, "updated", None),
            "categories": categories,
            "url": entry_id,
            "pdf_url": pdf_url,
            "doi": getattr(entry, "arxiv_doi", None),
            "journal_ref": getattr(entry, "arxiv_journal_ref", None),
            "source": "arXiv",
        })
    return papers


def allowed_category(paper: dict[str, Any], allowed: set[str]) -> bool:
    return True if not allowed else bool(set(paper.get("categories", [])) & allowed)


def must_have_llm_context(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(term.lower() in low for term in terms)


def relevance_score(paper: dict[str, Any], positive: dict[str, int], negative: dict[str, int], families: set[str]) -> int:
    title, abstract = paper["title"].lower(), paper["abstract"].lower()
    score = len(families)
    for kw, weight in positive.items():
        kw = kw.lower()
        if kw in title:
            score += int(weight) * 2
        elif kw in abstract:
            score += int(weight)
    for kw, weight in negative.items():
        kw = kw.lower()
        if kw in title:
            score -= int(weight) * 2
        elif kw in abstract:
            score -= int(weight)
    return score


def classify_taxonomy(paper: dict[str, Any], families: set[str], cfg: dict[str, Any]):
    text = f"{paper['title']} {paper['abstract']}".lower()
    priors = cfg["family_tier_prior"]
    tier_scores: dict[str, int] = {}
    best_subcat: dict[str, tuple[str, int]] = {}

    for tier, tier_cfg in cfg["tiers"].items():
        score = int(tier_cfg.get("prior_weight", 3)) if any(priors.get(f) == tier for f in families) else 0
        best = ("Other", 0)
        for subcat, keywords in tier_cfg["subcategories"].items():
            subscore = 0
            for keyword in keywords:
                kw = keyword.lower()
                if kw in paper["title"].lower():
                    subscore += 4
                elif kw in text:
                    subscore += 2
            if subscore > best[1]:
                best = (subcat, subscore)
            score += subscore
        tier_scores[tier] = score
        best_subcat[tier] = best

    ranked = sorted(tier_scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
    primary, primary_score = ranked[0]
    cross = [tier for tier, score in ranked[1:] if score >= 4 and score >= max(4, primary_score - 3)]
    return primary, best_subcat[primary][0], cross, tier_scores


def is_duplicate(paper: dict[str, Any], ids: set[str], titles: list[str], threshold: float) -> bool:
    if paper.get("arxiv_id") in ids:
        return True
    norm = normalize_title(paper["title"])
    for title in titles:
        if norm == normalize_title(title) or title_similarity(paper["title"], title) >= threshold:
            return True
    return False


def author_short(authors: list[str]) -> str:
    if not authors:
        return "Unknown authors"
    return authors[0] if len(authors) == 1 else f"{authors[0]} et al."


def year_of(paper: dict[str, Any]) -> str:
    dt = parse_dt(paper.get("published"))
    return str(dt.year) if dt else "n.d."


def render_auto_block(papers: list[dict[str, Any]], project: dict[str, Any], render: dict[str, Any]) -> str:
    now = utc_now()
    cutoff = now - timedelta(days=int(project.get("recent_window_days", 120)))
    recent = []
    for p in papers:
        d, pub = parse_dt(p.get("discovered_at")), parse_dt(p.get("published"))
        if (d and d >= cutoff) or (pub and pub >= cutoff):
            recent.append(p)
    recent.sort(key=lambda p: parse_dt(p.get("published")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    recent = recent[: int(project.get("max_readme_papers", 30))]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in recent:
        grouped[p["primary_tier"]].append(p)

    lines = [
        project["marker_start"],
        project["section_heading"],
        "",
        "> This section is generated by the weekly literature bot and reviewed through pull requests before merge.",
        f"> Last generated: **{now.date().isoformat()}** · Showing up to **{len(recent)}** recent papers.",
        "",
    ]
    for tier in ["Data Substrates", "Data Creation and Selection", "Data Ingestion Strategies"]:
        if not grouped.get(tier):
            continue
        lines += [f"### {tier}", ""]
        for p in grouped[tier]:
            cross = ""
            if render.get("include_cross_tier_label", True) and p.get("cross_tiers"):
                cross = f" · _Cross-tier: {', '.join(p['cross_tiers'])}_"
            lines.append(f"- **[{p['title']}]({p['url']})** — {author_short(p.get('authors', []))} ({year_of(p)}). *{p.get('subcategory', 'Other')}*{cross}")
        lines.append("")
    if not recent:
        lines += ["_No recent bot-curated papers yet._", ""]
    lines.append(project["marker_end"])
    return "\n".join(lines)


def replace_or_insert_block(readme: str, block: str, project: dict[str, Any]) -> str:
    start, end = project["marker_start"], project["marker_end"]
    if start in readme and end in readme:
        return re.sub(re.escape(start) + r".*?" + re.escape(end), block, readme, count=1, flags=re.S)
    anchor = project.get("insert_before_heading")
    if anchor and anchor in readme:
        return readme.replace(anchor, block + "\n\n" + anchor, 1)
    fallback = "## Contributing"
    if fallback in readme:
        return readme.replace(fallback, block + "\n\n" + fallback, 1)
    return readme.rstrip() + "\n\n" + block + "\n"


def build_pr_body(new: list[dict[str, Any]], render: dict[str, Any]) -> str:
    lines = [
        "## Automated literature update", "",
        f"Found **{len(new)}** new candidate paper(s) after date/category/relevance filtering and README/database deduplication.", "",
        "### Review checklist",
        "- [ ] Titles and links are correct",
        "- [ ] Papers are in scope for the LLM data lifecycle",
        "- [ ] Primary tier/subcategory placement is reasonable",
        "- [ ] Cross-tier labels are reasonable where shown",
        "- [ ] No duplicate publication versions are being added", "",
    ]
    if not new:
        return "\n".join(lines + ["_No new papers._"])
    for p in new:
        lines += [
            f"### {p['title']}",
            f"- **Primary:** {p['primary_tier']} → {p['subcategory']}",
            f"- **Cross-tier:** {', '.join(p.get('cross_tiers', [])) or 'None'}",
            f"- **Relevance score:** {p['relevance_score']}",
            f"- **Tier scores:** `{json.dumps(p.get('tier_scores', {}), ensure_ascii=False)}`",
            f"- **arXiv:** {p['url']}",
            f"- **Published:** {p.get('published') or 'n.d.'}",
        ]
        if render.get("show_abstract_in_pr", True):
            n = int(render.get("abstract_chars_in_pr", 500))
            abstract = p.get("abstract", "")
            if len(abstract) > n:
                abstract = abstract[: n - 1].rstrip() + "…"
            lines.append(f"- **Abstract:** {abstract}")
        lines.append("")
    return "\n".join(lines + ["---", "Generated by `scripts/paper_bot.py`. The bot proposes changes via PR; it does not auto-merge."])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--readme", type=Path, required=True)
    ap.add_argument("--database", type=Path, required=True)
    ap.add_argument("--lookback-days", type=int, default=None)
    ap.add_argument("--pr-body", type=Path, default=Path("/tmp/paper_bot_pr_body.md"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    project, fetch, rel, tax, render = cfg["project"], cfg["fetch"], cfg["relevance"], cfg["taxonomy"], cfg.get("render", {})
    lookback = args.lookback_days or int(fetch.get("lookback_days", 14))
    cutoff = utc_now() - timedelta(days=lookback)

    readme = args.readme.read_text(encoding="utf-8")
    db = load_database(args.database)
    known_titles = read_readme_titles(readme) + [p.get("title", "") for p in db["papers"] if p.get("title")]
    known_ids = read_readme_arxiv_ids(readme) | {p.get("arxiv_id") for p in db["papers"] if p.get("arxiv_id")}

    allowed = set(fetch.get("allowed_categories", []))
    query_items = [(family, q) for family, qs in fetch["queries"].items() for q in qs]
    aggregated: dict[str, dict[str, Any]] = {}

    for i, (family, query) in enumerate(query_items):
        print(f"[fetch {i+1}/{len(query_items)}] {family}: {query}")
        try:
            results = request_arxiv(fetch["endpoint"], query, int(fetch.get("max_results_per_query", 75)), int(fetch.get("timeout_seconds", 60)), fetch["user_agent"])
        except requests.RequestException as exc:
            print(f"WARNING: arXiv query failed: {exc}")
            results = []
        for p in results:
            key = p.get("arxiv_id") or normalize_title(p["title"])
            if key not in aggregated:
                aggregated[key] = p
                aggregated[key]["source_families"] = set()
            aggregated[key]["source_families"].add(family)
        if i < len(query_items) - 1:
            time.sleep(float(fetch.get("request_delay_seconds", 3.2)))

    candidates = []
    for p in aggregated.values():
        pub = parse_dt(p.get("published"))
        if not pub or pub < cutoff or not allowed_category(p, allowed):
            continue
        text = f"{p['title']} {p['abstract']}"
        if not must_have_llm_context(text, rel.get("must_have_any", [])):
            continue
        families = set(p["source_families"])
        score = relevance_score(p, rel["positive_keywords"], rel["negative_keywords"], families)
        if score < int(rel.get("min_score", 6)):
            continue
        if is_duplicate(p, known_ids, known_titles, float(rel.get("title_duplicate_similarity", 0.94))):
            continue
        primary, subcat, cross, tier_scores = classify_taxonomy(p, families, tax)
        p.update({
            "source_families": sorted(families),
            "relevance_score": score,
            "primary_tier": primary,
            "subcategory": subcat,
            "cross_tiers": cross,
            "tier_scores": tier_scores,
            "discovered_at": utc_now().isoformat(),
        })
        candidates.append(p)

    candidates.sort(key=lambda p: (p["relevance_score"], parse_dt(p.get("published")) or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    print(f"Fetched unique papers: {len(aggregated)}")
    print(f"New relevant candidates: {len(candidates)}")
    for p in candidates:
        print(f"  [{p['relevance_score']:>2}] {p['primary_tier']} / {p['subcategory']} :: {p['title']}")

    if args.dry_run:
        return 0

    if candidates:
        db["papers"].extend(candidates)
        db["updated_at"] = utc_now().isoformat()
        deduped, seen = [], set()
        for p in db["papers"]:
            key = p.get("arxiv_id") or normalize_title(p.get("title", ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)
        db["papers"] = deduped
        block = render_auto_block(db["papers"], project, render)
        args.readme.write_text(replace_or_insert_block(readme, block, project), encoding="utf-8")
        args.database.parent.mkdir(parents=True, exist_ok=True)
        args.database.write_text(json.dumps(db, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    args.pr_body.parent.mkdir(parents=True, exist_ok=True)
    args.pr_body.write_text(build_pr_body(candidates, render) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
