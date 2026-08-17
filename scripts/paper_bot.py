#!/usr/bin/env python3
"""LLM Data Literature Bot v2 for Awesome-LLMs-Data-AI.

Two operating modes are supported:

1. Weekly/live mode (default): discover new papers, update data/papers.json and
   the bounded auto-generated block in README.md, then let GitHub Actions open
   a PR.
2. Historical report mode (--report-only): scan an exact inclusive date range
   and write a standalone Markdown report without modifying README.md or the
   persistent database. This is designed for backfills such as
   2026-07-01 through 2026-08-17.

The classifier is deterministic and auditable. It follows the repository's
operation-centered taxonomy and supports secondary/cross-tier labels.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date, datetime, time as dt_time, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import feedparser
import requests
import yaml

ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", re.I)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]{8,250})\]\((https?://[^)]+)\)")
SPACE_RE = re.compile(r"\s+")


@dataclass
class SearchStats:
    query_requests: int = 0
    query_failures: int = 0
    raw_records: int = 0
    unique_records: int = 0
    in_date_range: int = 0
    category_pass: int = 0
    context_pass: int = 0
    relevance_pass: int = 0
    duplicates_removed: int = 0
    final_candidates: int = 0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def parse_date_arg(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD, got: {value}") from exc


def date_bounds(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    if end_date < start_date:
        raise ValueError(f"end date {end_date} is earlier than start date {start_date}")
    start = datetime.combine(start_date, dt_time.min, tzinfo=timezone.utc)
    end = datetime.combine(end_date, dt_time.max, tzinfo=timezone.utc)
    return start, end


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
        return {"schema_version": 2, "updated_at": None, "papers": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("schema_version", 2)
    data.setdefault("papers", [])
    return data


def read_readme_titles(readme_text: str) -> list[str]:
    return [normalize_ws(m.group(1)) for m in MARKDOWN_LINK_RE.finditer(readme_text)]


def read_readme_arxiv_ids(readme_text: str) -> set[str]:
    return {m.group(1).replace(".pdf", "") for m in ARXIV_ID_RE.finditer(readme_text)}


def parse_arxiv_feed(xml_text: str) -> list[dict[str, Any]]:
    feed = feedparser.parse(xml_text)
    papers: list[dict[str, Any]] = []
    for entry in feed.entries:
        entry_id = normalize_ws(getattr(entry, "id", ""))
        arxiv_id = arxiv_id_from_url(entry_id) or entry_id.rsplit("/", 1)[-1]
        links = getattr(entry, "links", []) or []
        pdf_url = next(
            (getattr(x, "href", None) for x in links if getattr(x, "type", "") == "application/pdf"),
            None,
        )
        categories = [
            getattr(tag, "term", "")
            for tag in (getattr(entry, "tags", []) or [])
            if getattr(tag, "term", None)
        ]
        authors = [
            normalize_ws(getattr(a, "name", ""))
            for a in (getattr(entry, "authors", []) or [])
            if getattr(a, "name", None)
        ]
        papers.append(
            {
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
            }
        )
    return papers


def request_arxiv_page(
    endpoint: str,
    query: str,
    start: int,
    page_size: int,
    timeout: int,
    user_agent: str,
    session: requests.Session,
) -> list[dict[str, Any]]:
    params = {
        "search_query": query,
        "start": start,
        "max_results": page_size,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = session.get(
        endpoint,
        params=params,
        headers={"User-Agent": user_agent},
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_arxiv_feed(response.text)


def fetch_query_range(
    *,
    endpoint: str,
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    page_size: int,
    max_pages: int,
    timeout: int,
    user_agent: str,
    request_delay: float,
    session: requests.Session,
    stats: SearchStats,
) -> list[dict[str, Any]]:
    """Fetch a query in descending submitted-date order until start_dt is crossed.

    Pagination is important for historical backfills: a single top-N request can
    silently miss older papers inside a multi-week date range.
    """
    collected: list[dict[str, Any]] = []
    for page in range(max_pages):
        offset = page * page_size
        stats.query_requests += 1
        try:
            records = request_arxiv_page(
                endpoint, query, offset, page_size, timeout, user_agent, session
            )
        except requests.RequestException as exc:
            stats.query_failures += 1
            print(f"WARNING: arXiv request failed at offset={offset}: {exc}")
            break

        stats.raw_records += len(records)
        if not records:
            break

        published_dates = [parse_dt(r.get("published")) for r in records]
        published_dates = [d for d in published_dates if d is not None]

        for record in records:
            pub = parse_dt(record.get("published"))
            if pub and start_dt <= pub <= end_dt:
                collected.append(record)

        # Results are sorted descending. Once the oldest item on a full page is
        # earlier than the requested start date, later pages cannot contain an
        # in-range paper.
        if published_dates and min(published_dates) < start_dt:
            break
        if len(records) < page_size:
            break
        if page < max_pages - 1:
            time.sleep(request_delay)
    return collected


def allowed_category(paper: dict[str, Any], allowed: set[str]) -> bool:
    return True if not allowed else bool(set(paper.get("categories", [])) & allowed)


def must_have_llm_context(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(term.lower() in low for term in terms)


def relevance_score(
    paper: dict[str, Any],
    positive: dict[str, int],
    negative: dict[str, int],
    families: set[str],
) -> int:
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


def classify_taxonomy(
    paper: dict[str, Any], families: set[str], cfg: dict[str, Any]
) -> tuple[str, str, list[str], dict[str, int]]:
    text = f"{paper['title']} {paper['abstract']}".lower()
    priors = cfg["family_tier_prior"]
    tier_scores: dict[str, int] = {}
    best_subcat: dict[str, tuple[str, int]] = {}

    for tier, tier_cfg in cfg["tiers"].items():
        score = (
            int(tier_cfg.get("prior_weight", 3))
            if any(priors.get(f) == tier for f in families)
            else 0
        )
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
    cross = [
        tier
        for tier, score in ranked[1:]
        if score >= 4 and score >= max(4, primary_score - 3)
    ]
    return primary, best_subcat[primary][0], cross, tier_scores


def is_duplicate(
    paper: dict[str, Any], ids: set[str], titles: list[str], threshold: float
) -> bool:
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


def classify_candidates(
    aggregated: dict[str, dict[str, Any]],
    *,
    start_dt: datetime,
    end_dt: datetime,
    known_ids: set[str],
    known_titles: list[str],
    fetch_cfg: dict[str, Any],
    relevance_cfg: dict[str, Any],
    taxonomy_cfg: dict[str, Any],
    stats: SearchStats,
) -> list[dict[str, Any]]:
    allowed = set(fetch_cfg.get("allowed_categories", []))
    threshold = float(relevance_cfg.get("title_duplicate_similarity", 0.94))
    candidates: list[dict[str, Any]] = []

    for p in aggregated.values():
        pub = parse_dt(p.get("published"))
        if not pub or not (start_dt <= pub <= end_dt):
            continue
        stats.in_date_range += 1
        if not allowed_category(p, allowed):
            continue
        stats.category_pass += 1

        text = f"{p['title']} {p['abstract']}"
        if not must_have_llm_context(text, relevance_cfg.get("must_have_any", [])):
            continue
        stats.context_pass += 1

        families = set(p["source_families"])
        score = relevance_score(
            p,
            relevance_cfg["positive_keywords"],
            relevance_cfg["negative_keywords"],
            families,
        )
        if score < int(relevance_cfg.get("min_score", 6)):
            continue
        stats.relevance_pass += 1

        if is_duplicate(p, known_ids, known_titles, threshold):
            stats.duplicates_removed += 1
            continue

        primary, subcat, cross, tier_scores = classify_taxonomy(p, families, taxonomy_cfg)
        p = dict(p)
        p.update(
            {
                "source_families": sorted(families),
                "relevance_score": score,
                "primary_tier": primary,
                "subcategory": subcat,
                "cross_tiers": cross,
                "tier_scores": tier_scores,
                "discovered_at": utc_now().isoformat(),
            }
        )
        candidates.append(p)

    candidates.sort(
        key=lambda p: (
            p["relevance_score"],
            parse_dt(p.get("published")) or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    stats.final_candidates = len(candidates)
    return candidates


def render_auto_block(
    papers: list[dict[str, Any]], project: dict[str, Any], render: dict[str, Any]
) -> str:
    now = utc_now()
    cutoff = now - timedelta(days=int(project.get("recent_window_days", 120)))
    recent = []
    for p in papers:
        d, pub = parse_dt(p.get("discovered_at")), parse_dt(p.get("published"))
        if (d and d >= cutoff) or (pub and pub >= cutoff):
            recent.append(p)
    recent.sort(
        key=lambda p: parse_dt(p.get("published"))
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
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
    for tier in [
        "Data Substrates",
        "Data Creation and Selection",
        "Data Ingestion Strategies",
    ]:
        if not grouped.get(tier):
            continue
        lines += [f"### {tier}", ""]
        for p in grouped[tier]:
            cross = ""
            if render.get("include_cross_tier_label", True) and p.get("cross_tiers"):
                cross = f" · _Cross-tier: {', '.join(p['cross_tiers'])}_"
            lines.append(
                f"- **[{p['title']}]({p['url']})** — "
                f"{author_short(p.get('authors', []))} ({year_of(p)}). "
                f"*{p.get('subcategory', 'Other')}*{cross}"
            )
        lines.append("")
    if not recent:
        lines += ["_No recent bot-curated papers yet._", ""]
    lines.append(project["marker_end"])
    return "\n".join(lines)


def replace_or_insert_block(readme: str, block: str, project: dict[str, Any]) -> str:
    start, end = project["marker_start"], project["marker_end"]
    if start in readme and end in readme:
        return re.sub(
            re.escape(start) + r".*?" + re.escape(end),
            block,
            readme,
            count=1,
            flags=re.S,
        )
    anchor = project.get("insert_before_heading")
    if anchor and anchor in readme:
        return readme.replace(anchor, block + "\n\n" + anchor, 1)
    fallback = "## Contributing"
    if fallback in readme:
        return readme.replace(fallback, block + "\n\n" + fallback, 1)
    return readme.rstrip() + "\n\n" + block + "\n"


def short_abstract(text: str, max_chars: int = 420) -> str:
    text = normalize_ws(text)
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def render_report(
    candidates: list[dict[str, Any]],
    *,
    start_date: date,
    end_date: date,
    stats: SearchStats,
) -> str:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for p in candidates:
        grouped[p["primary_tier"]][p.get("subcategory", "Other")].append(p)

    lines = [
        f"# LLM Data Literature Backfill: {start_date.isoformat()} → {end_date.isoformat()}",
        "",
        "> Automatically discovered candidate papers for manual review before inclusion in the main curated README.",
        "> The report is generated in **report-only mode**: README.md and data/papers.json are not modified.",
        "",
        "## Search Summary",
        "",
        f"- **Inclusive date range (arXiv submitted date, UTC):** {start_date.isoformat()} → {end_date.isoformat()}",
        f"- **Generated at:** {utc_now().isoformat()}",
        f"- **arXiv API requests:** {stats.query_requests}",
        f"- **Request failures:** {stats.query_failures}",
        f"- **Raw records returned:** {stats.raw_records}",
        f"- **Unique records aggregated:** {stats.unique_records}",
        f"- **Records inside date range:** {stats.in_date_range}",
        f"- **Passed category filter:** {stats.category_pass}",
        f"- **Passed LLM-context filter:** {stats.context_pass}",
        f"- **Passed relevance threshold:** {stats.relevance_pass}",
        f"- **Already present / duplicate records removed:** {stats.duplicates_removed}",
        f"- **Final new candidates:** {stats.final_candidates}",
        "",
        "## Review Checklist",
        "",
        "- [ ] Relevant to the LLM data lifecycle",
        "- [ ] Not already represented by another publication version in README",
        "- [ ] Primary tier is correct",
        "- [ ] Subcategory is correct",
        "- [ ] Cross-tier label is appropriate",
        "- [ ] Prefer formal venue / official project link when available",
        "",
    ]

    if not candidates:
        lines += ["## Candidates", "", "_No new candidates were found for this date range._", ""]
        return "\n".join(lines)

    tier_order = [
        "Data Substrates",
        "Data Creation and Selection",
        "Data Ingestion Strategies",
    ]
    section_num = 1
    for tier in tier_order:
        if tier not in grouped:
            continue
        lines += [f"## {section_num}. {tier}", ""]
        sub_num = 1
        for subcat, papers in sorted(grouped[tier].items(), key=lambda kv: kv[0].lower()):
            lines += [f"### {section_num}.{sub_num} {subcat}", ""]
            papers.sort(
                key=lambda p: parse_dt(p.get("published"))
                or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            for p in papers:
                cross = ", ".join(p.get("cross_tiers", [])) or "None"
                published = parse_dt(p.get("published"))
                pub_date = published.date().isoformat() if published else "n.d."
                authors = ", ".join(p.get("authors", [])[:5])
                if len(p.get("authors", [])) > 5:
                    authors += ", et al."
                lines += [
                    f"#### [{p['title']}]({p['url']})",
                    "",
                    f"- **Submitted:** {pub_date}",
                    f"- **Authors:** {authors or 'Unknown'}",
                    f"- **arXiv ID:** `{p.get('arxiv_id', '')}`",
                    f"- **Categories:** {', '.join(p.get('categories', [])) or 'n.d.'}",
                    f"- **Primary placement:** {p['primary_tier']} → {p.get('subcategory', 'Other')}",
                    f"- **Cross-tier:** {cross}",
                    f"- **Relevance score:** {p.get('relevance_score', 0)}",
                    f"- **Query families:** {', '.join(p.get('source_families', []))}",
                    f"- **Abstract:** {short_abstract(p.get('abstract', ''))}",
                    "",
                ]
            sub_num += 1
        section_num += 1

    lines += [
        "---",
        "",
        "Generated by `scripts/paper_bot.py` v2. This file is a discovery report, not an automatically accepted bibliography.",
        "",
    ]
    return "\n".join(lines)


def build_pr_body(
    new: list[dict[str, Any]], render: dict[str, Any], *, mode: str, report_path: str | None = None
) -> str:
    title = "Historical literature backfill" if mode == "report" else "Automated literature update"
    lines = [
        f"## {title}",
        "",
        f"Found **{len(new)}** new candidate paper(s) after filtering and deduplication.",
        "",
    ]
    if report_path:
        lines += [f"- **Generated report:** `{report_path}`", ""]
    lines += [
        "### Review checklist",
        "- [ ] Titles and links are correct",
        "- [ ] Papers are in scope for the LLM data lifecycle",
        "- [ ] Primary tier/subcategory placement is reasonable",
        "- [ ] Cross-tier labels are reasonable where shown",
        "- [ ] No duplicate publication versions are being added",
        "",
    ]
    if not new:
        return "\n".join(lines + ["_No new papers._"])
    for p in new[:50]:
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
            lines.append(f"- **Abstract:** {short_abstract(p.get('abstract', ''), n)}")
        lines.append("")
    return "\n".join(
        lines
        + [
            "---",
            "Generated by `scripts/paper_bot.py` v2. The bot proposes changes via PR; it does not auto-merge.",
        ]
    )


def resolve_range(args: argparse.Namespace, fetch_cfg: dict[str, Any]) -> tuple[date, date]:
    today = utc_now().date()
    if args.start_date or args.end_date:
        if not args.start_date:
            raise SystemExit("--start-date is required when --end-date is supplied")
        start = args.start_date
        end = args.end_date or today
        if end < start:
            raise SystemExit("--end-date cannot be earlier than --start-date")
        return start, end

    lookback = args.lookback_days or int(fetch_cfg.get("lookback_days", 14))
    if lookback < 1:
        raise SystemExit("--lookback-days must be >= 1")
    # Inclusive range: lookback=14 means today plus the previous 13 calendar days.
    return today - timedelta(days=lookback - 1), today


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Discover, classify and report LLM-data literature.",
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--readme", type=Path, required=True)
    ap.add_argument("--database", type=Path, required=True)
    ap.add_argument("--lookback-days", type=int, default=None)
    ap.add_argument("--start-date", type=parse_date_arg, default=None, help="Inclusive UTC date, YYYY-MM-DD")
    ap.add_argument("--end-date", type=parse_date_arg, default=None, help="Inclusive UTC date, YYYY-MM-DD; defaults to today")
    ap.add_argument("--report-only", action="store_true", help="Write standalone Markdown report; do not modify README/database")
    ap.add_argument("--output-report", type=Path, default=None, help="Required with --report-only")
    ap.add_argument(
        "--dedup-scope",
        choices=["readme", "readme+database", "none"],
        default="readme+database",
        help="Which existing records count as duplicates",
    )
    ap.add_argument("--pr-body", type=Path, default=Path("/tmp/paper_bot_pr_body.md"))
    ap.add_argument("--dry-run", action="store_true", help="Fetch/classify/print only; write no files")
    args = ap.parse_args()

    if args.report_only and args.output_report is None and not args.dry_run:
        raise SystemExit("--output-report is required with --report-only unless --dry-run is used")

    cfg = load_yaml(args.config)
    project = cfg["project"]
    fetch = cfg["fetch"]
    rel = cfg["relevance"]
    tax = cfg["taxonomy"]
    render = cfg.get("render", {})
    start_date, end_date = resolve_range(args, fetch)
    start_dt, end_dt = date_bounds(start_date, end_date)

    readme = args.readme.read_text(encoding="utf-8")
    db = load_database(args.database)

    known_titles: list[str] = []
    known_ids: set[str] = set()
    if args.dedup_scope in {"readme", "readme+database"}:
        known_titles.extend(read_readme_titles(readme))
        known_ids |= read_readme_arxiv_ids(readme)
    if args.dedup_scope == "readme+database":
        known_titles.extend(p.get("title", "") for p in db["papers"] if p.get("title"))
        known_ids |= {p.get("arxiv_id") for p in db["papers"] if p.get("arxiv_id")}

    stats = SearchStats()
    query_items = [(family, q) for family, qs in fetch["queries"].items() for q in qs]
    aggregated: dict[str, dict[str, Any]] = {}
    session = requests.Session()

    page_size = int(fetch.get("page_size", 100))
    max_pages = int(fetch.get("max_pages_per_query", 10))
    request_delay = float(fetch.get("request_delay_seconds", 3.2))

    print(f"Date range (inclusive, UTC): {start_date} -> {end_date}")
    print(f"Mode: {'report-only' if args.report_only else 'weekly/live'}")
    print(f"Dedup scope: {args.dedup_scope}")

    for i, (family, query) in enumerate(query_items):
        print(f"[query {i + 1}/{len(query_items)}] {family}: {query}")
        records = fetch_query_range(
            endpoint=fetch["endpoint"],
            query=query,
            start_dt=start_dt,
            end_dt=end_dt,
            page_size=page_size,
            max_pages=max_pages,
            timeout=int(fetch.get("timeout_seconds", 60)),
            user_agent=fetch["user_agent"],
            request_delay=request_delay,
            session=session,
            stats=stats,
        )
        for p in records:
            key = p.get("arxiv_id") or normalize_title(p["title"])
            if key not in aggregated:
                aggregated[key] = p
                aggregated[key]["source_families"] = set()
            aggregated[key]["source_families"].add(family)
        if i < len(query_items) - 1:
            time.sleep(request_delay)

    stats.unique_records = len(aggregated)
    candidates = classify_candidates(
        aggregated,
        start_dt=start_dt,
        end_dt=end_dt,
        known_ids=known_ids,
        known_titles=known_titles,
        fetch_cfg=fetch,
        relevance_cfg=rel,
        taxonomy_cfg=tax,
        stats=stats,
    )

    print("\nSearch statistics:")
    for key, value in asdict(stats).items():
        print(f"  {key}: {value}")
    print("\nCandidates:")
    for p in candidates:
        print(
            f"  [{p['relevance_score']:>2}] {p['primary_tier']} / {p['subcategory']} :: {p['title']}"
        )

    if args.dry_run:
        print("\nDry run: no files were written.")
        return 0

    if args.report_only:
        report = render_report(
            candidates, start_date=start_date, end_date=end_date, stats=stats
        )
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(report + "\n", encoding="utf-8")
        args.pr_body.parent.mkdir(parents=True, exist_ok=True)
        args.pr_body.write_text(
            build_pr_body(
                candidates,
                render,
                mode="report",
                report_path=str(args.output_report),
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote report: {args.output_report}")
        print("README.md and data/papers.json were not modified.")
        return 0

    # Weekly/live mode.
    if candidates:
        db["papers"].extend(candidates)
        db["schema_version"] = 2
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
        args.readme.write_text(
            replace_or_insert_block(readme, block, project), encoding="utf-8"
        )
        args.database.parent.mkdir(parents=True, exist_ok=True)
        args.database.write_text(
            json.dumps(db, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    args.pr_body.parent.mkdir(parents=True, exist_ok=True)
    args.pr_body.write_text(
        build_pr_body(candidates, render, mode="weekly") + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
