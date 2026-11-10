#!/usr/bin/env python3
"""Refresh the AUTO-STATUS block in README.md from GitHub PRs and Actions history.

Standard library only, so it runs both locally and on a stock GitHub Actions runner.

Local (public repository, unauthenticated):
    python scripts/update_repo_status.py --readme README.md --dry-run

GitHub Actions:
    GITHUB_TOKEN and GITHUB_REPOSITORY are provided by the workflow.

The script only rewrites the region between AUTO-STATUS:START and
AUTO-STATUS:END, leaving the curated part of README.md untouched. It exits
non-zero if the GitHub API cannot be read, so a failed run never produces a
half-empty status table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API_ROOT = "https://api.github.com"
DEFAULT_REPO = "raojay7/Awesome-LLMs-Data-AI"
MARKER_START = "<!-- AUTO-STATUS:START -->"
MARKER_END = "<!-- AUTO-STATUS:END -->"
INSERT_BEFORE = "### Curation policy"
LITERATURE_DATE_RE = re.compile(r"Last generated:\s*\*\*(\d{4}-\d{2}-\d{2})\*\*")


def api_get(path: str, token: str | None, params: dict | None = None, page: int = 1) -> object:
    query = dict(params or {})
    query["per_page"] = 100
    query["page"] = page
    url = f"{API_ROOT}{path}?{urllib.parse.urlencode(query)}"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "awesome-llms-data-ai-status-bot",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def api_all_pages(
    path: str, token: str | None, params: dict | None = None, max_pages: int = 5, key: str | None = None
) -> list:
    """Collect paginated endpoints (GitHub caps per_page at 100).

    Endpoints such as /actions/runs wrap their list in an object, e.g.
    {"total_count": 12, "workflow_runs": [...]}. `key` selects that field.
    """
    items: list = []
    for page in range(1, max_pages + 1):
        payload = api_get(path, token, params, page)
        batch = payload.get(key, []) if key else payload
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected response for {path}: expected a list at {key or 'root'}")
        items.extend(batch)
        if len(batch) < 100:
            break
    return items


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def fmt_dt(value: str | None) -> str:
    dt = parse_dt(value)
    return dt.strftime("%Y-%m-%d %H:%M UTC") if dt else "-"


def fmt_date(value: str | None) -> str:
    dt = parse_dt(value)
    return dt.strftime("%Y-%m-%d") if dt else "-"


def age_days(value: str | None, now: datetime) -> str:
    dt = parse_dt(value)
    if not dt:
        return "-"
    return f"{(now - dt).days} d"


def md_cell(text: str, limit: int = 88) -> str:
    cleaned = " ".join(str(text or "").split())
    cleaned = cleaned.replace("|", "\\|").replace("\r", " ")
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def is_bot_pr(pr: dict) -> bool:
    login = (pr.get("user") or {}).get("login", "")
    head_ref = (pr.get("head") or {}).get("ref", "")
    return login.endswith("[bot]") or head_ref.startswith("bot/")


def collect(repo: str, token: str | None, max_runs: int, max_prs: int) -> dict:
    pulls = api_all_pages(f"/repos/{repo}/pulls", token, {"state": "all", "sort": "updated", "direction": "desc"})
    runs = api_all_pages(f"/repos/{repo}/actions/runs", token, max_pages=3, key="workflow_runs")
    workflows_resp = api_get(f"/repos/{repo}/actions/workflows", token)
    workflows = workflows_resp.get("workflows", []) if isinstance(workflows_resp, dict) else []

    now = datetime.now(timezone.utc)
    open_prs = [p for p in pulls if p.get("state") == "open"]
    open_prs.sort(key=lambda p: p.get("number", 0), reverse=True)
    merged_prs = [p for p in pulls if p.get("merged_at")]
    bot_open = [p for p in open_prs if is_bot_pr(p)]
    completed = [r for r in runs if r.get("status") == "completed"]
    succeeded = [r for r in completed if r.get("conclusion") == "success"]
    latest = runs[0] if runs else {}
    oldest_open = min(open_prs, key=lambda p: p.get("created_at") or "") if open_prs else None

    return {
        "now": now,
        "repo": repo,
        "pulls": pulls,
        "open_prs": open_prs[:max_prs],
        "open_prs_total": len(open_prs),
        "merged_total": len(merged_prs),
        "bot_open_total": len(bot_open),
        "human_open_total": len(open_prs) - len(bot_open),
        "oldest_open": oldest_open,
        "runs": runs,
        "recent_runs": runs[:max_runs],
        "completed_total": len(completed),
        "succeeded_total": len(succeeded),
        "latest": latest,
        "workflows": workflows,
    }


def render(data: dict, readme: str, papers_count: int | None) -> str:
    now: datetime = data["now"]
    repo = data["repo"]
    lines = [
        MARKER_START,
        "### Repository Automation Status",
        "",
        "> Auto-generated by `update-repo-status.yml` from the repository's pull requests and GitHub Actions run history.",
        f"> Last refreshed: **{now.strftime('%Y-%m-%d %H:%M UTC')}**. This block is regenerated automatically — do not edit by hand.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Workflow runs recorded | {len(data['runs'])} |",
        f"| Successful / completed runs | {data['succeeded_total']} / {data['completed_total']}"
        f" ({100.0 * data['succeeded_total'] / data['completed_total']:.1f}%) |"
        if data["completed_total"]
        else "| Successful / completed runs | — |",
        f"| Latest run | {md_cell(data['latest'].get('name', '-'), 40)} · "
        f"{data['latest'].get('conclusion') or data['latest'].get('status') or '-'} · "
        f"{fmt_dt(data['latest'].get('created_at'))} |",
        f"| Registered workflows | {len(data['workflows'])} "
        f"({sum(1 for w in data['workflows'] if w.get('state') == 'active')} active) |",
        f"| Pull requests (all / open / merged) | {len(data['pulls'])} / "
        f"{data['open_prs_total']} / {data['merged_total']} |",
        f"| Open PRs (bot / human) | {data['bot_open_total']} / {data['human_open_total']} |",
        f"| Oldest open PR | "
        + (
            f"[#{data['oldest_open']['number']}]({data['oldest_open'].get('html_url', '#')}) · "
            f"{fmt_date(data['oldest_open'].get('created_at'))} · "
            f"{age_days(data['oldest_open'].get('created_at'), now)} ago |"
            if data["oldest_open"]
            else "— |"
        ),
    ]

    lit_match = LITERATURE_DATE_RE.search(readme)
    lines.append(f"| Recent-paper block last generated | {lit_match.group(1) if lit_match else '—'} |")
    if papers_count is not None:
        lines.append(f"| Records in `data/papers.json` | {papers_count:,} |")

    lines += [
        "",
        "#### Recent workflow runs",
        "",
        "| Workflow | Trigger | Result | Started (UTC) |",
        "| --- | --- | --- | --- |",
    ]
    for run in data["recent_runs"]:
        lines.append(
            f"| {md_cell(run.get('name', '-'), 40)} | {md_cell(run.get('event', '-'), 18)} | "
            f"{run.get('conclusion') or run.get('status') or '-'} | {fmt_dt(run.get('created_at'))} |"
        )
    if not data["recent_runs"]:
        lines.append("| _No workflow runs yet._ | — | — | — |")

    lines += [
        "",
        "#### Open pull requests",
        "",
        "| PR | Title | Author | Opened | Age |",
        "| --- | --- | --- | --- | --- |",
    ]
    for pr in data["open_prs"]:
        number = pr.get("number", "-")
        link = f"[#{number}](https://github.com/{repo}/pull/{number})"
        lines.append(
            f"| {link} | {md_cell(pr.get('title', '-'))} | "
            f"`{md_cell((pr.get('user') or {}).get('login', '-'), 30)}` | "
            f"{fmt_date(pr.get('created_at'))} | {age_days(pr.get('created_at'), now)} |"
        )
    if not data["open_prs"]:
        lines.append("| _No open pull requests._ | — | — | — | — |")
    if data["open_prs_total"] > len(data["open_prs"]):
        lines.append("")
        lines.append(f"_Showing {len(data['open_prs'])} of {data['open_prs_total']} open pull requests._")

    if data["bot_open_total"]:
        lines += [
            "",
            f"> **Review backlog:** {data['bot_open_total']} bot-generated literature PRs are awaiting review. "
            "Each weekly PR is a full snapshot, so the newest one carries the latest state of `README.md` "
            "and `data/papers.json`; older snapshots can be closed once the newest is merged.",
        ]

    lines += [
        "",
        "#### Registered workflows",
        "",
        "| Workflow | State | Definition |",
        "| --- | --- | --- |",
    ]
    for wf in data["workflows"]:
        lines.append(
            f"| {md_cell(wf.get('name', '-'), 40)} | {wf.get('state', '-')} | "
            f"`{md_cell(wf.get('path', '-'), 60)}` |"
        )
    if not data["workflows"]:
        lines.append("| _No workflows registered._ | — | — |")

    lines += ["", MARKER_END]
    return "\n".join(lines)


def replace_or_insert(readme: str, block: str) -> str:
    pattern = re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END)
    if re.search(pattern, readme, flags=re.S):
        return re.sub(pattern, lambda _m: block, readme, count=1, flags=re.S)
    if INSERT_BEFORE in readme:
        return readme.replace(INSERT_BEFORE, block + "\n\n" + INSERT_BEFORE, 1)
    return readme.rstrip() + "\n\n" + block + "\n"


def without_refresh_stamp(text: str) -> str:
    """Ignore the refresh timestamp when deciding whether a commit is needed."""
    return re.sub(r"> Last refreshed: \*\*[^*]+\*\*", "> Last refreshed: **<stamp>**", text)


def count_papers(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(payload, dict) and isinstance(payload.get("papers"), list):
        return len(payload["papers"])
    if isinstance(payload, list):
        return len(payload)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--database", type=Path, default=Path("data/papers.json"))
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO)
    parser.add_argument("--max-runs", type=int, default=5)
    parser.add_argument("--max-prs", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="print the block without writing")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or None
    try:
        data = collect(args.repo, token, args.max_runs, args.max_prs)
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, ValueError) as exc:
        print(f"Failed to read GitHub data for {args.repo}: {exc}", file=sys.stderr)
        return 1

    readme = args.readme.read_text(encoding="utf-8")
    block = render(data, readme, count_papers(args.database))
    updated = replace_or_insert(readme, block)

    if args.dry_run:
        print(block)
        print(f"\n[dry-run] {len(data['runs'])} runs, {len(data['pulls'])} PRs; README unchanged.", file=sys.stderr)
        return 0

    if updated == readme:
        print("Status block already up to date; no changes written.")
        return 0

    if without_refresh_stamp(updated) == without_refresh_stamp(readme):
        print("Only the refresh timestamp changed; skipping the write to avoid empty commits.")
        return 0

    args.readme.write_text(updated, encoding="utf-8")
    print(
        f"Updated status block in {args.readme}: "
        f"{len(data['runs'])} runs, {data['open_prs_total']} open PRs, "
        f"{data['succeeded_total']}/{data['completed_total']} successful runs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
