# LLM Data Literature Bot v2 — Deployment & Usage

v2 keeps the original weekly updater and adds an **exact-date historical backfill workflow** that produces a standalone Markdown report without touching the curated README.

## What changed in v2

- `--start-date YYYY-MM-DD` and `--end-date YYYY-MM-DD` for exact inclusive date ranges.
- `--report-only` for standalone discovery reports.
- `--output-report updates/<range>.md` for a new file rather than a README edit.
- Paginated arXiv retrieval, so a multi-week backfill is not limited to only the newest N results of each query.
- `--dedup-scope readme|readme+database|none`.
- Search statistics in the generated report.
- A separate `.github/workflows/backfill-papers.yml` workflow.
- The original weekly workflow remains available.

## Repository layout

```text
Awesome-LLMs-Data-AI/
├── README.md
├── requirements.txt
├── config/
│   ├── paper_bot.yaml
│   └── quality_ranking.yaml           # quality proxy rules
├── data/
│   └── papers.json
├── scripts/
│   ├── paper_bot.py
│   ├── update_repo_status.py         # stdlib-only PR/Actions status generator
│   ├── render_paper_index.py         # paper database → PAPERS.md index
│   ├── extract_ideas.py              # abstract → one-sentence idea
│   └── rank_papers.py                # quality proxy → ranked report
├── updates/                         # created automatically by backfill
└── .github/
    └── workflows/
        ├── update-papers.yml        # weekly/live mode
        ├── backfill-papers.yml      # exact historical range → new Markdown file
        └── update-repo-status.yml   # PR + Actions history → README status block
```

## Repository status sync (stdlib only)

`update-repo-status.yml` refreshes the `AUTO-STATUS` block in `README.md` from the
GitHub PR list and Actions run history. It runs weekly, after the literature bot
finishes, when PRs open/close, and on manual dispatch. Only the marked block is
rewritten, and the generated commit is skipped by CI triggers.

To preview locally:

```bash
python scripts/update_repo_status.py --readme README.md --dry-run
python scripts/update_repo_status.py --readme README.md   # writes the block
```

Unauthenticated requests work for public repositories (60 requests/hour); set
`GITHUB_TOKEN` for private repositories or higher limits.

## Complete paper index (`PAPERS.md`)

`data/papers.json` is the bot's persistent discovery database, but the README's
recent-paper block only shows the newest `max_readme_papers` entries (currently
60). `PAPERS.md` lists **every** unique paper in the database grouped by primary
tier and subcategory, together with a month-by-month breakdown. It is rebuilt
by `update-repo-status.yml` whenever the status block is refreshed:

```bash
python scripts/render_paper_index.py --database data/papers.json --output PAPERS.md
```

If weekly literature PRs pile up unmerged, their branches still contain the
papers discovered that week. Merge those branches (or merge their
`data/papers.json` files, deduplicated by arXiv ID) so the discoveries are
not lost — the bot otherwise starts every run from the stale main-branch
database and rediscovers the same papers.

## One-sentence ideas

`scripts/extract_ideas.py` deterministically selects the contribution sentence
("We introduce ...", "This work proposes ...") from each arXiv abstract and
stores it as the `idea` field of `data/papers.json`; `PAPERS.md` renders it under
every entry. `update-repo-status.yml` fills the field for newly discovered
papers on each run. This is extraction, not a model-generated summary.

`PAPER_IDEAS_ZH.md` is the human-written layer: a systematic review of the
corpus (tier/subcategory distribution, title-term hotspots, four research
threads) plus Chinese one-sentence summaries for the 60 highest-relevance
papers. Extend it manually when the curation scope grows.

## Quality ranking rules

`relevance_score` comes from keyword rules and measures **topical fit**; it says
nothing about how much evidence a paper reports. `config/quality_ranking.yaml`
defines a separate, fully auditable **quality proxy** (0–100):

```
score = evidence(30) + reproducibility(25) + maturity(15)
      + fit(20) + clarity(10) - penalty
```

Each dimension is a sum of boolean signals — numeric results, baselines,
ablations, code links, open-source statements, benchmarks/datasets, DOI or
journal reference, revisions, named methods, abstract length, plus a bounded
`relevance_norm`. Signals are regexes in the config; the exact hits for each
paper are stored in `quality_evidence` inside `data/papers.json`, so any score
can be traced. Grades A/B/C/D use absolute thresholds calibrated on this corpus.

```bash
python scripts/rank_papers.py --sample 5          # inspect without writing
python scripts/rank_papers.py                    # score new papers + rebuild report
python scripts/rank_papers.py --force            # recompute everything
python scripts/render_paper_index.py --database data/papers.json --output PAPERS.md
```

`PAPERS.md` orders papers by this score within every subcategory, and
`PAPER_QUALITY_RANKING.md` contains the rules, distribution and global Top 100.
To change the weights or thresholds, edit the YAML and **bump its `version`** —
the workflow then invalidates all stored scores. Locally, `--force` does the
same. `update-repo-status.yml` runs `rank_papers.py` on every refresh (it
installs PyYAML for the rule file).

This is a proxy, not a quality judgement: there are no citation counts, review
scores or full texts. A polished abstract can score well regardless of
correctness, and mentioning a benchmark is not the same as releasing one.

## 1. Upgrade from v1

Copy/overwrite these files in the repo root:

```text
.github/workflows/update-papers.yml
.github/workflows/backfill-papers.yml
scripts/paper_bot.py
config/paper_bot.yaml
requirements.txt
```

If your existing `data/papers.json` already contains records, **keep it**. You do not need to replace it with the empty v2 example. The script accepts the existing structure and will write `schema_version: 2` on the next live update.

Then commit:

```bash
git add .github/workflows/update-papers.yml \
        .github/workflows/backfill-papers.yml \
        scripts/paper_bot.py \
        config/paper_bot.yaml \
        requirements.txt

git commit -m "feat: upgrade literature bot to v2"
git push
```

## 2. GitHub permission (one-time)

Repository → **Settings → Actions → General → Workflow permissions**.

Allow GitHub Actions to create pull requests. The workflows request:

```yaml
permissions:
  contents: write
  pull-requests: write
```

## 3. Exact local backfill: 2026-07-01 → 2026-08-17

First install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Preview only — no file writes

```bash
python scripts/paper_bot.py \
  --config config/paper_bot.yaml \
  --readme README.md \
  --database data/papers.json \
  --start-date 2026-07-01 \
  --end-date 2026-08-17 \
  --dedup-scope readme \
  --dry-run
```

### Generate the standalone new file

```bash
python scripts/paper_bot.py \
  --config config/paper_bot.yaml \
  --readme README.md \
  --database data/papers.json \
  --start-date 2026-07-01 \
  --end-date 2026-08-17 \
  --report-only \
  --output-report updates/2026-07-01_to_2026-08-17.md \
  --dedup-scope readme \
  --pr-body /tmp/paper_bot_pr_body.md
```

Result:

```text
updates/2026-07-01_to_2026-08-17.md
```

In report-only mode:

- `README.md` is unchanged.
- `data/papers.json` is unchanged.
- Only the requested Markdown report and PR-body file are written.

Inspect it:

```bash
cat updates/2026-07-01_to_2026-08-17.md
# or
git diff -- updates/2026-07-01_to_2026-08-17.md
```

## 4. Generate “July 1 to today” automatically

Omit `--end-date`; v2 uses the current UTC date:

```bash
python scripts/paper_bot.py \
  --config config/paper_bot.yaml \
  --readme README.md \
  --database data/papers.json \
  --start-date 2026-07-01 \
  --report-only \
  --output-report updates/2026-07-01_to_today.md \
  --dedup-scope readme
```

For reproducible archival filenames, explicit `--end-date` is preferable.

## 5. Run the historical backfill from GitHub UI

After pushing `backfill-papers.yml` to the default branch:

1. Open the repository.
2. Go to **Actions**.
3. Select **Historical LLM Data Literature Backfill**.
4. Click **Run workflow**.
5. Enter:
   - `start_date`: `2026-07-01`
   - `end_date`: `2026-08-17` (or leave blank to use the current UTC date)
6. Run it.

The workflow creates only:

```text
updates/2026-07-01_to_2026-08-17.md
```

on a bot branch and opens a PR such as:

```text
📚 Literature Backfill: 2026-07-01 → 2026-08-17
```

Review the file, then merge if the candidate list looks useful.

## 6. Why backfill uses `--dedup-scope readme`

For a historical report, the question is usually:

> “Which papers from this date range are not already in my current curated README?”

Therefore the GitHub backfill workflow ignores the bot's internal `data/papers.json` when deciding whether a paper is new. This avoids hiding a paper merely because the bot saw it previously but it was never added to the curated README.

Options:

```text
readme            deduplicate only against current README
readme+database   deduplicate against README and bot database (weekly default)
none              do not deduplicate against existing repo content
```

## 7. Weekly mode still works

Dry run:

```bash
python scripts/paper_bot.py \
  --config config/paper_bot.yaml \
  --readme README.md \
  --database data/papers.json \
  --lookback-days 14 \
  --dry-run
```

Real local update:

```bash
python scripts/paper_bot.py \
  --config config/paper_bot.yaml \
  --readme README.md \
  --database data/papers.json \
  --lookback-days 14 \
  --pr-body /tmp/paper_bot_pr_body.md
```

This mode may update `README.md` and `data/papers.json` when candidates exist.

The scheduled GitHub workflow runs every Monday at 10:17 in `Asia/Shanghai`.

## 8. Important v2 retrieval change

v1 requested only one fixed-size page per query. That is acceptable for a short weekly scan but can miss older records during a long historical range.

v2 uses:

```yaml
page_size: 100
max_pages_per_query: 10
```

and requests pages in descending submitted-date order until it reaches a page containing papers older than the requested `start_date`.

For a busy or very broad query, increase:

```yaml
max_pages_per_query: 15
```

The trade-off is a longer Action run and more arXiv API requests.

## 9. If the report is too noisy

In `config/paper_bot.yaml` increase:

```yaml
relevance:
  min_score: 6
```

to, for example:

```yaml
relevance:
  min_score: 9
```

Prefer refining queries/keywords before making the threshold excessively high.

## 10. If the report is too small

Try:

- adding query phrases under `fetch.queries`;
- increasing `max_pages_per_query`;
- reducing `min_score` slightly;
- adding a missing keyword to the appropriate taxonomy subcategory.

## 11. Recommended operating pattern

```text
Historical backfill
    ↓
standalone updates/<date-range>.md
    ↓
human review
    ↓
move important items into curated README sections

Weekly bot
    ↓
recent auto block + papers.json
    ↓
PR review
    ↓
merge
```

The generated historical file is a **candidate discovery report**, not a claim that every paper should be included in the survey.
