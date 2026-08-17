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
│   └── paper_bot.yaml
├── data/
│   └── papers.json
├── scripts/
│   └── paper_bot.py
├── updates/                         # created automatically by backfill
└── .github/
    └── workflows/
        ├── update-papers.yml        # weekly/live mode
        └── backfill-papers.yml      # exact historical range → new Markdown file
```

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
