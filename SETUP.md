# LLM Data Literature Bot — Setup & Operations

This package turns `Awesome-LLMs-Data-AI` into a living literature list.

## What it does

1. Runs weekly on GitHub Actions, or manually via `workflow_dispatch`.
2. Queries recent arXiv papers with topic-specific search families.
3. Filters by recency, arXiv category, LLM/data relevance, and duplicates.
4. Classifies candidates into your three-tier taxonomy:
   - Data Substrates
   - Data Creation and Selection
   - Data Ingestion Strategies
5. Updates a bounded `Recent Automatically Discovered Papers` block in `README.md`.
6. Creates a Pull Request for human review. It never auto-merges.

The bot uses the **operation rather than the paper as a whole** as the classification unit, so cross-tier labels are supported.

---

## 1. Copy files into the repo

Copy the package contents into the root of:

`raojay7/Awesome-LLMs-Data-AI`

Final structure:

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
└── .github/
    └── workflows/
        └── update-papers.yml
```

You do **not** need to add README markers manually. On the first successful run the bot inserts:

```md
<!-- AUTO-LITERATURE:START -->
## Recent Automatically Discovered Papers
...
<!-- AUTO-LITERATURE:END -->
```

before `## 6. Cross-Tier Synthesis`. Later runs only rewrite text between those markers.

---

## 2. One-time GitHub setting

Go to:

**Repository → Settings → Actions → General → Workflow permissions**

Make sure GitHub Actions is allowed to create pull requests. The workflow requests:

```yaml
permissions:
  contents: write
  pull-requests: write
```

No external API key is required in this version.

---

## 3. Recommended first local dry run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python scripts/paper_bot.py \
  --config config/paper_bot.yaml \
  --readme README.md \
  --database data/papers.json \
  --lookback-days 14 \
  --dry-run
```

`--dry-run` prints candidates and classifications but does not modify the README or database.

For a wider first scan:

```bash
python scripts/paper_bot.py \
  --config config/paper_bot.yaml \
  --readme README.md \
  --database data/papers.json \
  --lookback-days 30 \
  --dry-run
```

---

## 4. Test a real local update

Create a temporary branch:

```bash
git checkout -b test-literature-bot
```

Run:

```bash
python scripts/paper_bot.py \
  --config config/paper_bot.yaml \
  --readme README.md \
  --database data/papers.json \
  --lookback-days 14 \
  --pr-body /tmp/paper_bot_pr_body.md
```

Inspect:

```bash
git diff -- README.md data/papers.json
cat /tmp/paper_bot_pr_body.md
```

If this was only a test:

```bash
git restore README.md data/papers.json
git checkout -
git branch -D test-literature-bot
```

---

## 5. Commit the bot files

```bash
git add .github/workflows/update-papers.yml \
        scripts/paper_bot.py \
        config/paper_bot.yaml \
        data/papers.json \
        requirements.txt

git commit -m "feat: add automatic LLM data literature bot"
git push
```

The workflow must exist on the repository default branch before it can be started manually from the GitHub Actions UI.

---

## 6. Run manually on GitHub

Open:

**Repository → Actions → Weekly LLM Data Literature Bot → Run workflow**

Use:

- `7` days: weekly catch-up
- `14` days: recommended default
- `30` days: first run / after a pause

The pipeline is:

```text
arXiv
  ↓
query families
  ↓
date/category filter
  ↓
LLM/data relevance score
  ↓
README + papers.json deduplication
  ↓
three-tier taxonomy classification
  ↓
README auto section
  ↓
bot branch
  ↓
Pull Request
```

Review the PR and merge manually.

---

## 7. Scheduled execution

Default schedule in `update-papers.yml`:

```yaml
- cron: "17 10 * * 1"
  timezone: "Asia/Shanghai"
```

This runs every Monday at 10:17 Asia/Shanghai.

Every day at 09:17:

```yaml
- cron: "17 9 * * *"
  timezone: "Asia/Shanghai"
```

Monday + Thursday at 10:17:

```yaml
- cron: "17 10 * * 1,4"
  timezone: "Asia/Shanghai"
```

---

## 8. Tune topic coverage

Edit `config/paper_bot.yaml`.

### Add a query

Example for agent memory / experience data:

```yaml
queries:
  substrate:
    - '(all:"LLM agent" OR all:"language model agent") AND (all:"memory" OR all:"experience trajectory")'
```

### Add taxonomy keywords

Example:

```yaml
"Agent and Tool Use":
  - "agent trajectory"
  - "executable environment"
  - "agent memory"
```

### Make filtering stricter

Raise:

```yaml
min_score: 6
```

to `8` or `9` if PRs are noisy.

If useful papers are missed, add more precise search/positive terms before simply lowering the score threshold.

---

## 9. How classification works

For relevance:

```text
query-family agreement
+ title keyword matches (higher weight)
+ abstract keyword matches
- negative keywords
= relevance score
```

For taxonomy placement:

```text
query-family prior
+ tier/subcategory keyword matches
= tier score
```

A candidate may look like:

```json
{
  "primary_tier": "Data Creation and Selection",
  "subcategory": "Synthetic Data",
  "cross_tiers": ["Data Substrates"],
  "tier_scores": {
    "Data Substrates": 8,
    "Data Creation and Selection": 11,
    "Data Ingestion Strategies": 2
  }
}
```

The PR body exposes these scores so ambiguous placements are quick to review.

---

## 10. Deduplication behavior

After a PR is merged, papers are stored in `data/papers.json` and rendered in the README auto section.

Future runs deduplicate against:

1. arXiv IDs in the current README;
2. arXiv IDs in `data/papers.json`;
3. titles in the current README;
4. fuzzy title similarity for near-duplicate versions.

This catches many arXiv/conference duplicate cases, but publication-version review should remain part of the PR check.

---

## 11. Recommended curation policy

Do not treat the auto section as the permanent final taxonomy database.

Recommended operating loop:

```text
Bot discovers papers
   ↓
Auto section + PR
   ↓
Human review
   ↓
Merge
   ↓
Periodically move important papers
into the main curated sections
```

This keeps discovery fast without turning the main list into a noisy feed.

---

## 12. Troubleshooting

### Workflow cannot create a PR

Check:

**Settings → Actions → General → Workflow permissions**

and ensure Actions can create pull requests.

### arXiv temporarily returns an HTTP error

The script has a custom User-Agent, sleeps 3.2 seconds between requests, and treats a single failed query as non-fatal. Rerun the workflow later if arXiv is unavailable.

### Too many irrelevant papers

Increase `relevance.min_score` and tighten queries.

### Too few papers

Increase `max_results_per_query`, extend `lookback_days`, or add topic-specific queries/keywords.

---

## 13. Recommended next upgrade

After this deterministic baseline has run stably for several weeks, add an **optional LLM judge only after the rule-based high-recall stage**:

```text
arXiv retrieval
  ↓
rule-based high-recall filter
  ↓
optional LLM relevance/taxonomy judge
  ↓
PR
```

Keep human PR review and avoid auto-merge.
