# Package contents

- `.github/workflows/update-papers.yml` — scheduled/manual GitHub Actions workflow
- `.github/workflows/backfill-papers.yml` — exact-date historical backfill workflow
- `.github/workflows/update-repo-status.yml` — PR + Actions history → README status block
- `scripts/paper_bot.py` — fetch/filter/deduplicate/classify/render pipeline
- `scripts/update_repo_status.py` — GitHub REST API → PR/run statistics → Markdown block (stdlib only)
- `scripts/render_paper_index.py` — paper database → complete grouped index (stdlib only)
- `PAPERS.md` — auto-generated full index of every discovered paper
- `config/paper_bot.yaml` — search queries, taxonomy keywords, thresholds
- `data/papers.json` — persistent bot-curated paper database
- `requirements.txt` — Python dependencies
- `SETUP.md` — installation, local testing, GitHub execution, tuning, troubleshooting
- `.gitignore.addition` — optional lines to append to your repository `.gitignore`
