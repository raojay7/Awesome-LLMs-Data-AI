#!/usr/bin/env python3
"""Attach a one-sentence idea to every paper in data/papers.json.

The idea is extracted deterministically from the arXiv abstract: preference is
given to the sentence that states the contribution ("We introduce/propose ...",
"This paper presents ..."), with background sentences deprioritised. Standard
library only, so `update-repo-status.yml` can keep the field complete as new
papers arrive.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FIELD = "idea"
SOURCE_FIELD = "idea_source"
IDEA_SOURCE = "abstract-contribution-sentence"

# Split on sentence boundaries while keeping common abbreviations intact.
ABBREV = re.compile(r"\b(e\.g|i\.e|et al|vs|cf|Dr|Prof|Fig|Eq|Sec|No|approx)\.\s", re.I)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z$\"(\[])")
CONTRIBUTION_VERB = re.compile(
    r"\b(introduce|introduces|propose|proposes|present|presents|develop|develops|"
    r"design|designs|build|builds|release|releases|construct|constructs|"
    r"demonstrate|demonstrates|show|shows|find|finds|argue|argues|"
    r"investigate|investigates|evaluate|evaluates|provide|provides|"
    r"offer|offers|explore|explores|formulate|formulates|establish|establishes|"
    r"identify|identifies|reveal|reveals|derive|derives|create|creates|"
    r"study|studies|address|addresses|tackle|tackles|leverage|leverages)\b",
    re.I,
)
FIRST_PERSON = re.compile(r"\b(we|our|here)\b", re.I)
PAPER_SUBJECT = re.compile(r"\bthis (paper|work|study|article|report|research)\b", re.I)
BACKGROUND = re.compile(
    r"\b(recent|recently|however|despite|although|existing|prior|previous|"
    r"has become|increasingly|has attracted|long-standing|widely|traditionally|"
    r"in this context|background|motivated by)\b",
    re.I,
)
LATEX = re.compile(r"\\[a-zA-Z]+\{?|\$|\{|\}|~|\\")


def clean(text: str) -> str:
    text = text.replace("\n", " ")
    text = LATEX.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip(" -")


def sentences(abstract: str) -> list[str]:
    protected = ABBREV.sub(lambda m: m.group(0).replace(". ", "<DOT> "), abstract)
    parts = SENTENCE_SPLIT.split(protected)
    return [clean(p.replace("<DOT>", ".")) for p in parts if len(clean(p)) > 25]


def score(sentence: str, position: int) -> float:
    value = 0.0
    if CONTRIBUTION_VERB.search(sentence):
        value += 3.0
    if FIRST_PERSON.search(sentence) and CONTRIBUTION_VERB.search(sentence):
        value += 2.5
    if PAPER_SUBJECT.search(sentence):
        value += 2.0
    if BACKGROUND.search(sentence):
        value -= 2.0
    n = len(sentence)
    if 70 <= n <= 320:
        value += 1.0
    elif n < 50 or n > 450:
        value -= 1.0
    value -= min(position, 4) * 0.35
    return value


def one_liner(abstract: str, max_chars: int = 240) -> str:
    candidates = sentences(abstract)
    if not candidates:
        return ""
    best = max(range(len(candidates)), key=lambda i: score(candidates[i], i))
    chosen = candidates[best]
    if len(chosen) > max_chars:
        cut = chosen[:max_chars].rsplit(" ", 1)[0].rstrip(",;:")
        chosen = cut + "…"
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/papers.json"))
    parser.add_argument("--sample", type=int, default=0, help="print N examples and do not write")
    parser.add_argument("--force", action="store_true", help="recompute ideas already present")
    args = parser.parse_args()

    db = json.loads(args.database.read_text(encoding="utf-8"))
    papers = db.get("papers", [])
    added = 0
    samples: list[tuple[str, str]] = []
    for p in papers:
        if not args.force and p.get(FIELD):
            continue
        idea = one_liner(p.get("abstract") or "")
        if not idea:
            continue
        p[FIELD] = idea
        p[SOURCE_FIELD] = IDEA_SOURCE
        added += 1
        if len(samples) < args.sample:
            samples.append((p.get("title", ""), idea))

    if args.sample:
        for title, idea in samples:
            print(f"- {title}\n  => {idea}\n")
        print(f"[sample] {len(samples)} shown; {added} ideas would be added.")
        return 0

    if added == 0:
        print(f"All {len(papers):,} papers already have an idea; nothing changed.")
        return 0
    args.database.write_text(json.dumps(db, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Added {added:,} one-sentence ideas to {args.database} ({len(papers):,} papers total).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
