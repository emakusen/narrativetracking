# Proposition tracker

Traces how **narratives change over time** in long documents (blog posts, newsletters, speeches, transcripts).
Instead of clustering short texts, the unit of analysis is a *proposition*:

    actor + action + object  (+ orientation + emotion)

A narrative is not a sequence of propositions with similar wording. It is an evolving story of an issue: 
- who is involved,
- what is happening,
- why it matters,
- what should happen next, and
- how the situation is interpreted.

The propositions are the observable building blocks. The narrative is a thread that connects the propositions over time.

## Pipeline

    raw.csv ──clean_input.py──▶ cleaned.csv ──proposition_tracker.py──▶ proposition_out/<name>/ ──analyze_corpus.py──▶ analysis_out/
             ISO dates, ids,                  propositions, narratives,                              period shares, themes,
             footers removed                  figures, viewer.html                                   bootstrap comparison

| Script | Purpose |
|---|---|
| `clean_input.py` | Parse dates with an explicit format (writes ISO), strip newsletter footers / repeated paragraphs, write `id` |
| `proposition_tracker.py` | Paragraphs → propositions (LLM, or spaCy alone with `--extractor spacy`) → merge within document → link across documents → narrative figures |
| `build_viewer.py` | Called by the tracker; builds `viewer.html` (self-contained), `narrative_trajectories.csv`, `narrative_summary.csv`. Flags unchanged chains as likely boilerplate |
| `analyze_corpus.py` | Corpus-level figures and tables per period (emotion, orientation, actors, keyword themes, bootstrap comparison) |
| `spacy_props.py` | Rule-based proposition extraction from the dependency parse (no LLM) |
| `emotions.py`, `llm_utils.py` | Shared emotion scheme; Ollama client and spaCy SVO helper |

## Install

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.9+
pip install -r requirements.txt
python -m spacy download en_core_web_sm                  # de_core_news_sm for --lang de
ollama pull gpt-oss:20b && ollama serve                  # local LLM; any Ollama model works via --llm-model
```

The LLM is optional: with `--extractor spacy` only spaCy is needed.

## Input format

A CSV (also `.json`, `.jsonl`, `.parquet`) with one row per document:

| column | required | |
|---|---|---|
| `full_text` | yes | paragraphs separated by blank lines |
| `date` | yes | **ISO (`2026-04-11`)**; run `clean_input.py` if yours look like `11.04.2026` |
| `title`, `id`, `source` | no | `id` is written by `clean_input.py`; `source` (blog / newsletter / ...) is used by `analyze_corpus.py --meta` |

> **Why `clean_input.py` matters.** Dates like `11.04.2026` are read month-first by pandas when the day is ≤ 12
> (→ 4 November). Newsletter footers turn into "propositions" that chain into perfect, meaningless narratives.
> Use `--report-repeats repeats.csv` first, look at what would be removed, then add `--drop-repeats`.

## Try it on the demo data

`demo/` gives a small **synthetic** corpus (15 documents of a fictional "Riverside Climate Collective" in English with German-style
dates and with a repeated footer to mimic newsletters). It contains one storyline: the council plans a tram extension,
postpones, cuts and cancels it, then reinstates it under pressure, with a stance and emotion trajectory that should flip.

```bash
# 1. clean (demo/demo_corpus_clean.csv is already the result)
python clean_input.py --input demo/demo_corpus_raw.csv --output cleaned.csv --date-format "%d.%m.%Y" \
    --report-repeats repeats.csv --drop-repeats --repeat-min-docs 4

# 2. track (needs Ollama running)
python proposition_tracker.py --input demo/demo_corpus_clean.csv --lang en --out demo_out \
    --context "a small local climate association"

# 3. analyze
python analyze_corpus.py --input-dir demo_out/demo_corpus_clean --out analysis_out \
    --meta demo/demo_corpus_clean.csv --config demo/analysis_config.json
```

Once done, open `demo_out/demo_corpus_clean/viewer.html` in a browser. To run without an LLM, add `--extractor spacy` to step 2. Whether the 
tram storyline comes out as one chain depends on how consistently your LLM words actor/action/object; with a 15-document corpus, expect to 
tune `--link-thr`. Without an LLM, `--no-llm` falls back to rule-based SVO only (orientation and emotion stay neutral).

## Choosing an extractor

| | `--extractor llm` (default) | `--extractor spacy` |
|---|---|---|
| actor / action / object | LLM, canonical English wording | dependency parse: subject, verb lemma, object, in the document's language |
| orientation, emotion | LLM label per paragraph | always `neutral` (cannot be read off the syntax) |
| claim | LLM one-sentence English paraphrase | the source sentence |
| narrative titles / summaries | LLM | generic ("actor: action object") |
| speed, cost | slow, needs Ollama | fast, no network |

The spaCy extractor produces one proposition per verb with a subject or object, keeps the `--max-props` best (both subject and
object found), and handles conjoined verbs, "wants to X", `it/they` resolved to the main-clause subject, negation
(`not extend`) and English passives (`cancelled by the council` → actor = council). It is also the fallback when
`--no-llm` is set or a single LLM call fails.

Limits: no co-reference beyond the pronoun rule, no stance or emotion, and wording is not normalized across synonyms
("scrap" and "cancel" stay different verbs; the embeddings still treat them as close). With `--lang de` the claim and
elements stay German, and the small model (`de_core_news_sm`) mis-tags some verbs; use `_md`, `_lg` or `_trf` for better results.
The `orientation`/`emotion` figures in `analyze_corpus.py` are meaningless for a spaCy-only run.

## Output of `proposition_tracker.py` (per dataset)

`paragraphs.csv`, `mentions.csv`, `propositions.csv`, `edges.csv`, `narratives.csv`, `narrative_XX.png` / `.json`,
`identical_chains.csv` (verbatim repeats that were skipped), `isolated_propositions.csv`, `config.json`, `viewer.html`,
`narrative_trajectories.csv` (every step measured against day 1), `narrative_summary.csv`. LLM answers are cached in
`<out>/llm_cache.json` (keyed by model + prompt, so changing `--context`, `--max-props` etc. re-queries the model).

## Settings 

| Option | Default | Effect |
|---|---|---|
| `--context` | "a social movement or advocacy organisation" | Who wrote the texts; inserted into the LLM prompts |
| `--merge-thr` | 0.80 | Same proposition in consecutive paragraphs of one document |
| `--link-thr` | 0.60 | Link propositions across documents (looser on purpose: transformation is allowed) |
| `--link-min-element-cos` | 0.50 | Every comparable element must reach this, so one matching element can't carry a link |
| `--max-gap` | 3 | Documents a narrative may skip |
| `--same-thr` / `--drift-thr` | 0.85 / 0.60 | Element cosine ≥ same = unchanged, ≥ drift = drifted, else changed |
| `--weights` | 0.25,0.25,0.5 | actor, action, object |

`analyze_corpus.py` has keyword rules for themes and actor classes built in (climate-movement vocabulary). Replace them
with `--config your.json` (see `demo/analysis_config.json`), and choose periods with `--periods year` (default) or
`--periods "2020,2021,2022-2023"`.

## Tests

```bash
pip install -r requirements-dev.txt && pytest
```

Unit tests plus an end-to-end test with a stub LLM and embedder (no models needed). They check the plumbing,
not the quality of real LLM output.

## Keep in mind

- With the LLM extractor, orientation and emotion are **one LLM label per paragraph**; validate on a hand-coded sample before drawing conclusions.
- The mapping of emotions to *encouraging / dampening / mixed* (`EMOTION_DIRECTION` in `emotions.py`) is the author's reading
  of Jasper's table, not part of it. It only decides what counts as an emotion "flip".
- Themes and actor classes in `analyze_corpus.py` are keyword rules and cover only part of the propositions.
- Bootstrap intervals are 95% percentile intervals over documents, not corrected for multiple comparisons.
- Element vectors are English: the LLM is asked to answer in English even for German sources.

## Reference

Jasper, J. M. (1998). The emotions of protest: Affective and reactive emotions in and around social movements.
*Sociological Forum, 13*(3), 397-424.
Grobelscheg, L., Kahr, E. and Strembeck, M. (2026). Exploring Narrative Continuity, Narrative Drift, and Narrative Gaps in Social Media. 
*The 13th International Conference on Social Networks Analysis, Management and Security (SNAMS 2026)*, Barcelona, Spain, 17.11.-20.11.2026. 
