#!/usr/bin/env python3
"""
clean_input.py - prepare the merged corpus for proposition_tracker.py

Fixes the problems found in the first run:
  1. DATES. proposition_tracker.py parses `date` with pandas format="mixed" and no day-first
     setting. German-style dates such as 11.04.2026 with a day <= 12 are then read month-first
     (-> 4 November). This script parses the dates with an explicit format you give it and
     writes them as ISO (YYYY-MM-DD), which cannot be misread.
  2. BOILERPLATE. Removes newsletter footers (consent / data-protection text, etc.) with
     regular expressions you provide, before the pipeline turns them into "propositions".
  3. REPEATED TEXT. Detects paragraphs (or sentences) that occur in many documents (newsletter footers,
     "printed on recyclable pixels", donation appeals, imprint) and can report or remove them.
  4. IDS. Writes an `id` column (doc001, doc002, ...) in date order if there is none, so that
     the doc_ids in the pipeline output can later be joined to titles / source types.

It also prints warnings for dates in the future and for many documents on the same date
(usually a scrape or upload date, not a publication date).

Finding repeated text (do this BEFORE the pipeline: the LLM turns every footer into "propositions"
and the linker chains them into perfect, meaningless narratives):

  # 1. report only: look at repeats.csv, check what is boilerplate and what is a real recurring message
  python clean_input.py --input raw.csv --output cleaned.csv --date-format "%d.%m.%Y" --report-repeats repeats.csv
  # 2. remove everything that occurs in >= 4 documents (raise --repeat-min-docs if real slogans are caught)
  python clean_input.py ... --drop-repeats --repeat-min-docs 4
  # footer glued to the last paragraph instead of standing alone? use --repeat-unit sentence

Examples
--------
  # raw dates look like 11.04.2026
  python clean_input.py --input raw.csv --output cleaned.csv --date-format "%d.%m.%Y" \
      --remove-regex "Ich stimme der Datenverarbeitung.*?(?=\\n\\n|$)"

  # raw dates are already ISO (2026-04-11)
  python clean_input.py --input raw.csv --output cleaned.csv --date-format "%Y-%m-%d"

Keep any extra columns you have (e.g. `source` = blog / newsletter / press / youtube, `title`):
they are passed through unchanged.
"""
import argparse
import re
import sys

import pandas as pd


def norm_key(u):
    """Comparison key: lower case, digits -> #, punctuation removed, first 100 characters."""
    k = re.sub(r"\d+", "#", u.lower())
    return re.sub(r"[^\w#]+", " ", k).strip()[:100]


def paragraphs(text):
    return [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]


def units(text, unit):
    out = []
    for b in paragraphs(text):
        out += [b] if unit == "paragraph" else [x for x in re.split(r"(?<=[.!?])\s+", b) if x.strip()]
    return out


def strip_units(text, repeated, unit):
    keep = []
    for b in paragraphs(text):
        if unit == "paragraph":
            if norm_key(b) not in repeated:
                keep.append(b)
        else:
            kept = [x for x in re.split(r"(?<=[.!?])\s+", b) if norm_key(x) not in repeated]
            if kept:
                keep.append(" ".join(kept))
    return "\n\n".join(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="raw CSV (needs a text and a date column)")
    ap.add_argument("--output", required=True, help="cleaned CSV to write")
    ap.add_argument("--text-col", default="full_text")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--date-format", required=True,
                    help='strptime format of the RAW dates, e.g. "%%d.%%m.%%Y" or "%%Y-%%m-%%d"')
    ap.add_argument("--remove-regex", action="append", default=[],
                    help="regex to delete from every text (repeatable; case-insensitive, . matches newlines "
                         "only if you add (?s))")
    ap.add_argument("--report-repeats", default=None, metavar="CSV",
                    help="write units (paragraphs/sentences) found in many documents to this CSV")
    ap.add_argument("--drop-repeats", action="store_true", help="remove those repeated units from the texts")
    ap.add_argument("--repeat-min-docs", type=int, default=4, help="a unit counts as repeated in >= this many documents")
    ap.add_argument("--repeat-unit", choices=["paragraph", "sentence"], default="paragraph")
    ap.add_argument("--drop-future", action="store_true", help="drop documents dated after today")
    ap.add_argument("--today", default=None, help="override today's date (YYYY-MM-DD)")
    ap.add_argument("--batch-min", type=int, default=4, help="warn when this many documents share one date")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    for col in (a.text_col, a.date_col):
        if col not in df.columns:
            sys.exit(f"column '{col}' not found; columns are: {list(df.columns)}")
    n0 = len(df)

    # ---- dates -------------------------------------------------------------------------------
    parsed = pd.to_datetime(df[a.date_col].astype(str).str.strip(), format=a.date_format, errors="coerce")
    bad = parsed.isna()
    if bad.any():
        print(f"[dates] {int(bad.sum())} rows do not match format {a.date_format!r} and are dropped. Examples:")
        print(df.loc[bad, a.date_col].head(5).to_string())
    df = df.loc[~bad].copy()
    df[a.date_col] = parsed[~bad].dt.strftime("%Y-%m-%d")
    today = pd.Timestamp(a.today) if a.today else pd.Timestamp.today().normalize()
    future = pd.to_datetime(df[a.date_col]) > today
    if future.any():
        print(f"[dates] WARNING: {int(future.sum())} documents are dated after {today.date()}. "
              f"Check the format ({a.date_format!r}) and the raw values:")
        print(df.loc[future, [a.date_col]].head(10).to_string())
        if a.drop_future:
            df = df.loc[~future].copy()
            print("[dates] dropped them (--drop-future)")
    counts = df[a.date_col].value_counts()
    batches = counts[counts >= a.batch_min]
    if len(batches):
        print(f"[dates] {len(batches)} dates carry >= {a.batch_min} documents each (scrape/upload dates?):")
        print(batches.to_string())

    # ---- boilerplate -------------------------------------------------------------------------
    texts = df[a.text_col].fillna("").astype(str)
    for pat in a.remove_regex:
        rx = re.compile(pat, re.IGNORECASE)
        hit = texts.map(lambda t: bool(rx.search(t)))
        texts = texts.map(lambda t: rx.sub("", t))
        print(f"[regex] {pat[:60]!r}: removed from {int(hit.sum())} documents")
    if a.report_repeats or a.drop_repeats:
        docs_by_key, example = {}, {}
        for i, t in texts.items():
            for u in units(t, a.repeat_unit):
                k = norm_key(u)
                if len(k) >= 20:
                    example.setdefault(k, u)
                    docs_by_key.setdefault(k, set()).add(i)
        rep_keys = {k for k, d in docs_by_key.items() if len(d) >= a.repeat_min_docs}
        rows = sorted(((len(docs_by_key[k]), k) for k in rep_keys), reverse=True)
        print(f"[repeats] {len(rows)} {a.repeat_unit}s occur in >= {a.repeat_min_docs} documents")
        for n, k in rows[:10]:
            print(f"    {n:4d} docs | {example[k][:110]!r}")
        if a.report_repeats:
            pd.DataFrame([{"n_docs": n, "first_date": df.loc[sorted(docs_by_key[k]), a.date_col].min(),
                           "last_date": df.loc[sorted(docs_by_key[k]), a.date_col].max(), "example": example[k]}
                          for n, k in rows]).to_csv(a.report_repeats, index=False)
            print(f"[repeats] full list -> {a.report_repeats}")
        if a.drop_repeats and rep_keys:
            texts = texts.map(lambda t: strip_units(t, rep_keys, a.repeat_unit))
            print(f"[repeats] removed from {len(set().union(*(docs_by_key[k] for k in rep_keys)))} documents")
    texts = texts.map(lambda t: re.sub(r"\n{3,}", "\n\n", t).strip())
    df[a.text_col] = texts
    empty = texts.str.len() == 0
    if empty.any():
        print(f"[text] {int(empty.sum())} documents are empty after cleaning and are dropped")
        df = df.loc[~empty].copy()

    # ---- ids ---------------------------------------------------------------------------------
    df = df.sort_values(a.date_col, kind="stable").reset_index(drop=True)
    if "id" not in df.columns:
        df.insert(0, "id", [f"doc{i + 1:03d}" for i in range(len(df))])
        print("[id] wrote an `id` column (doc001... in date order)")

    df.to_csv(a.output, index=False)
    print(f"\n{n0} rows in, {len(df)} rows out -> {a.output}")
    print(f"date range: {df[a.date_col].min()} .. {df[a.date_col].max()}")


if __name__ == "__main__":
    main()
