#!/usr/bin/env python3
"""
analyze_corpus.py - corpus-level analysis of proposition_tracker.py output (PNG figures + CSV tables)

Reads, from the pipeline's output folder:
    propositions.csv        (required)
    narrative_summary.csv   (optional, for the narrative-timeline figure)

Usage
-----
    python analyze_corpus.py --input-dir proposition_out/<dataset> --out analysis_out
    python analyze_corpus.py --input-dir proposition_out/<dataset> --out analysis_out --periods year
    python analyze_corpus.py --input-dir ... --out ... --meta cleaned.csv     # adds source-type figure
        --meta may be the CSV you gave the tracker. If it has an `id` column that is used as doc_id; if not,
        the script rebuilds the tracker's numbering (doc001... after sorting by date) and checks it against
        the dates in propositions.csv before using it.

Needs only pandas, numpy, matplotlib (plus emotions.py next to it).

What it does
------------
 1. Removes boilerplate: any (actor, action, object) triple found in >= --boilerplate-min-docs
    different documents (e.g. a newsletter consent footer). The removed rows are saved so you can check.
 2. Data-quality report: future dates, many documents on one date, missing document ids.
 3. Groups documents into periods and computes, per period: emotion, orientation, actor class,
    keyword themes, top objects. All numbers are saved as CSV in <out>/tables.
 4. Forest plot: last period (or --focus) vs all other periods with document-level bootstrap
    intervals (documents, not propositions, are resampled, because propositions from one document
    are not independent).
 5. Timeline of the narrative chains, if narrative_summary.csv exists.

CAVEATS (say them in the talk)
 - Themes and actor classes are KEYWORD RULES (edit THEMES / ACTOR_CLASSES below). They are rough
   and cover only part of the propositions. Validate on a hand-checked sample.
 - Emotion / orientation come from one LLM label per paragraph.
 - Intervals are 95% percentile bootstrap intervals, not corrected for multiple comparisons.
 - With few documents in a period (see the N labels), shares are unstable.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from emotions import EMOTION_GROUP  # noqa: E402

# --------------------------------------------------------------------------------------------- #
# Editable keyword rules
# --------------------------------------------------------------------------------------------- #
THEMES = {
    "policy / law / targets": r"klimaschutzgesetz|climate protection law|climate law|\blaw\b|legislat|\bpolic(?:y|ies)|target|budget|regulat|\bban\b|subsid|\btax|parliament|political",
    "fossil fuels / energy": r"fossil|\bgas\b|\boil\b|coal|energy|wind|solar|renewable|\blng\b|pipeline|power plant|nuclear|photovoltaic|heating",
    "mobility / infrastructure": r"transport|traffic|\bcars?\b|motorway|highway|autobahn|lobau|\brail|train|\bbike|cycl|mobility|flight|airport|\broads?\b|tunnel|\bbus\b|public transit",
    "protest / mobilization": r"strike|protest|demonstrat|\bmarch|rally|mobili[sz]|street|join us|join the|sit-in|occupation|blockade|\bblock\b|petition",
    "justice / solidarity": r"justice|racis|solidarity|human rights|inequalit|global south|social|migrat|refugee|feminis|gender|decoloni",
    "science / impacts": r"scien|ipcc|\bheat|drought|flood|extreme weather|temperature|warming|glacier|sea level|biodivers|species|forest|wildfire",
    "generic crisis framing": r"climate crisis|climate change|climate emergency|climate catastroph|climate breakdown",
    "pandemic": r"pandemic|corona|covid|lockdown",
    "war / security": r"ukraine|russia|\bwar\b|military|invasion",
    "education / youth": r"school|student|teacher|universit|education|youth|young people|children|generation",
}

ACTOR_CLASSES = {
    "collective (we / FFF / activists)": {
        "we", "us", "our", "wir", "fridays for future", "fff", "fridays for future austria", "activists",
        "young activists", "young people", "climate activists", "the movement", "movement", "fff activists",
        "youth", "students"},
    "addressee (you)": {"you", "du", "ihr", "everyone", "everybody", "anyone", "each of us", "all of us"},
    "individual (I)": {"i", "ich", "me", "my"},
    "institutions / politics": {
        "government", "politicians", "politics", "eu", "austria", "state", "politician", "parliament",
        "federal government", "chancellor", "ministry", "minister", "policy makers", "decision makers",
        "european union", "vienna", "city of vienna", "city", "city government"},
    "meta (speaker / author)": {"speaker", "author", "narrator", "the speaker", "the author", "interviewee"},
}

# Emotion labels follow Jasper (1998), Table I; scheme and grouping live in emotions.py
EMOTION_ORDER = list(EMOTION_GROUP)  # table order, 'neutral' last
GROUP_ORDER = ["affective", "reactive", "mood", "neutral", "other/legacy"]
GROUP_COLORS = {"affective": "#2b6cb0", "reactive": "#d1495b", "mood": "#2a9d8f", "neutral": "#c9c9c9",
                "other/legacy": "#e9c46a"}  # other/legacy = labels from an older run, e.g. 'determination' 
ORIENT_ORDER = ["supportive", "neutral", "ambivalent", "critical"]
ORIENT_COLORS = {"supportive": "#2a9d8f", "neutral": "#c9c9c9", "ambivalent": "#e9c46a", "critical": "#d1495b"}
ACTOR_COLORS = ["#2a9d8f", "#e9c46a", "#9b8bb8", "#d1495b", "#8ab17d", "#c9c9c9"]


# --------------------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------------------- #
def parse_periods(spec, years):
    """'2020,2021,2022-2023' -> [(label, first_year, last_year)]; 'year' -> one period per year."""
    if spec.strip().lower() == "year":
        return [(str(y), y, y) for y in sorted(years)]
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        a, b = (int(x) for x in tok.split("-")) if "-" in tok else (int(tok), int(tok))
        out.append((str(a) if a == b else f"{a}\u2013{str(b)[-2:]}", a, b))
    return out


def actor_class(a):
    a = str(a).lower().strip()
    for name, members in ACTOR_CLASSES.items():
        if a in members:
            return name
    return "other"


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", path)


def period_labels_with_n(order, docs_n, props_n):
    return [f"{p}\n{docs_n.get(p, 0)} docs\n{props_n.get(p, 0)} props" for p in order]


def stacked(ax, share, order_cols, colors, labels):
    """100% stacked bars, one bar per period (index of `share`)."""
    share = share.reindex(columns=[c for c in order_cols if c in share.columns] +
                          [c for c in share.columns if c not in order_cols]).fillna(0)
    bottom = np.zeros(len(share))
    x = np.arange(len(share))
    for col in share.columns:
        ax.bar(x, share[col].values, bottom=bottom, color=colors.get(col, "#bbbbbb"), label=col,
               edgecolor="white", linewidth=0.6)
        for xi, (v, b) in enumerate(zip(share[col].values, bottom)):
            if v >= 6:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8, color="black")
        bottom += share[col].values
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% of propositions")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=9)


def heatmap(ax, df, title, fmt="{:.0f}", cmap="YlOrRd", xlabels=None):
    vals = df.values.astype(float)
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=0, vmax=max(np.nanmax(vals), 1e-9))
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(xlabels if xlabels is not None else df.columns, fontsize=9)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index, fontsize=9)
    thr = np.nanmax(vals) * 0.6
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if vals[i, j] > thr else "black")
    ax.set_title(title, fontsize=10)
    return im


# --------------------------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------------------------- #
def main():
    global THEMES, ACTOR_CLASSES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="pipeline output folder holding propositions.csv")
    ap.add_argument("--out", default="analysis_out", help="folder for figures and tables")
    ap.add_argument("--config", default=None, metavar="JSON",
                    help='JSON with optional keys "themes" {name: regex} and "actor_classes" {name: [actor, ...]}; '
                         "replaces the built-in THEMES / ACTOR_CLASSES (see demo/analysis_config.json)")
    ap.add_argument("--periods", default="year",
                    help="comma list of years or year ranges, or 'year' for one period per year")
    ap.add_argument("--focus", default=None, help="period label to compare with all others (default: last)")
    ap.add_argument("--boilerplate-min-docs", type=int, default=5,
                    help="remove (actor, action, object) triples seen in at least this many documents")
    ap.add_argument("--no-boilerplate-filter", action="store_true")
    ap.add_argument("--batch-min", type=int, default=4, help="flag dates carrying at least this many documents")
    ap.add_argument("--today", default=None, help="override today's date (YYYY-MM-DD)")
    ap.add_argument("--meta", default=None, help="optional CSV with an `id` column (= doc_id) and `source`/`title`")
    ap.add_argument("--drop-future", action="store_true",
                    help="exclude documents dated after today from all figures and tables")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    if a.config:
        import json
        user = json.loads(Path(a.config).read_text(encoding="utf-8"))
        THEMES = user.get("themes", THEMES)
        ACTOR_CLASSES = {k: {x.lower() for x in v} for k, v in user.get("actor_classes", ACTOR_CLASSES).items()}

    inp, out = Path(a.input_dir), Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    figs = out
    pf = inp / "propositions.csv"
    if not pf.exists():
        sys.exit(f"{pf} not found")
    raw = pd.read_csv(pf)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).copy()
    for col in ["actor", "action", "object", "claim", "emotion", "orientation"]:
        raw[col] = raw[col].fillna("").astype(str)
    raw["emotion"] = raw["emotion"].str.lower().replace("", "unspecified")
    raw["orientation"] = raw["orientation"].str.lower().replace("", "unspecified")
    today = pd.Timestamp(a.today) if a.today else pd.Timestamp.today().normalize()
    print(f"{len(raw)} propositions, {raw.doc_id.nunique()} documents, "
          f"{raw.date.min().date()} .. {raw.date.max().date()}")

    # ---- 1. boilerplate ----------------------------------------------------------------------
    key = raw.actor.str.lower().str.strip() + " | " + raw.action.str.lower().str.strip() + " | " + \
        raw.object.str.lower().str.strip()
    raw["_key"] = key
    if a.no_boilerplate_filter:
        df = raw.copy()
    else:
        ndocs = raw.groupby("_key").doc_id.nunique()
        bad = ndocs[ndocs >= a.boilerplate_min_docs].index
        removed = raw[raw._key.isin(bad)]
        df = raw[~raw._key.isin(bad)].copy()
        removed.drop(columns="_key").to_csv(out / "tables" / "removed_boilerplate.csv", index=False)
        print(f"\n[boilerplate] removed {len(removed)} propositions "
              f"({removed.doc_id.nunique()} docs) in {len(bad)} repeated triples:")
        for k in bad:
            print(f"    {k}   ({ndocs[k]} docs)")
        if len(bad) == 0:
            print("    none")

    # ---- 2. data quality ---------------------------------------------------------------------
    docs = df.groupby("doc_id").agg(date=("date", "first"), n_props=("id", "count")).reset_index()
    all_docs = raw.groupby("doc_id").agg(date=("date", "first")).reset_index()
    future = all_docs[all_docs.date > today]
    per_date = all_docs.groupby("date").size()
    batches = per_date[per_date >= a.batch_min]
    nums = sorted(int(m.group()) for m in (re.search(r"\d+", str(d)) for d in all_docs.doc_id) if m)
    missing = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums)) if nums else []
    ambiguous = (all_docs.date.dt.day <= 12).mean()
    q = [f"Documents with propositions: {len(all_docs)}",
         f"Dates after today ({today.date()}): {len(future)}  " + ", ".join(f"{r.doc_id}={r.date.date()}" for r in future.itertuples()),
         f"Dates carrying >= {a.batch_min} documents: " + (", ".join(f"{d.date()} ({n})" for d, n in batches.items()) or "none"),
         f"Document numbers without any proposition (gaps in id sequence): {missing if missing else 'none'}",
         f"Share of documents with day-of-month <= 12 (dates that could be day/month swapped): {ambiguous:.0%}"]
    (out / "tables" / "data_quality.txt").write_text("\n".join(q) + "\n", encoding="utf-8")
    print("\n[data quality]\n  " + "\n  ".join(q))

    if a.drop_future and len(future):
        df = df[df.date <= today].copy()
        print(f"  --drop-future: excluded {len(future)} documents dated after {today.date()}")

    # ---- optional metadata -------------------------------------------------------------------
    has_source = False
    if a.meta:
        meta = pd.read_csv(a.meta)
        usable = True
        if "id" in meta.columns:
            meta["id"] = meta["id"].astype(str)
        else:
            # rebuild the tracker's numbering: drop rows without a valid date or text, stable sort by date,
            # then doc001, doc002, ...
            if not {"full_text", "date"} <= set(meta.columns):
                print("  --meta has no `id` column and lacks full_text/date to rebuild it; skipping source figure")
                usable = False
            else:
                d = pd.to_datetime(meta["date"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
                ok = d.notna() & meta["full_text"].fillna("").astype(str).str.strip().ne("")
                meta = meta.loc[ok].assign(_d=d[ok]).sort_values("_d", kind="stable").reset_index(drop=True)
                meta["id"] = [f"doc{i + 1:03d}" for i in range(len(meta))]
                chk = meta.merge(all_docs[["doc_id", "date"]].rename(columns={"date": "_pdate"}),
                                 left_on="id", right_on="doc_id")
                agree = (chk["_d"].dt.normalize() == chk["_pdate"].dt.normalize()).mean() if len(chk) else 0
                print(f"  --meta: rebuilt doc ids; dates agree with propositions.csv for {agree:.1%} of documents")
                if agree < 0.95:
                    print("  numbering does not match the pipeline run (different input file or cleaning?); "
                          "skipping source figure")
                    usable = False
        if usable:
            keep = ["id"] + [c for c in ("source", "title") if c in meta.columns]
            df = df.merge(meta[keep].rename(columns={"id": "doc_id"}), on="doc_id", how="left")
            has_source = "source" in df.columns and df["source"].notna().any()
            if not has_source:
                print("  --meta has no usable `source` column (or no ids matched); skipping source figure")

    # ---- 3. periods and features -------------------------------------------------------------
    df["year"] = df.date.dt.year
    periods = parse_periods(a.periods, df.year.unique())
    ymap = {y: lab for lab, y0, y1 in periods for y in range(y0, y1 + 1)}
    df["period"] = df.year.map(ymap)
    dropped = df.period.isna().sum()
    if dropped:
        print(f"\n[periods] {dropped} propositions fall outside --periods and are ignored "
              f"(years {sorted(df[df.period.isna()].year.unique())})")
    df = df.dropna(subset=["period"]).copy()
    order = [lab for lab, _, _ in periods if lab in set(df.period)]
    docs_n = df.groupby("period").doc_id.nunique().to_dict()
    props_n = df.groupby("period").size().to_dict()
    xl = period_labels_with_n(order, docs_n, props_n)
    pd.DataFrame({"documents": docs_n, "propositions": props_n}).loc[order].to_csv(out / "tables" / "period_sizes.csv")
    sparse = [p for p in order if docs_n[p] < 8]
    if sparse:
        print(f"\n[note] periods with fewer than 8 documents (interpret with care): {sparse}")

    df["actor_class"] = df.actor.map(actor_class)
    txt = (df.object + " " + df.claim).str.lower()
    for name, rx in THEMES.items():
        df["T:" + name] = txt.str.contains(rx, regex=True)
    theme_cols = ["T:" + t for t in THEMES]
    cov = df[theme_cols].any(axis=1).mean()
    print(f"\n[themes] keyword themes cover {cov:.0%} of propositions; "
          f"'other' actors = {(df.actor_class == 'other').mean():.0%}")

    df["emotion_group"] = df.emotion.map(EMOTION_GROUP).fillna("other/legacy")
    legacy = sorted(set(df.emotion) - set(EMOTION_GROUP))
    if legacy:
        print(f"\n[emotion] labels not in Jasper's list (older run?): {legacy}")

    def share(col):
        return (pd.crosstab(df.period, df[col], normalize="index") * 100).reindex(order)

    emo, ori, act = share("emotion"), share("orientation"), share("actor_class")
    emg = share("emotion_group")
    emg.round(1).to_csv(out / "tables" / "emotion_group_share_by_period.csv")
    emo.round(1).to_csv(out / "tables" / "emotion_share_by_period.csv")
    ori.round(1).to_csv(out / "tables" / "orientation_share_by_period.csv")
    act.round(1).to_csv(out / "tables" / "actor_class_share_by_period.csv")
    theme_prop = (df.groupby("period")[theme_cols].mean() * 100).reindex(order)
    theme_doc = (df.groupby(["period", "doc_id"])[theme_cols].any().groupby("period").mean() * 100).reindex(order)
    theme_prop.columns = theme_doc.columns = list(THEMES)
    theme_prop.round(1).to_csv(out / "tables" / "theme_share_of_propositions_by_period.csv")
    theme_doc.round(1).to_csv(out / "tables" / "theme_share_of_documents_by_period.csv")
    docw = df.groupby(["period", "doc_id"]).emotion.value_counts(normalize=True).unstack(fill_value=0)
    (docw.groupby(level=0).mean() * 100).reindex(order).round(1).to_csv(
        out / "tables" / "emotion_share_by_period_document_weighted.csv")
    top_actors = df.assign(a=df.actor.str.lower().str.strip()).groupby(["period", "a"]).size() \
        .rename("n").reset_index().sort_values(["period", "n"], ascending=[True, False]).groupby("period").head(15)
    top_actors.to_csv(out / "tables" / "top_actors_by_period.csv", index=False)

    print("\nWriting figures:")
    # ---- fig 01: corpus overview ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    y = df.groupby("year").agg(docs=("doc_id", "nunique"), props=("id", "count"))
    ax[0].bar(y.index.astype(str), y.docs, color="#577590")
    ax[0].set_title("Documents per year (with \u22651 proposition)")
    ax[1].bar(y.index.astype(str), y.props, color="#2a9d8f")
    ax[1].set_title("Propositions per year (after boilerplate removal)")
    for a_, col in zip(ax, ("docs", "props")):
        for i, v in enumerate(y[col]):
            a_.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        a_.spines[["top", "right"]].set_visible(False)
    save(fig, figs / "fig01_corpus_overview.png")

    # ---- fig 02: timeline / date quality ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    d2 = docs.copy()
    cnt = d2.groupby("date").doc_id.transform("count")
    colors = np.where(d2.date > today, "#d1495b", np.where(cnt >= a.batch_min, "#f4a261", "#577590"))
    ax.scatter(d2.date, d2.n_props, c=colors, s=28, alpha=0.85)
    ax.axvline(today, color="#d1495b", ls="--", lw=1)
    ax.set_ylabel("propositions per document")
    ax.set_title("Each dot = one document   (orange: date shared by \u2265%d documents, red: after today)" % a.batch_min)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, figs / "fig02_document_timeline.png")

    # ---- fig 03-05: stacked shares ----------------------------------------------------------
    for name, tab, cols, colors, title in [
        ("fig03_emotion_group_by_period", emg, GROUP_ORDER, GROUP_COLORS,
         "Emotion type per proposition (Jasper 1998: affective / reactive / moods)"),
        ("fig04_orientation_by_period", ori, ORIENT_ORDER, ORIENT_COLORS, "Orientation (author's stance towards the claim)"),
        ("fig05_actor_class_by_period", act, list(ACTOR_CLASSES) + ["other"],
         dict(zip(list(ACTOR_CLASSES) + ["other"], ACTOR_COLORS)), "Who acts? (keyword classes of the actor)"),
    ]:
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        stacked(ax, tab, cols, colors, xl)
        ax.set_title(title)
        save(fig, figs / f"{name}.png")

    # ---- fig 03b: the 17 categories as a heatmap ---------------------------------------------
    cats = EMOTION_ORDER + [c for c in emo.columns if c not in EMOTION_ORDER]  # show all, also empty categories
    fig, ax = plt.subplots(figsize=(8.5, 0.38 * len(cats) + 1.8))
    im = heatmap(ax, emo.reindex(columns=cats, fill_value=0).T, "Emotion category per proposition (% of the period's propositions)",
                 fmt="{:.1f}", cmap="YlOrRd", xlabels=xl)
    ax.set_yticklabels([f"{c}  [{EMOTION_GROUP.get(c, 'legacy')[:3]}]" for c in cats], fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03)
    save(fig, figs / "fig03b_emotion_categories_by_period.png")

    # ---- fig 06: themes --------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.55})
    im0 = heatmap(ax[0], theme_prop.T, "% of propositions touching the theme", xlabels=xl)
    heatmap(ax[1], theme_doc.T, "% of documents with \u22651 such proposition", xlabels=xl, cmap="PuBu")
    ax[1].set_yticklabels([])
    fig.suptitle(f"Keyword themes (cover {cov:.0%} of propositions; a proposition can have several themes)", y=1.0, fontsize=11)
    save(fig, figs / "fig06_themes_by_period.png")

    # ---- fig 07: top objects ---------------------------------------------------------------
    obj = df.assign(o=df.object.str.lower().str.strip())
    obj = obj[~obj.o.isin(["", "unspecified"])]
    top = obj.o.value_counts().head(15).index
    tab = obj[obj.o.isin(top)].groupby(["o", "period"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    tab = tab.loc[top]
    tab = tab / pd.Series(props_n).reindex(order).values * 100
    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = heatmap(ax, tab, "Most frequent objects (per 100 propositions of the period)", fmt="{:.1f}", cmap="YlGnBu", xlabels=xl)
    fig.colorbar(im, ax=ax, fraction=0.03)
    save(fig, figs / "fig07_top_objects.png")
    tab.round(2).to_csv(out / "tables" / "top_objects_per_100_propositions.csv")

    # ---- fig 08: focus period vs rest, document-level bootstrap ----------------------------
    focus = a.focus or order[-1]
    if focus not in order or len(order) < 2:
        print(f"  fig08 needs a --focus period among >= 2 periods (focus {focus!r}, periods {order}); skipping")
    else:
        feats = {}
        for g in ["affective", "reactive", "mood"]:
            feats[f"emotion group: {g}"] = df.emotion_group == g
        for e in EMOTION_ORDER:
            feats[f"emotion: {e}"] = df.emotion == e
        for o in ["supportive", "critical", "neutral"]:
            feats[f"orientation: {o}"] = df.orientation == o
        for c in list(ACTOR_CLASSES):
            feats[f"actor: {c}"] = df.actor_class == c
        for t in THEMES:
            feats[f"theme: {t}"] = df["T:" + t]
        F = pd.DataFrame(feats).astype(float)
        F = F.loc[:, F.mean() >= 0.02]
        F["_n"] = 1.0
        per_doc = F.groupby(df.doc_id).sum()
        is_focus = df.groupby("doc_id").period.first().reindex(per_doc.index) == focus
        Xf, Xo = per_doc[is_focus], per_doc[~is_focus]
        rng = np.random.default_rng(1)

        def shares(X, idx=None):
            Z = X.values if idx is None else X.values[idx]
            return Z[:, :-1].sum(0) / Z[:, -1].sum()

        est = shares(Xf) - shares(Xo)
        draws = np.empty((a.bootstrap, len(est)))
        for b in range(a.bootstrap):
            draws[b] = shares(Xf, rng.integers(0, len(Xf), len(Xf))) - shares(Xo, rng.integers(0, len(Xo), len(Xo)))
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        res = pd.DataFrame({"feature": per_doc.columns[:-1], "share_focus": shares(Xf) * 100,
                            "share_rest": shares(Xo) * 100, "diff_pp": est * 100,
                            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100}).sort_values("diff_pp")
        res.round(2).to_csv(out / "tables" / f"focus_{focus}_vs_rest.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(res) + 1.6))
        for i, r in enumerate(res.itertuples()):
            sig = r.ci_low_pp > 0 or r.ci_high_pp < 0
            col = ("#d1495b" if r.diff_pp > 0 else "#2a6f97") if sig else "#999999"
            ax.plot([r.ci_low_pp, r.ci_high_pp], [i, i], color=col, lw=2)
            ax.plot(r.diff_pp, i, "o", color=col, ms=5)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(res)))
        ax.set_yticklabels(res.feature, fontsize=8)
        ax.set_xlabel("difference in share, percentage points  (focus period minus all other periods)")
        ax.set_title(f"{focus} ({len(Xf)} docs) vs all other periods ({len(Xo)} docs)\n"
                     "95% document-level bootstrap interval; coloured = interval excludes 0 (not multiplicity-corrected)",
                     fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, figs / f"fig08_focus_{focus}_vs_rest.png")

    # ---- fig 09: narrative chains ----------------------------------------------------------
    ns = inp / "narrative_summary.csv"
    if ns.exists():
        n = pd.read_csv(ns)
        n["s"], n["e"] = pd.to_datetime(n.start_date), pd.to_datetime(n.end_date)
        if "likely_boilerplate" in n.columns:  # flag written by build_viewer.py
            n["boiler"] = n["likely_boilerplate"].astype(str).str.lower().eq("true")
        else:  # older output: unchanged similarity over >= 4 documents
            n["boiler"] = (n.total_drop.fillna(0) == 0) & (n.n_documents >= 4)
        n = n.sort_values("s").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(11, 0.55 * len(n) + 1.8))
        for i, r in n.iterrows():
            width = max((r.e - r.s).days, 14)  # very short chains stay visible
            col = "#d1495b" if str(r.get("emotion_valence_flip", "")).lower() == "true" else "#577590"
            ax.barh(i, width, left=r.s, color=col, hatch="//" if r.boiler else None,
                    edgecolor="white" if r.boiler else col, height=0.6)
            lab = f"{r.title}  \u2013 {int(r.n_documents)} docs, sim to day 1 = {r.sim_to_day1_last:.2f}"
            if r.boiler:
                lab += "  [likely boilerplate]"
            ax.text(r.e + pd.Timedelta(days=8), i, lab, va="center", fontsize=8)
        ax.set_yticks([])
        ax.invert_yaxis()
        lo_x = min(n.s.min(), df.date.min())
        ax.set_xlim(lo_x - pd.Timedelta(days=20), max(n.e.max(), df.date.max()) + pd.Timedelta(days=20))  # labels spill right; save() uses bbox_inches="tight"
        ax.set_title("Narrative chains found by the pipeline (red = emotion flips between encouraging and dampening; hatched = likely boilerplate)")
        ax.legend(handles=[Patch(color="#577590", label="no valence flip"), Patch(color="#d1495b", label="valence flip")],
                  loc="lower left", frameon=False, fontsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        save(fig, figs / "fig09_narrative_chains.png")
        print(f"  note: longest chain spans {int(n.days_span.max())} days; chains with >= 4 documents: "
              f"{int((n.n_documents >= 4).sum())}")
    else:
        print("  narrative_summary.csv not found; skipping fig09")

    # ---- fig 10: source type ---------------------------------------------------------------
    if has_source:
        sd = df.groupby("doc_id").agg(period=("period", "first"), source=("source", "first")).reset_index()
        mix = (pd.crosstab(sd.period, sd.source, normalize="index") * 100).reindex(order)
        emo_src = (pd.crosstab(df.source, df.emotion_group, normalize="index") * 100)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"wspace": 0.18})
        mix.plot(kind="bar", stacked=True, ax=ax[0], colormap="Set2", rot=0, edgecolor="white")
        ax[0].set_title("Source type of documents per period (% of documents)")
        ax[0].set_xlabel("")
        ax[0].set_xticklabels([f"{p_}\n{docs_n.get(p_, 0)} docs" for p_ in mix.index], rotation=0)
        ax[0].set_ylim(0, 100)
        ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(mix.columns), frameon=False, fontsize=9)
        ax[0].spines[["top", "right"]].set_visible(False)
        cols = [c for c in GROUP_ORDER if c in emo_src.columns]
        emo_src[cols].plot(kind="bar", stacked=True, ax=ax[1], color=[GROUP_COLORS[c] for c in cols], rot=0, edgecolor="white")
        ax[1].set_title("Emotion type by source type (% of propositions)")
        ax[1].set_xlabel("")
        ax[1].set_ylim(0, 100)
        ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(cols), frameon=False, fontsize=9)
        ax[1].spines[["top", "right"]].set_visible(False)
        save(fig, figs / "fig10_source_type.png")
        emo_src.round(1).to_csv(out / "tables" / "emotion_group_share_by_source.csv")
        mix.round(1).to_csv(out / "tables" / "source_mix_by_period.csv")

    print(f"\nDone. Figures in {out}/, tables in {out}/tables/")


if __name__ == "__main__":
    main()
