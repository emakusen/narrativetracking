#!/usr/bin/env python3
r"""
proposition_tracker.py
======================
Narrative tracing for LONG documents (blog posts, speeches, transcripts) instead of tweets.

Instead of clustering short texts, the unit of analysis is a PROPOSITION:

    actor + action + object  (+ orientation + emotion)

Pipeline per document
---------------------
1. Split the document into paragraphs (blank-line separated; paragraphs longer than
   --max-sentences are cut into chunks of at most that many sentences).
2. For every paragraph: rule-based SVO (spaCy) + one LLM call that returns a short description
   and up to --max-props propositions (actor, action, object, orientation, emotion, claim).
3. Merge: a proposition in paragraph p+1 whose SVO structure matches one in paragraph p
   (weighted cosine of the element embeddings >= --merge-thr) is the SAME proposition. The merged
   proposition keeps every mention, so within-document change stays traceable.

Without an LLM: `--extractor spacy` builds the propositions from the dependency parse (spacy_props.py).
Orientation and emotion then stay 'neutral'.

Across documents (time)
-----------------------
4. Propositions of consecutive documents are linked when their SVO similarity is >= --link-thr
   (looser than merging on purpose: transformation is allowed). Orientation and emotion are never
   required to match here.
5. Each chain is one narrative. For each step we compare ALL elements with the previous step:
   actor / action / object by cosine similarity, orientation (supportive, critical, neutral,
   ambivalent) and emotion by category, with flips (supportive <-> critical; encouraging <-> dampening
   emotion, see EMOTION_DIRECTION) flagged separately. Emotions follow Jasper's (1998) typology of protest emotions. One matrix figure per narrative shows every element over time.

Usage
-----
    ollama pull gpt-oss:20b && ollama serve
    python -m spacy download de_core_news_sm          # or en_core_web_sm for --lang en
    python clean_input.py --input raw.csv --output cleaned.csv --date-format "%d.%m.%Y"    # ISO dates, ids, footers
    python proposition_tracker.py --input cleaned.csv --lang en --out proposition_out \
        --context "a climate-movement organisation" --max-sentences 4 --merge-thr 0.80 --link-thr 0.60

Per dataset the output folder also contains viewer.html (interactive, open it in a browser),
narrative_trajectories.csv (every step measured against day 1) and narrative_summary.csv.

Each input needs `full_text` and `date` columns (ISO dates, see clean_input.py); optional `title`, `id`, `source`.
LLM output (elements, claim) is always in English so matching works across German/English sources;
the evidence paragraphs stay in their original language.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import build_viewer  # noqa: E402
from emotions import (EMOTION_ALIASES, EMOTION_DIRECTION, EMOTION_GROUP, EMOTIONS, JASPER_EMOTIONS)  # noqa: E402,F401
from llm_utils import LLM, extract_sao, load_spacy, parse_json  # noqa: E402
from spacy_props import extract_propositions  # noqa: E402

log = logging.getLogger("proposition_tracker")

ELEMENTS = ("actor", "action", "object")
ORIENTATIONS = ["supportive", "critical", "neutral", "ambivalent"]
UNRECOGNISED_EMOTIONS: Counter = Counter()  # labels returned by the LLM that match nothing (counted as neutral)
EMOTION_PROMPT = (
    "- emotion: the dominant emotion the author expresses with it, following Jasper's typology of protest emotions. "
    "Answer with exactly ONE of these keys, or 'neutral' if no emotion is expressed:\n"
    + "\n".join(f"    {k}: {t}" for k, (_g, _w, t) in JASPER_EMOTIONS.items()) + "\n"
)


def norm_emotion(v) -> str:
    """Map the LLM's answer to one of EMOTIONS; unknown answers become 'neutral' and are counted."""
    raw = str(v or "").strip().lower()
    if raw in EMOTIONS:
        return raw
    if raw in {"", "none", "n/a", "null", "unknown", "unspecified", "no emotion"}:
        return "neutral"
    for tok in re.findall(r"[a-z\u00e4\u00f6\u00fc\u00df]+", raw):
        if tok in EMOTION_ALIASES:
            return EMOTION_ALIASES[tok]
    UNRECOGNISED_EMOTIONS[raw] += 1
    return "neutral"
SPACY_MODELS = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
LANG_NAME = {"en": "English", "de": "German"}
STATUS_COLOR = {"start": "#9a9a94", "same": "#639922", "drift": "#BA7517", "shift": "#BA7517",
                "changed": "#E24B4A", "flip": "#E24B4A", "na": "#bdbdb5"}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    lang: str = "en"
    max_sentences: int = 4          # paragraph = at most this many sentences
    min_chars: int = 80             # shorter chunks are merged into the previous one or dropped
    max_props: int = 6              # propositions extracted per paragraph
    # --- merging inside one document ("same proposition") ---
    merge_thr: float = 0.80         # weighted SVO similarity needed to merge
    merge_window: int = 1           # 1 = only the directly preceding paragraph
    merge_require: str = "none"     # none | orientation | emotion | both must also be equal
    min_element_cos: float = 0.50   # every SVO element must reach this to merge
    # --- linking across documents ("same narrative strand, possibly transformed") ---
    link_thr: float = 0.60
    max_gap: int = 3                # documents that may lie between two linked propositions
    link_min_element_cos: float = 0.50   # every comparable SVO element must reach this to link across documents
    link_ignore_pronoun_actor: bool = True  # 'we'/'I' vs 'we'/'I' counts for nothing when linking
    # --- element status colours in the figure ---
    same_thr: float = 0.85          # element cosine >= this = unchanged
    drift_thr: float = 0.60         # >= this = drifted, below = changed
    weights: tuple = (0.25, 0.25, 0.50)  # actor, action, object
    min_path_len: int = 2
    top_n: int = 10
    skip_identical: bool = True    # do not render chains whose steps are verbatim repeats (footers, fixed slogans)
    identical_thr: float = 0.99    # a step counts as verbatim when its link similarity is at least this
    # --- narrative-aware linking ---
    candidate_top_k: int = 5       # historical candidates retained per proposition
    candidate_min_score: float = 0.42
    narrative_link_thr: float = 0.58
    llm_link_classification: bool = True
    max_link_calls: int = 0        # 0 = unlimited; cap for inexpensive dry runs
    max_steps: int = 5              # steps per figure page
    max_docs: int = 0               # 0 = all (use a small number for a dry run)
    extractor: str = "llm"          # llm | spacy (rule-based, no LLM calls; orientation/emotion stay neutral)
    context: str = "a social movement or advocacy organisation"  # who wrote the documents; goes into the LLM prompts
    force_english: bool = False     # ask the LLM to translate leftover German words (changes the prompt -> new cache keys)


@dataclass
class Mention:
    doc: int
    par: int
    actor: str
    action: str
    object: str
    orientation: str
    emotion: str
    claim: str
    evidence: str
    issue: str = ""
    stakes: str = ""
    goal: str = ""
    causal: str = ""
    evaluation: str = ""
    vec: dict = field(default_factory=dict, repr=False)
    frame_vec: dict = field(default_factory=dict, repr=False)


@dataclass
class Node:
    id: str
    doc: int
    doc_id: str
    date: pd.Timestamp
    title: str
    par_first: int
    par_last: int
    mentions: list
    vec: dict = field(repr=False)
    actor: str = ""
    action: str = ""
    object: str = ""
    orientation: str = "neutral"
    emotion: str = "neutral"
    claim: str = ""
    evidence: str = ""
    issue: str = ""
    stakes: str = ""
    goal: str = ""
    causal: str = ""
    evaluation: str = ""
    frame_vec: dict = field(default_factory=dict, repr=False)


# --------------------------------------------------------------------------- #
# Loading and paragraph segmentation
# --------------------------------------------------------------------------- #
URL_RE = re.compile(r"https?://\S+")
TS_RE = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\]")  # transcript time codes like [00:12:31]


def clean_doc(t: str) -> str:
    t = URL_RE.sub("", str(t)).replace("\r\n", "\n").replace("\r", "\n")
    t = TS_RE.sub("", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def load_documents(path) -> pd.DataFrame:
    path = Path(path)
    sfx = path.suffix.lower()
    if sfx in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True, convert_dates=False)
    elif sfx == ".json":
        df = pd.read_json(path, convert_dates=False)
    elif sfx == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    missing = {"full_text", "date"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path.name}: missing required column(s): {sorted(missing)}")
    if df["date"].astype(str).str.match(r"^\s*\d{1,2}\.\d{1,2}\.\d{2,4}").any():
        log.warning("%s: dates like 11.04.2026 found. pandas reads these month-first when the day is <= 12. "
                    "Run clean_input.py --date-format \"%%d.%%m.%%Y\" first.", path.name)
    created = pd.to_datetime(df["date"], errors="coerce", utc=True, format="mixed")
    if created.isna().any():
        log.warning("%s: %d rows with unparsable date dropped", path.name, int(created.isna().sum()))
    out = pd.DataFrame({
        "text": df["full_text"].map(clean_doc),
        "date": created.dt.tz_localize(None),
        "title": df["title"].astype(str) if "title" in df.columns else None,
        "doc_id": df["id"].astype(str) if "id" in df.columns else None,
    })
    out = out.dropna(subset=["date"])
    out = out[out.text.str.len() > 0].sort_values("date", kind="stable").reset_index(drop=True)
    if out.title.isna().all():
        out["title"] = out.text.str.slice(0, 50)
    out["title"] = out.title.fillna(out.text.str.slice(0, 50))
    out["doc_id"] = out.doc_id.where(out.doc_id.notna(), pd.Series([f"doc{i + 1:03d}" for i in range(len(out))]))
    log.info("%s: %d documents, %s to %s", path.name, len(out), out.date.min().date(), out.date.max().date())
    return out


def split_sentences(text: str, nlp) -> list:
    if nlp is not None:
        return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def split_paragraphs(text: str, nlp, cfg: Config) -> list:
    """Blank-line paragraphs, cut into chunks of <= max_sentences sentences."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()] or [text]
    out = []
    for b in blocks:
        sents, chunks = split_sentences(b, nlp), []
        for i in range(0, len(sents), cfg.max_sentences):
            chunk = " ".join(sents[i:i + cfg.max_sentences]).strip()
            if len(chunk) < cfg.min_chars and chunks:
                chunks[-1] += " " + chunk  # fold a tiny tail into the previous chunk
            else:
                chunks.append(chunk)
        out += [c for c in chunks if len(c) >= cfg.min_chars]
    return out


# --------------------------------------------------------------------------- #
# Paragraph -> propositions (SVO hints + LLM)
# --------------------------------------------------------------------------- #
def norm_choice(v, allowed, default):
    v = str(v or "").strip().lower()
    if v in allowed:
        return v
    return next((a for a in allowed if v.startswith(a[:4])), default) if v else default


def norm_text(v, maxlen=60) -> str:
    v = re.sub(r"\s+", " ", str(v or "")).strip().strip(".")
    return v[:maxlen] if v and v.lower() not in {"n/a", "none", "null", "unknown"} else "unspecified"


def llm_prompt(text: str, sao: dict, cfg: Config) -> str:
    return (
        f"You analyse one paragraph ({LANG_NAME[cfg.lang]}) from a blog post, newsletter, speech or transcript "
        f"of {cfg.context}.\n"
        f'Paragraph:\n"""\n{text}\n"""\n\n'
        f"Rule-based subject-verb-object hints (may be noisy): {json.dumps(sao, ensure_ascii=False)}\n\n"
        "Return ONLY JSON in English with this shape:\n"
        '{"description": "1-2 neutral sentences on what the paragraph argues or reports",\n'
        ' "propositions": [{"actor": "...", "action": "...", "object": "...", "orientation": "...", '
        '"emotion": "...", "claim": "...", "issue": "...", "stakes": "...", '
        '"goal": "...", "causal": "...", "evaluation": "..."}]}\n'
        f"Extract up to {cfg.max_props} DISTINCT independently meaningful propositions, not just the main claim. Do not omit reactions, demands, consequences, institutional actions or campaign actions.\n"
        "- actor: who acts or is held responsible; short canonical noun phrase (max 4 words), e.g. 'government', "
        "'young activists', 'we'\n"
        "- action: what the actor does or should do; lemmatised verb phrase (max 4 words)\n"
        "- object: what is acted upon or the topic (max 5 words)\n"
        "- orientation: the AUTHOR'S stance toward this proposition/object; one of "
        f"{ORIENTATIONS}\n"
        + EMOTION_PROMPT +
        "- claim: the proposition as ONE English sentence (max 25 words)\n"
        "Narrative frame fields: issue = underlying issue/topic (not just a broad domain); stakes = what is at risk/why it matters; "
        "goal = desired outcome or proposed solution; causal = attributed cause/responsibility; evaluation = how the situation/actors are judged. "
        "Use concise canonical phrases. These fields describe THIS proposition's framing and may legitimately change over time. "
        "Do not force different propositions into the same wording. Extract each distinct claim even when it is a response to another claim."
        + (" ALL values (actor, action, object, claim) MUST be English: translate any German words." if cfg.force_english else "")
    )


def spacy_propositions(text: str, nlp, cfg: Config) -> list:
    """Propositions from the dependency parse (see spacy_props.py); orientation and emotion are 'neutral'."""
    return [{"actor": norm_text(p["actor"]), "action": norm_text(p["action"]), "object": norm_text(p["object"]),
             "orientation": "neutral", "emotion": "neutral", "claim": norm_text(p["claim"], 220),
             "issue": norm_text(p.get("object") or p.get("claim")), "stakes": "unspecified", "goal": "unspecified",
             "causal": "unspecified", "evaluation": "unspecified"}
            for p in extract_propositions(text, nlp, cfg.lang, cfg.max_props)]


def analyse_paragraph(llm: LLM, text: str, sao: dict, cfg: Config, nlp=None):
    """Returns (description, [proposition dicts]). Uses spaCy when cfg.extractor == 'spacy', when the LLM is
    unavailable, or when a single LLM call fails."""
    out = None
    if cfg.extractor == "llm" and llm.ok:
        out = parse_json(llm._complete(llm_prompt(text, sao, cfg), 2500))
    if out is None:
        return "", spacy_propositions(text, nlp, cfg)
    props = []
    for p in ((out or {}).get("propositions") or [])[: cfg.max_props]:
        if not isinstance(p, dict):
            continue
        props.append({
            "actor": norm_text(p.get("actor")), "action": norm_text(p.get("action")),
            "object": norm_text(p.get("object")),
            "orientation": norm_choice(p.get("orientation"), ORIENTATIONS, "neutral"),
            "emotion": norm_emotion(p.get("emotion")),
            "claim": norm_text(p.get("claim"), 220),
            "issue": norm_text(p.get("issue") or p.get("object") or p.get("claim"), 120),
            "stakes": norm_text(p.get("stakes"), 120),
            "goal": norm_text(p.get("goal"), 120),
            "causal": norm_text(p.get("causal"), 120),
            "evaluation": norm_text(p.get("evaluation"), 120),
        })
    return norm_text(out.get("description"), 400), props


# --------------------------------------------------------------------------- #
# Similarity, merging, linking
# --------------------------------------------------------------------------- #
MIN_PRESENT = 2  # a proposition needs at least two specified SVO elements to be matched


def cos_or_none(a, b):
    """Cosine of two element vectors; None when either element is 'unspecified' (zero vector)."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return None if na < 1e-6 or nb < 1e-6 else float(a @ b) / (na * nb)


def prop_sim(a: dict, b: dict, w) -> tuple:
    """Weighted SVO similarity over the elements both propositions actually specify."""
    cos = {e: cos_or_none(a[e], b[e]) for e in ELEMENTS}
    present = [(wi, cos[e]) for wi, e in zip(w, ELEMENTS) if cos[e] is not None]
    if len(present) < MIN_PRESENT:
        return 0.0, cos
    tw = sum(x for x, _ in present)
    return float(sum(x * c for x, c in present) / tw), cos


FIRST_PERSON = {"we", "i", "us", "me", "our", "my", "wir", "ich", "uns", "mir", "mich", "unser"}


def link_sim(a, b, cfg: Config) -> tuple:
    """Similarity used for linking propositions across documents.

    Differs from prop_sim in two ways:
      * every comparable element must reach cfg.link_min_element_cos (no link carried by one element alone);
      * if both actors are first-person pronouns, the actor is ignored (it says nothing about the topic).
    Returns (score, full cos dict); score is 0.0 when the pair is not linkable. The cos dict still contains
    the actor cosine, so later change reporting is unaffected.
    """
    _, cos = prop_sim(a.vec, b.vec, cfg.weights)
    use = dict(cos)
    if cfg.link_ignore_pronoun_actor and a.actor.lower() in FIRST_PERSON and b.actor.lower() in FIRST_PERSON:
        use["actor"] = None
    present = [(w, use[e]) for w, e in zip(cfg.weights, ELEMENTS) if use[e] is not None]
    if len(present) < MIN_PRESENT or min(c for _, c in present) < cfg.link_min_element_cos:
        return 0.0, cos
    tw = sum(w for w, _ in present)
    return float(sum(w * c for w, c in present) / tw), cos


def attach_vectors(mentions: list, embedder) -> None:
    """Embed SVO elements and narrative-frame fields separately."""
    strings = sorted({getattr(m, e) for m in mentions for e in ELEMENTS})
    if strings:
        vecs = np.asarray(embedder.encode(strings, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        lookup = {s: (np.zeros(vecs.shape[1], dtype=np.float32) if s == "unspecified" else v) for s, v in zip(strings, vecs)}
        for m in mentions:
            m.vec = {e: lookup[getattr(m, e)] for e in ELEMENTS}
    fields = ("claim", "issue", "stakes", "goal", "causal", "evaluation")
    vals = sorted({getattr(m, f) for m in mentions for f in fields if getattr(m, f) and getattr(m, f) != "unspecified"})
    if vals:
        arr = np.asarray(embedder.encode(vals, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        lookup2 = {v: x for v, x in zip(vals, arr)}
        for m in mentions:
            m.frame_vec = {f: lookup2.get(getattr(m, f), np.zeros(arr.shape[1], dtype=np.float32)) for f in fields}


def frame_similarity(a, b, field_name):
    va, vb = a.frame_vec.get(field_name), b.frame_vec.get(field_name)
    return cos_or_none(va, vb) if va is not None and vb is not None else None


def candidate_score(a, b):
    """Recall-oriented similarity for retrieving narrative-related candidates."""
    weights = {"claim": .30, "issue": .25, "stakes": .15, "goal": .10, "causal": .10, "evaluation": .05}
    vals = [(w, frame_similarity(a, b, k)) for k, w in weights.items()]
    vals = [(w, v) for w, v in vals if v is not None]
    svo, _ = prop_sim(a.vec, b.vec, (.25, .25, .50))
    # Actor overlap is deliberately weak: actor changes are common in narrative development.
    actor = cos_or_none(a.vec["actor"], b.vec["actor"])
    if vals:
        frame = sum(w*v for w, v in vals) / sum(w for w, _ in vals)
    else:
        frame = 0.0
    score = .78 * frame + .17 * svo + .05 * (actor if actor is not None else 0.0)
    return float(score), {k: frame_similarity(a, b, k) for k in weights} | {"svo": svo, "actor": actor}


def classify_relationship(llm, earlier, later, score, cfg):
    """Classify narrative relation; topic similarity alone is not a narrative link."""
    allowed = ["continuation", "elaboration", "reaction", "consequence", "reframing", "contradiction",
               "topic_related_not_continuous", "unrelated"]
    if not cfg.llm_link_classification or not llm.ok:
        return ("continuation" if score >= cfg.narrative_link_thr else "topic_related_not_continuous"), "embedding-based fallback"
    prompt = (
        "You are deciding whether a later proposition belongs to the same evolving narrative as an earlier proposition. "
        "Narrative identity is continuity of the underlying storyline, not wording similarity. Topic similarity alone is insufficient. "
        "Actor, action, wording, stance and emotion may change. Assess whether the later claim develops, elaborates, reacts to, "
        "results from, reframes or contradicts the earlier storyline. If they merely concern the same broad topic without a "
        "clear storyline relationship, choose topic_related_not_continuous. Choose unrelated when there is no meaningful relation.\n\n"
        f"Context: {cfg.context}\nEarlier ({earlier.date.date()}, {earlier.doc_id}): {earlier.claim}\n"
        f"Frame: issue={earlier.issue}; stakes={earlier.stakes}; goal={earlier.goal}; causal={earlier.causal}; evaluation={earlier.evaluation}\n\n"
        f"Later ({later.date.date()}, {later.doc_id}): {later.claim}\n"
        f"Frame: issue={later.issue}; stakes={later.stakes}; goal={later.goal}; causal={later.causal}; evaluation={later.evaluation}\n\n"
        f"Candidate similarity: {score:.3f}\nReturn ONLY JSON with keys relation and rationale; relation must be one of: {', '.join(allowed)}."
    )
    try:
        out = parse_json(llm._complete(prompt, 900)) or {}
        rel = str(out.get("relation", "topic_related_not_continuous")).strip().lower()
        if rel not in allowed:
            rel = "topic_related_not_continuous"
        return rel, norm_text(out.get("rationale", ""), 240)
    except Exception as exc:
        log.warning("relationship classification failed: %s", exc)
        return ("continuation" if score >= cfg.narrative_link_thr else "topic_related_not_continuous"), "classifier fallback"


def merge_document(mentions: list, cfg: Config) -> list:
    """Greedy paragraph-by-paragraph merge of matching propositions. Returns list of mention lists."""
    threads, by_par = [], {}
    for m in mentions:
        by_par.setdefault(m.par, []).append(m)
    for par in sorted(by_par):
        cands = []
        for mi, m in enumerate(by_par[par]):
            for ti, t in enumerate(threads):
                if not (par - cfg.merge_window <= t["last"] < par):
                    continue
                last = t["m"][-1]
                s, cos = prop_sim(last.vec, m.vec, cfg.weights)
                if s < cfg.merge_thr or min(c for c in cos.values() if c is not None) < cfg.min_element_cos:
                    continue
                if cfg.merge_require in ("orientation", "both") and last.orientation != m.orientation:
                    continue
                if cfg.merge_require in ("emotion", "both") and last.emotion != m.emotion:
                    continue
                cands.append((s, mi, ti))
        used_m, used_t = set(), set()
        for s, mi, ti in sorted(cands, reverse=True):
            if mi in used_m or ti in used_t:
                continue
            threads[ti]["m"].append(by_par[par][mi])
            threads[ti]["last"] = par
            used_m.add(mi), used_t.add(ti)
        for mi, m in enumerate(by_par[par]):
            if mi not in used_m:
                threads.append({"m": [m], "last": par})
    return [t["m"] for t in threads]


def make_node(doc_rank: int, row, k: int, ms: list, cfg: Config) -> Node:
    vec = {}
    for e in ELEMENTS:
        vs = [m.vec[e] for m in ms if np.linalg.norm(m.vec[e]) > 1e-6]
        if vs:
            v = np.mean(vs, axis=0)
            vec[e] = (v / np.linalg.norm(v)).astype(np.float32)
        else:
            vec[e] = np.zeros_like(ms[0].vec[e])
    if len(ms) > 1:  # medoid mention = most representative wording
        medoid = max(ms, key=lambda a: sum(prop_sim(a.vec, b.vec, cfg.weights)[0] for b in ms))
    else:
        medoid = ms[0]
    def pick(attr):  # most frequent value among the mentions; ties go to the medoid's value
        counts = Counter(getattr(m, attr) for m in ms)
        return max(counts, key=lambda v: (counts[v], v == getattr(medoid, attr)))

    return Node(id=f"{row.doc_id}_P{k}", doc=doc_rank, doc_id=str(row.doc_id), date=row.date, title=str(row.title),
                par_first=min(m.par for m in ms), par_last=max(m.par for m in ms), mentions=ms, vec=vec,
                actor=medoid.actor, action=medoid.action, object=medoid.object,
                orientation=pick("orientation"), emotion=pick("emotion"),
                claim=medoid.claim, evidence=medoid.evidence,
                issue=pick("issue"), stakes=pick("stakes"), goal=pick("goal"), causal=pick("causal"),
                evaluation=pick("evaluation"), frame_vec={f: medoid.frame_vec.get(f) for f in ("claim", "issue", "stakes", "goal", "causal", "evaluation")})


def build_graph(nodes: list, cfg: Config, llm=None) -> nx.DiGraph:
    """Build a temporal narrative graph with multiple typed edges and branch support.

    For corpora above 2,000 propositions, use optional FAISS HNSW retrieval on claim
    embeddings, then rerank retrieved candidates with the full narrative-frame score.
    If FAISS is unavailable, fall back to exact all-history retrieval.
    """
    G = nx.DiGraph()
    ordered = sorted(nodes, key=lambda n: (n.date, n.doc, n.id))
    for n in ordered:
        G.add_node(n.id, node=n)
    calls = 0
    indexes = None
    retrieval_fields = ("claim", "issue", "stakes", "goal")
    if len(ordered) > 2000:
        try:
            import faiss
            dim = len(ordered[0].frame_vec["claim"])
            indexes = {}
            for field_name in retrieval_fields:
                ix = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
                ix.hnsw.efSearch = max(64, cfg.candidate_top_k * 8)
                indexes[field_name] = ix
            log.info("Using FAISS HNSW multi-field retrieval for %d propositions", len(ordered))
        except Exception as exc:
            indexes = None
            log.warning("FAISS unavailable (%s); using exact historical candidate retrieval", exc)
    for idx, later in enumerate(ordered):
        if indexes is not None and idx:
            retrieve_n = min(idx, max(cfg.candidate_top_k * 8, 40))
            retrieved = set()
            for field_name, ix in indexes.items():
                q = np.asarray(later.frame_vec[field_name], dtype=np.float32).reshape(1, -1)
                _, positions = ix.search(q, retrieve_n)
                retrieved.update(int(x) for x in positions[0] if int(x) >= 0)
            candidate_positions = sorted(retrieved)
        else:
            candidate_positions = list(range(idx))
        candidates = []
        for pos in candidate_positions:
            earlier = ordered[pos]
            if earlier.doc >= later.doc:  # narrative edges connect documents, not propositions within one document
                continue
            score, details = candidate_score(earlier, later)
            if score >= cfg.candidate_min_score:
                candidates.append((score, earlier, details))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for score, earlier, details in candidates[:cfg.candidate_top_k]:
            if cfg.max_link_calls and calls >= cfg.max_link_calls:
                rel, rationale = ("continuation" if score >= cfg.narrative_link_thr else "topic_related_not_continuous"), "max_link_calls fallback"
            else:
                rel, rationale = classify_relationship(llm, earlier, later, score, cfg)
                calls += int(cfg.llm_link_classification and llm is not None and llm.ok)
            if rel in {"topic_related_not_continuous", "unrelated"}:
                continue
            G.add_edge(earlier.id, later.id, sim=float(score), cos=details, relation=rel, rationale=rationale,
                       gap=max(1, later.doc-earlier.doc), days=int((later.date-earlier.date).days))
        if indexes is not None:
            for field_name, ix in indexes.items():
                v = np.asarray(later.frame_vec[field_name], dtype=np.float32).reshape(1, -1)
                ix.add(v)
    log.info("Narrative graph: %d nodes, %d typed edges; %d relationship-classifier calls", G.number_of_nodes(), G.number_of_edges(), calls)
    return G


def main_path(G: nx.DiGraph, comp: set) -> list:
    order = sorted(comp, key=lambda i: G.nodes[i]["node"].doc)
    best = {}
    for i in order:
        cands = [(best[p][0] + 1, best[p][1] + G[p][i]["sim"], p) for p in G.predecessors(i) if p in comp]
        best[i] = max(cands) if cands else (1, 0.0, None)
    end = max(order, key=lambda i: best[i][:2])
    path = [end]
    while best[path[-1]][2] is not None:
        path.append(best[path[-1]][2])
    return path[::-1]


# --------------------------------------------------------------------------- #
# Transformation analysis
# --------------------------------------------------------------------------- #
def orientation_status(a: str, b: str) -> str:
    if a == b:
        return "same"
    return "flip" if {a, b} == {"supportive", "critical"} else "shift"


def emotion_status(a: str, b: str) -> str:
    if a == b:
        return "same"
    return "flip" if EMOTION_DIRECTION.get(a, 0) * EMOTION_DIRECTION.get(b, 0) < 0 else "shift"


def compress(seq: list) -> str:
    out = [x for i, x in enumerate(seq) if i == 0 or x != seq[i - 1]]
    return " -> ".join(out)


def describe_narrative(llm: LLM, steps: list, transitions: list, cfg: Config):
    lines = []
    for i, s in enumerate(steps):
        lead = f" [S={transitions[i - 1]['sim']:.2f}; {transitions[i - 1]['label']}]" if i else ""
        lines.append(f"{s['date']}{lead}: {s['actor']} | {s['action']} | {s['object']} | orientation={s['orientation']} "
                     f"| emotion={s['emotion']} | issue={s.get('issue', '')} | stakes={s.get('stakes', '')} | "
                     f"goal={s.get('goal', '')} | causal={s.get('causal', '')} | evaluation={s.get('evaluation', '')} | claim: {s['claim']}")
    prompt = ("These are a temporally ordered dominant path through ONE evolving narrative thread from " + cfg.context + ". "
              "The propositions need not express the same SVO structure. Explain the storyline's development, including "
              "changes in problem definition, actors and agency, stakes, goals, causal attribution, evaluation, orientation "
              "and emotion. Distinguish continuity of the underlying storyline from changes in its framing. Do not invent "
              "developments not supported by the propositions.\n" + "\n".join(lines) +
              '\n\nReturn ONLY JSON: {"title": "max 8 words", "summary": "3-5 sentences describing the narrative development and key transformations"}')
    return parse_json(llm._complete(prompt, 2000)) if llm.ok else None


def build_record(G: nx.DiGraph, path: list, llm: LLM, cfg: Config, extra: int) -> dict:
    nodes = [G.nodes[i]["node"] for i in path]
    steps = [{
        "id": n.id, "doc_id": n.doc_id, "date": n.date.strftime("%Y-%m-%d"), "title": n.title,
        "paragraphs": f"{n.par_first}" if n.par_first == n.par_last else f"{n.par_first}-{n.par_last}",
        "n_mentions": len(n.mentions), "actor": n.actor, "action": n.action, "object": n.object,
        "orientation": n.orientation, "emotion": n.emotion, "claim": n.claim, "evidence": n.evidence[:900],
        "issue": n.issue, "stakes": n.stakes, "goal": n.goal, "causal": n.causal, "evaluation": n.evaluation,
        "mention_trajectory": [{"paragraph": m.par, "actor": m.actor, "action": m.action, "object": m.object,
                                "orientation": m.orientation, "emotion": m.emotion} for m in n.mentions],
    } for n in nodes]

    for n, st in zip(nodes, steps):  # everything measured against DAY 1 (the first step)
        sim0, cos0 = (1.0, {e: 1.0 for e in ELEMENTS}) if n is nodes[0] else prop_sim(nodes[0].vec, n.vec, cfg.weights)
        st["sim_to_start"] = float(sim0)
        st["cos_to_start"] = {k: (None if v is None else float(v)) for k, v in cos0.items()}
        st["days_since_start"] = int((n.date - nodes[0].date).days)
        st["orientation_vs_start"] = orientation_status(nodes[0].orientation, n.orientation)
        st["emotion_vs_start"] = emotion_status(nodes[0].emotion, n.emotion)

    transitions = []
    for a, b, na, nb in zip(path[:-1], path[1:], nodes[:-1], nodes[1:]):
        e = G[a][b]
        # Edge-level "cos" stores narrative-frame candidate-retrieval diagnostics
        # (claim/issue/stakes/goal/causal/evaluation/SVO/actor), not the SVO
        # element dictionary expected by the legacy transformation report.
        # Recompute actor/action/object cosine values from the proposition vectors.
        _, cos = prop_sim(na.vec, nb.vec, cfg.weights)
        status = {}
        for el in ELEMENTS:
            status[el] = ("na" if cos[el] is None else "same" if cos[el] >= cfg.same_thr
                          else "drift" if cos[el] >= cfg.drift_thr else "changed")
        status["orientation"] = orientation_status(na.orientation, nb.orientation)
        status["emotion"] = emotion_status(na.emotion, nb.emotion)
        frame_cos = {f: cos_or_none(na.frame_vec.get(f), nb.frame_vec.get(f))
                     for f in ("claim", "issue", "stakes", "goal", "causal", "evaluation")}
        frame_status = {f: ("na" if v is None else "same" if v >= cfg.same_thr
                            else "drift" if v >= cfg.drift_thr else "changed") for f, v in frame_cos.items()}
        changed = [k for k, v in status.items() if v not in ("same", "na")]
        if status["orientation"] == "flip":
            label = "orientation reversal" + (f" + {', '.join(c for c in changed if c != 'orientation')}"
                                              if len(changed) > 1 else "")
        elif changed:
            label = "transformation: " + ", ".join(changed)
        else:
            label = "stable"
        label = f"{e.get('relation', 'continuation')}: {label}"
        if e["gap"] > 1:
            label += f" (skips {e['gap'] - 1} document(s))"
        transitions.append({"from": a, "to": b, "relation": e.get("relation", "continuation"), "rationale": e.get("rationale", ""), "sim": float(e["sim"]), "cos": {k: (None if v is None else float(v)) for k, v in cos.items()},
                            "status": status, "frame_cos": frame_cos, "frame_status": frame_status, "changed": changed, "days": e["days"], "gap": e["gap"], "label": label})

    T = max(len(transitions), 1)
    notes = []
    for el in ELEMENTS:
        compared = [t for t in transitions if t["status"][el] != "na"]
        n_ch = sum(t["status"][el] == "changed" for t in compared)
        n_dr = sum(t["status"][el] == "drift" for t in compared)
        c_end = cos_or_none(nodes[0].vec[el], nodes[-1].vec[el])
        c_txt = "n/a" if c_end is None else f"{c_end:.2f}"
        hidden = (" Cumulative change not visible in any single step."
                  if c_end is not None and c_end < cfg.drift_thr and n_ch == 0 else "")
        notes.append(f"{el.capitalize()}: compared in {len(compared)} of {T} steps ({n_ch} changed, {n_dr} drifted); "
                     f"start vs end cos {c_txt} ('{steps[0][el]}' -> '{steps[-1][el]}').{hidden}")
    for field_name in ("issue", "stakes", "goal", "causal", "evaluation"):
        vals = [t["frame_status"][field_name] for t in transitions if t["frame_status"][field_name] != "na"]
        start_end = cos_or_none(nodes[0].frame_vec.get(field_name), nodes[-1].frame_vec.get(field_name))
        start_end_txt = "n/a" if start_end is None else f"{start_end:.2f}"
        notes.append(f"Narrative {field_name}: {sum(v == 'changed' for v in vals)} changed and "
                     f"{sum(v == 'drift' for v in vals)} drifted transition(s); start-to-end similarity {start_end_txt}.")
    for el, fn in (("orientation", orientation_status), ("emotion", emotion_status)):
        n_ch = sum(t["status"][el] != "same" for t in transitions)
        n_fl = sum(t["status"][el] == "flip" for t in transitions)
        notes.append(f"{el.capitalize()}: {compress([s[el] for s in steps])} "
                     f"({n_ch} change(s), {n_fl} flip(s)).")
    skipped = [t for t in transitions if t["gap"] > 1]
    if skipped:
        notes.append("Interrupted: " + "; ".join(
            f"{t['gap'] - 1} document(s) without it between {steps[i]['date']} and {steps[i + 1]['date']}"
            for i, t in enumerate(transitions) if t["gap"] > 1) + ".")
    if extra:
        notes.append(f"{extra} further proposition(s) belong to this narrative but lie off the main path.")
    span = (nodes[-1].date - nodes[0].date).days
    notes.append(f"Span: {steps[0]['date']} to {steps[-1]['date']} ({span} days), {len(steps)} documents, "
                 f"{sum(s['n_mentions'] for s in steps)} paragraph mentions.")

    desc = describe_narrative(llm, steps, transitions, cfg) or {}
    return {
        "title": str(desc.get("title") or f"{steps[0]['actor']}: {steps[0]['action']} {steps[0]['object']}"),
        "summary": str(desc.get("summary") or steps[0]["claim"]),
        "steps": steps, "transitions": transitions, "notes": notes,
        "flags": {"orientation_reversal": any(t["status"]["orientation"] == "flip" for t in transitions),
                  "emotion_valence_flip": any(t["status"]["emotion"] == "flip" for t in transitions),
                  "interrupted": bool(skipped),
                  "elements_changed": sorted({k for t in transitions for k in t["changed"]})},
    }


# --------------------------------------------------------------------------- #
# Figure: elements (rows) x time (columns)
# --------------------------------------------------------------------------- #
def wrap(s: str, width: int, max_lines: int) -> list:
    lines = textwrap.wrap(str(s), width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,;:") + " ..."
    return lines


def render_page(rec: dict, lo: int, hi: int, path: Path, cfg: Config) -> None:
    steps, trans = rec["steps"], rec["transitions"]
    cols = list(range(lo, hi + 1))
    k, GRAY = len(cols), "#6b6b66"
    LBL, COLW, GAP = 1.45, 2.3, 0.34
    X0 = 0.35 + LBL
    W = X0 + k * COLW + (k - 1) * GAP + 0.35
    rows = [("claim", "Claim", 1.4), ("actor", "Actor", 0.62), ("action", "Action", 0.62),
            ("object", "Object", 0.62), ("orientation", "Orientation", 0.48), ("emotion", "Emotion", 0.48),
            ("change", "Change", 0.9)]
    summary = wrap(rec["summary"], int((W - 0.8) / 0.082), 6)
    notes = [wrap(n, int((W - 1.0) / 0.07), 2) for n in rec["notes"]]
    head = 0.35 + 0.45 + 0.21 * len(summary) + 0.5
    foot = 0.6 + sum(0.18 * len(n) + 0.08 for n in notes)
    H = head + 0.85 + sum(r[2] + 0.08 for r in rows) + foot + 0.2

    fig = plt.figure(figsize=(W, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(0, H), ax.axis("off")

    y = H - 0.3
    ax.text(0.35, y, rec["title"], fontsize=17, weight="bold", va="top")
    y -= 0.45
    for line in summary:
        ax.text(0.35, y, line, fontsize=10.5, color=GRAY, va="top")
        y -= 0.21
    y -= 0.08
    legend = [("unchanged", "same"), ("drifted/shift", "drift"), ("changed/flip", "changed"),
              ("first step", "start"), ("not specified", "na")]
    dx = (W - 0.7) / len(legend)
    for i, (lab, st_) in enumerate(legend):
        ax.scatter([0.45 + i * dx], [y - 0.1], s=40, color=STATUS_COLOR[st_])
        ax.text(0.58 + i * dx, y - 0.1, lab, fontsize=8, color=GRAY, va="center")
    y -= 0.4

    xs = [X0 + j * (COLW + GAP) for j in range(k)]
    for j, g in enumerate(cols):  # column headers
        s = steps[g]
        cx = xs[j] + COLW / 2
        ax.text(cx, y - 0.04, s["date"], ha="center", va="top", fontsize=11, weight="bold")
        ax.text(cx, y - 0.27, "\n".join(wrap(s["title"], 32, 2)), ha="center", va="top", fontsize=8.5, color=GRAY)
        ax.text(cx, y - 0.68, f"paragraph {s['paragraphs']}  |  {s['n_mentions']} mention(s)", ha="center", va="top",
                fontsize=7.5, color=GRAY)
    y -= 0.85

    for key, label, h in rows:
        ax.text(0.35, y - h / 2, label, fontsize=9.5, weight="bold", color=GRAY, va="center")
        for j, g in enumerate(cols):
            s, t = steps[g], (trans[g - 1] if g > 0 else None)
            ls, ec, lw = "-", "#cfcfc8", 1.2
            if key == "claim":
                txt, fs, style = '"' + "\n".join(wrap(s["claim"], 31, 7)) + '"', 8.5, "italic"
            elif key == "change":
                if t is None:
                    txt, fs, style, ec, lw = "first step", 8.5, "normal", STATUS_COLOR["start"], 2
                else:
                    ec, lw = (STATUS_COLOR["same"] if t["sim"] >= cfg.merge_thr else STATUS_COLOR["drift"]), 2
                    ls = "--" if t["gap"] > 1 else "-"
                    txt = f"S = {t['sim']:.2f}   ({t['days']:+d} d)\n" + "\n".join(wrap(t["label"], 31, 4))
                    fs, style = 8.5, "normal"
            else:
                st = "start" if t is None else t["status"][key]
                ec, lw = STATUS_COLOR[st], 2.2
                txt = "\n".join(wrap(s[key].capitalize() if key in ("orientation", "emotion") else s[key], 28, 3))
                fs, style = 9.5, "normal"
            ax.add_patch(FancyBboxPatch((xs[j], y - h), COLW, h, boxstyle="round,pad=0,rounding_size=0.1",
                                        fc="white", ec=ec, lw=lw, ls=ls))
            ax.text(xs[j] + COLW / 2, y - h / 2, txt, ha="center", va="center", fontsize=fs, style=style,
                    linespacing=1.25)
            if key == "claim" and j < k - 1:
                nt = trans[g]
                col = STATUS_COLOR["same"] if nt["sim"] >= cfg.merge_thr else STATUS_COLOR["drift"]
                ax.annotate("", xy=(xs[j + 1] - 0.02, y - h / 2), xytext=(xs[j] + COLW + 0.02, y - h / 2),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=2, ls="--" if nt["gap"] > 1 else "-"))
        y -= h + 0.08

    y -= 0.12
    ax.plot([0.35, W - 0.35], [y, y], color="#d8d8d2", lw=0.8)
    y -= 0.2
    ax.text(0.35, y, "Transformation summary", fontsize=11, weight="bold", va="top")
    y -= 0.34
    for n in notes:
        ax.text(0.4, y, "\u25CF", fontsize=7, va="top", color=GRAY)
        ax.text(0.65, y, "\n".join(n), fontsize=9, va="top", linespacing=1.3)
        y -= 0.18 * len(n) + 0.08
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_figure(rec: dict, prefix: Path, cfg: Config) -> list:
    n, m = len(rec["steps"]), cfg.max_steps
    spans = [(0, n - 1)] if n <= m else [(i, min(i + m - 1, n - 1)) for i in range(0, n - 1, m - 1)]
    paths = []
    for pi, (lo, hi) in enumerate(spans, 1):
        p = Path(f"{prefix}.png") if len(spans) == 1 else Path(f"{prefix}_part{pi}.png")
        render_page(rec, lo, hi, p, cfg)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def process_dataset(path: Path, args, cfg: Config, nlp, llm: LLM, embedder, out_root: Path) -> None:
    out = out_root / path.stem
    out.mkdir(parents=True, exist_ok=True)
    log.info("=== Dataset %s ===", path.name)
    docs = load_documents(path)
    if cfg.max_docs:
        docs = docs.head(cfg.max_docs)

    mentions, par_rows = [], []
    for rank, row in enumerate(docs.itertuples()):
        pars = split_paragraphs(row.text, nlp, cfg)
        log.info("[%d/%d] %s: %d paragraphs", rank + 1, len(docs), row.doc_id, len(pars))
        for pi, ptxt in enumerate(pars, 1):
            sao = extract_sao([ptxt], nlp, cfg.lang) if cfg.extractor == "llm" and llm.ok else {}
            desc, props = analyse_paragraph(llm, ptxt, sao, cfg, nlp)
            par_rows.append({"doc_id": row.doc_id, "date": row.date.date(), "paragraph": pi, "description": desc,
                             "svo_hint": json.dumps(sao, ensure_ascii=False), "n_propositions": len(props),
                             "text": ptxt})
            mentions += [Mention(doc=rank, par=pi, evidence=ptxt, **p) for p in props]
    pd.DataFrame(par_rows).to_csv(out / "paragraphs.csv", index=False)
    if not mentions:
        log.error("%s: no propositions extracted (is Ollama running? is spaCy installed?)", path.name)
        return
    attach_vectors(mentions, embedder)
    pd.DataFrame([{"doc": docs.doc_id[m.doc], "paragraph": m.par, "actor": m.actor, "action": m.action,
                   "object": m.object, "orientation": m.orientation, "emotion": m.emotion, "claim": m.claim, "issue": m.issue, "stakes": m.stakes, "goal": m.goal, "causal": m.causal, "evaluation": m.evaluation}
                  for m in mentions]).to_csv(out / "mentions.csv", index=False)

    nodes, by_doc = [], {}
    for m in mentions:
        by_doc.setdefault(m.doc, []).append(m)
    for rank, row in enumerate(docs.itertuples()):
        ms = by_doc.get(rank, [])
        for k, thread in enumerate(merge_document(ms, cfg), 1):
            nodes.append(make_node(rank, row, k, thread, cfg))
    log.info("%d mentions merged into %d document-level propositions", len(mentions), len(nodes))
    pd.DataFrame([{"id": n.id, "doc_id": n.doc_id, "date": n.date.date(), "paragraphs": f"{n.par_first}-{n.par_last}",
                   "n_mentions": len(n.mentions), "actor": n.actor, "action": n.action, "object": n.object,
                   "orientation": n.orientation, "emotion": n.emotion, "claim": n.claim, "issue": n.issue, "stakes": n.stakes, "goal": n.goal, "causal": n.causal, "evaluation": n.evaluation}
                  for n in nodes]).to_csv(out / "propositions.csv", index=False)

    G = build_graph(nodes, cfg, llm)
    pd.DataFrame([{"src": a, "dst": b, "sim": d["sim"], **{f"cos_{k}": v for k, v in d["cos"].items()},
                   "gap": d["gap"], "days": d["days"]} for a, b, d in G.edges(data=True)]).to_csv(
        out / "edges.csv", index=False)

    # Full narrative graph is the primary result; rendered paths below are convenience views.
    graph_nodes = []
    for nid, data in G.nodes(data=True):
        n = data["node"]
        graph_nodes.append({"id": nid, "doc_id": n.doc_id, "date": n.date.date(), "claim": n.claim,
                            "issue": n.issue, "stakes": n.stakes, "goal": n.goal, "causal": n.causal,
                            "evaluation": n.evaluation, "actor": n.actor, "action": n.action, "object": n.object,
                            "orientation": n.orientation, "emotion": n.emotion})
    pd.DataFrame(graph_nodes).to_csv(out / "narrative_graph_nodes.csv", index=False)
    graph_edges = [{"src": a, "dst": b, "similarity": d["sim"], "relation": d["relation"],
                    "rationale": d["rationale"], "gap": d["gap"], "days": d["days"],
                    **{f"sim_{k}": v for k, v in d["cos"].items()}} for a, b, d in G.edges(data=True)]
    pd.DataFrame(graph_edges).to_csv(out / "narrative_graph_edges.csv", index=False)
    thread_rows = []
    for ti, comp in enumerate(sorted(nx.weakly_connected_components(G), key=lambda c: (-len(c), min(G.nodes[x]["node"].date for x in c))), 1):
        ns = sorted((G.nodes[x]["node"] for x in comp), key=lambda n: (n.date, n.id))
        thread_rows.append({"narrative_id": f"thread_{ti:04d}", "n_propositions": len(ns),
                            "n_documents": len({n.doc_id for n in ns}), "start": ns[0].date.date(),
                            "end": ns[-1].date.date(), "proposition_ids": " | ".join(n.id for n in ns),
                            "claims": " || ".join(n.claim for n in ns)})
    pd.DataFrame(thread_rows).to_csv(out / "narrative_threads.csv", index=False)

    ndocs = lambda c: len({G.nodes[i]["node"].doc for i in c})  # noqa: E731
    comps = sorted(nx.weakly_connected_components(G), key=lambda c: (-ndocs(c), -len(c)))
    isolated = [G.nodes[next(iter(c))]["node"] for c in comps if len(c) == 1]
    rows, rendered, identical = [], 0, []
    for comp in comps:
        if len(comp) == 1 or rendered >= cfg.top_n:
            continue
        ids = main_path(G, comp)
        if len(ids) < cfg.min_path_len:
            continue
        if cfg.skip_identical and all(G[a_][b_]["sim"] >= cfg.identical_thr for a_, b_ in zip(ids, ids[1:])):
            n0, n1 = G.nodes[ids[0]]["node"], G.nodes[ids[-1]]["node"]
            identical.append({"n_documents": len(ids), "start": n0.date.date(), "end": n1.date.date(), "actor": n0.actor,
                              "action": n0.action, "object": n0.object, "claim": n0.claim})
            continue  # zero-change chain: listed in identical_chains.csv, does not use up a top_n slot
        rendered += 1
        rec = build_record(G, ids, llm, cfg, extra=len(comp) - len(ids))
        stem = f"narrative_{rendered:02d}"
        render_figure(rec, out / stem, cfg)
        (out / f"{stem}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        rows.append({"narrative": stem, "title": rec["title"], "start": rec["steps"][0]["date"],
                     "end": rec["steps"][-1]["date"], "n_documents": len(rec["steps"]), **rec["flags"]})
        log.info("Wrote %s/%s", out.name, stem)
    pd.DataFrame(rows).to_csv(out / "narratives.csv", index=False)
    if identical:
        pd.DataFrame(identical).to_csv(out / "identical_chains.csv", index=False)
        log.info("%d zero-change chains (verbatim repeats) were skipped; see identical_chains.csv", len(identical))
    (out / "config.json").write_text(json.dumps({"same_thr": cfg.same_thr, "drift_thr": cfg.drift_thr,
                                                 "merge_thr": cfg.merge_thr, "link_thr": cfg.link_thr,
                                                 "link_min_element_cos": cfg.link_min_element_cos, "candidate_top_k": cfg.candidate_top_k,
                                                 "candidate_min_score": cfg.candidate_min_score, "narrative_link_thr": cfg.narrative_link_thr,
                                                 "llm_link_classification": cfg.llm_link_classification,
                                                 "link_ignore_pronoun_actor": cfg.link_ignore_pronoun_actor,
                                                 "emotion_scheme": "Jasper (1998) Table I", "emotion_labels": EMOTIONS}, indent=2))
    build_viewer.build(out)  # viewer.html + narrative_trajectories.csv + narrative_summary.csv
    pd.DataFrame([{"id": n.id, "date": n.date.date(), "claim": n.claim, "n_mentions": len(n.mentions)}
                  for n in isolated]).to_csv(out / "isolated_propositions.csv", index=False)
    log.info("%s done: %d narratives rendered, %d single-document propositions", path.name, rendered, len(isolated))
    if UNRECOGNISED_EMOTIONS:
        log.warning("emotion answers not in Jasper's list (counted as neutral): %s",
                    dict(UNRECOGNISED_EMOTIONS.most_common(15)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", required=True, help="datasets with `full_text` and `date` columns")
    p.add_argument("--out", default="proposition_out")
    p.add_argument("--lang", default="en", choices=sorted(SPACY_MODELS), help="language of the source documents")
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="embeds the (English) proposition elements")
    g = p.add_argument_group("LLM (Ollama)")
    g.add_argument("--llm-model", default="gpt-oss:20b")
    g.add_argument("--ollama-host", default="http://localhost:11434")
    g.add_argument("--ollama-think", default="low", choices=["low", "medium", "high"])
    g.add_argument("--extractor", default="llm", choices=["llm", "spacy"],
                   help="llm: propositions, orientation, emotion from the LLM (default). spacy: propositions from the "
                        "dependency parse, no LLM calls at all (orientation/emotion neutral, narrative titles generic)")
    g.add_argument("--no-llm", action="store_true", help="never call the LLM; same extraction as --extractor spacy")
    d = Config()
    g = p.add_argument_group("Paragraphs")
    g.add_argument("--context", default=d.context,
                   help="who wrote the documents, inserted in the LLM prompts, e.g. \"a climate-movement organisation (Fridays for Future Austria)\"; "
                        "changing it invalidates cached LLM answers")
    g.add_argument("--max-sentences", type=int, default=d.max_sentences, help="max sentences per paragraph chunk")
    g.add_argument("--min-chars", type=int, default=d.min_chars)
    g.add_argument("--max-props", type=int, default=d.max_props, help="propositions per paragraph")
    g.add_argument("--max-docs", type=int, default=d.max_docs, help="process only the first N documents (dry run)")
    g.add_argument("--force-english", action="store_true",
                   help="tell the LLM to translate leftover German words (invalidates cached paragraph answers)")
    g = p.add_argument_group("Similarity thresholds")
    g.add_argument("--merge-thr", type=float, default=d.merge_thr, help="same proposition inside a document")
    g.add_argument("--merge-window", type=int, default=d.merge_window, help="paragraphs a proposition may skip")
    g.add_argument("--merge-require", default=d.merge_require, choices=["none", "orientation", "emotion", "both"],
                   help="also require equal orientation / emotion to merge")
    g.add_argument("--min-element-cos", type=float, default=d.min_element_cos)
    g.add_argument("--link-thr", type=float, default=d.link_thr, help="legacy compatibility option; use --candidate-min-score and --narrative-link-thr")
    g.add_argument("--max-gap", type=int, default=d.max_gap, help="legacy compatibility option; narrative retrieval can reconnect after long gaps")
    g.add_argument("--link-min-element-cos", type=float, default=d.link_min_element_cos,
                   help="deprecated compatibility option; narrative-aware linking no longer applies a per-element SVO gate")
    g.add_argument("--candidate-top-k", type=int, default=d.candidate_top_k, help="historical candidates classified per proposition")
    g.add_argument("--candidate-min-score", type=float, default=d.candidate_min_score, help="minimum recall-oriented retrieval score")
    g.add_argument("--narrative-link-thr", type=float, default=d.narrative_link_thr, help="fallback threshold when relationship LLM is unavailable")
    g.add_argument("--no-link-classifier", action="store_true", help="use embedding-based fallback rather than LLM relationship classification")
    g.add_argument("--max-link-calls", type=int, default=d.max_link_calls, help="cap relationship LLM calls; 0 = unlimited")
    g.add_argument("--keep-pronoun-actor", action="store_true",
                   help="let a matching first-person actor (we/I) count towards linking (old behaviour)")
    g.add_argument("--same-thr", type=float, default=d.same_thr, help="element cosine >= this = unchanged")
    g.add_argument("--drift-thr", type=float, default=d.drift_thr, help="element cosine >= this = drifted")
    g.add_argument("--weights", default="0.25,0.25,0.5", help="actor,action,object weights")
    g = p.add_argument_group("Output")
    g.add_argument("--min-path-len", type=int, default=d.min_path_len)
    g.add_argument("--top-n", type=int, default=d.top_n)
    g.add_argument("--keep-identical", action="store_true",
                   help="also render chains made only of verbatim repeats (default: skip, list in identical_chains.csv)")
    g.add_argument("--max-steps", type=int, default=d.max_steps, help="steps per figure page")
    return p.parse_args(argv)


def config_from_args(a) -> Config:
    w = tuple(float(x) for x in a.weights.split(","))
    if len(w) != 3 or abs(sum(w) - 1) > 1e-6:
        raise SystemExit("--weights needs three numbers summing to 1, e.g. 0.25,0.25,0.5")
    cfg = Config(lang=a.lang, max_sentences=a.max_sentences, min_chars=a.min_chars, max_props=a.max_props,
                 merge_thr=a.merge_thr, merge_window=a.merge_window, merge_require=a.merge_require,
                 min_element_cos=a.min_element_cos, link_thr=a.link_thr, max_gap=a.max_gap, same_thr=a.same_thr,
                 link_min_element_cos=a.link_min_element_cos, link_ignore_pronoun_actor=not a.keep_pronoun_actor,
                 drift_thr=a.drift_thr, weights=w, min_path_len=a.min_path_len, top_n=a.top_n, skip_identical=not a.keep_identical,
                 max_steps=a.max_steps, max_docs=a.max_docs, force_english=a.force_english, context=a.context,
                 extractor="spacy" if a.no_llm else a.extractor, candidate_top_k=a.candidate_top_k,
                 candidate_min_score=a.candidate_min_score, narrative_link_thr=a.narrative_link_thr,
                 llm_link_classification=not a.no_link_classifier, max_link_calls=a.max_link_calls)
    if not cfg.drift_thr <= cfg.same_thr:
        raise SystemExit("--drift-thr must be <= --same-thr")
    return cfg


def run(args) -> None:
    cfg = config_from_args(args)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(args.embedding_model)
    nlp = load_spacy(SPACY_MODELS[cfg.lang])
    if cfg.extractor == "spacy" and nlp is None:
        raise SystemExit(f"--extractor spacy needs spaCy: pip install spacy && python -m spacy download {SPACY_MODELS[cfg.lang]}")
    llm = LLM(args.llm_model, args.ollama_host, out_root / "llm_cache.json", enabled=cfg.extractor == "llm",
              think=args.ollama_think)
    for path in args.input:
        process_dataset(Path(path), args, cfg, nlp, llm, embedder, out_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(parse_args())
