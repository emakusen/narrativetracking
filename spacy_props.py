"""
spacy_props.py - rule-based proposition extraction (no LLM)

One proposition per verb with a usable subject or object, read off the dependency parse:

    actor  = grammatical subject (or the "by"-agent of a passive), canonicalised: lemma + compounds/adjectives
    action = verb lemma (+ particle), prefixed with "not" when negated
    object = direct / prepositional object (or the subject of a passive), "X of Y" kept together
    claim  = the sentence the proposition comes from

What it does NOT give you: orientation and emotion. They stay 'neutral', because they cannot be read off the
syntax. Use the LLM extractor (the default in proposition_tracker.py) when you need them.

Handled: conjoined verbs and "want to X" (they inherit the subject), "it/they/er/sie" in a subordinate clause
(resolved to the subject of the main clause), negation, English passives with a "by"-agent.
German uses the TIGER labels; passives and negation are English-only for now.
"""
from __future__ import annotations

import re

from llm_utils import DEPS, PRONOUNS, SKIP_VERBS

THIRD_PERSON = {"it", "they", "he", "she", "er", "sie", "es"}  # replaced by the subject of the governing clause
MAX_PHRASE_WORDS = 4


SKIP_MODIFIERS = {"few", "many", "more", "most", "much", "other", "several", "own", "same", "next", "first", "last",
                  "new", "further", "whole", "such", "some", "any", "all"}


def _phrase(tok) -> str:
    """lemma with its compound / adjective modifiers: 'young activists' -> 'young activist',
    'tram line extension' -> 'tram line extension' (compounds are followed recursively)."""
    left = [c for c in tok.children if c.i < tok.i and (
        (c.dep_ == "compound" and c.pos_ in ("NOUN", "PROPN")) or
        (c.dep_ == "amod" and c.pos_ == "ADJ" and c.lemma_.lower() not in SKIP_MODIFIERS))]
    words = [_phrase(c) if c.dep_ == "compound" else c.lemma_.lower() for c in left] + [tok.lemma_.lower()]
    return " ".join(" ".join(words).split()[-MAX_PHRASE_WORDS:])


def _noun_phrase(tok) -> str:
    """_phrase plus one 'of'-complement: 'extension of tram line'."""
    out = _phrase(tok)
    for p in tok.children:
        if p.dep_ in ("prep", "mo") and p.lemma_.lower() in ("of", "von") and p.pos_ == "ADP":
            for o in p.children:
                if o.dep_ in ("pobj", "nk") and o.pos_ in ("NOUN", "PROPN"):
                    return f"{out} of {_phrase(o)}"
    return out


def _is_actor(tok) -> bool:
    return tok.pos_ in ("NOUN", "PROPN") or tok.text.lower() in PRONOUNS or tok.text.lower() in THIRD_PERSON


def _inherit(verb):
    """Verb that shares its subject with a governing verb: 'wants to extend', 'plans and builds',
    German 'hat ... gestrichen' (subject on the auxiliary) and 'fordern und organisieren' (verb -cj-> und -cd-> verb)."""
    if verb.dep_ not in ("conj", "xcomp", "cj", "oc"):
        return None
    head = verb.head
    if verb.dep_ == "cj" and head.dep_ == "cd":
        head = head.head
    return head if head is not verb and head.pos_ in ("VERB", "AUX") else None


def _subject(verb, D, depth=0):
    subs = [c for c in verb.children if c.dep_ in D["subj"] and _is_actor(c)]
    if subs:
        s = subs[0]
        if s.text.lower() in THIRD_PERSON and verb.dep_ in ("ccomp", "advcl", "xcomp") and depth < 3:
            outer = _subject(verb.head, D, depth + 1)  # "the council announced that it will extend ..."
            if outer is not None and outer.text.lower() not in THIRD_PERSON:
                return outer
        return s
    parent = _inherit(verb)
    return _subject(parent, D, depth + 1) if parent is not None and depth < 3 else None


def _objects(verb, D, lang):
    toks = [c for c in verb.children if c.dep_ in D["obj"] and c.pos_ in ("NOUN", "PROPN")]
    for p in (c for c in verb.children if c.dep_ in D["prep"] and (lang == "en" or c.pos_ == "ADP")):
        if p.lemma_.lower() in ("by", "von") and lang == "en":
            continue  # the passive agent is the actor, not the object
        toks += [o for o in p.children if o.dep_ in D["pobj"] and o.pos_ in ("NOUN", "PROPN")]
    return toks


def _action(verb) -> str:
    parts = [verb.lemma_.lower()] + [c.lemma_.lower() for c in verb.children if c.dep_ in ("prt", "svp")]
    neg = any(c.dep_ in ("neg", "ng") for c in verb.children)
    return ("not " if neg else "") + " ".join(parts)


def extract_propositions(text: str, nlp, lang: str = "en", max_props: int = 3) -> list:
    """[{'actor','action','object','claim'}, ...] for one paragraph, best first (both subject and object found)."""
    if nlp is None or not text.strip():
        return []
    D = DEPS.get(lang, DEPS["en"])
    found = []
    for sent in nlp(text).sents:
        sent_txt = re.sub(r"\s+", " ", sent.text).strip()
        for verb in sent:
            if verb.pos_ != "VERB" or verb.lemma_.lower() in SKIP_VERBS:
                continue
            subj = _subject(verb, D)
            objs = _objects(verb, D, lang)
            if lang == "en":
                if any(c.dep_ == "nsubjpass" for c in verb.children):  # "the line was cancelled by the council"
                    # DEPS["en"]["obj"] already lists nsubjpass, so the patient is in `objs`; the actor is the by-agent
                    agents = [o for p in verb.children if p.dep_ == "agent" for o in p.children
                              if o.dep_ == "pobj" and _is_actor(o)]
                    subj = agents[0] if agents else None
            if subj is None and not objs:
                continue
            found.append({
                "actor": _phrase(subj) if subj is not None else "",
                "action": _action(verb),
                "object": _noun_phrase(objs[0]) if objs else "",
                "claim": sent_txt,
                "_score": (subj is not None) + bool(objs),
                "_pos": verb.i,
            })
    found.sort(key=lambda p: (-p["_score"], p["_pos"]))
    out, seen = [], set()
    for p in found:
        key = (p["actor"], p["action"], p["object"])
        if key in seen:
            continue
        seen.add(key)
        out.append({k: v for k, v in p.items() if not k.startswith("_")})
        if len(out) >= max_props:
            break
    return out
