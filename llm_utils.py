"""
llm_utils.py - helpers shared by proposition_tracker.py

  * load_spacy / extract_sao : rule-based subject-verb-object hints (English and German)
  * LLM                      : thin, cached Ollama client (swap `_complete` to use another backend)
  * parse_json               : pull the first JSON object out of an LLM answer

Extracted from narrative_drift.py (the tweet / BERTopic pipeline), which is not needed here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path

log = logging.getLogger("llm_utils")


SKIP_VERBS = {"be", "have", "do", "sein", "haben", "werden"}
PRONOUNS = {"i", "we", "they", "you", "ich", "wir", "ihr", "sie", "du"}
# dependency labels per language: English (UD/ClearNLP) and German (TIGER)
DEPS = {
    "en": dict(subj={"nsubj"}, obj={"dobj", "obj", "attr", "dative", "nsubjpass"}, prep={"prep"}, pobj={"pobj"}),
    "de": dict(subj={"sb"}, obj={"oa", "da", "og", "pd"}, prep={"mo", "op"}, pobj={"nk"}),
}


def load_spacy(model="en_core_web_sm"):
    try:
        import spacy

        nlp = spacy.load(model, disable=["ner"])
        nlp.max_length = 3_000_000
        return nlp
    except Exception as exc:  # noqa: BLE001
        log.warning("spaCy unavailable (%s). Run: python -m spacy download %s", exc, model)
        return None


def _phrase(tok) -> str:
    parts = [c.lemma_.lower() for c in tok.children if c.dep_ == "compound"]
    return " ".join(parts + [tok.lemma_.lower()])


def extract_sao(texts, nlp, lang="en") -> dict:
    """Most frequent agents, actions and objects across texts (rule-based, dependency parse)."""
    if nlp is None:
        return {}
    D = DEPS.get(lang, DEPS["en"])
    agents, verbs, objects, triples = Counter(), Counter(), Counter(), Counter()
    for doc in nlp.pipe(texts, batch_size=32):
        for tok in doc:
            if tok.pos_ != "VERB" or tok.lemma_.lower() in SKIP_VERBS:
                continue
            subj = [_phrase(c) for c in tok.children
                    if c.dep_ in D["subj"] and (c.pos_ in ("NOUN", "PROPN") or c.text.lower() in PRONOUNS)]
            obj_toks = [c for c in tok.children if c.dep_ in D["obj"]]
            for p in (c for c in tok.children if c.dep_ in D["prep"] and (lang == "en" or c.pos_ == "ADP")):
                obj_toks += [c for c in p.children if c.dep_ in D["pobj"]]
            obj = [_phrase(o) for o in obj_toks if o.pos_ in ("NOUN", "PROPN")]
            v = tok.lemma_.lower()
            if not (subj or obj):
                continue
            verbs[v] += 1
            agents.update(subj)
            objects.update(obj)
            if subj:
                triples[(subj[0], v, obj[0] if obj else "")] += 1
    top = lambda c, k=3: [w for w, _ in c.most_common(k)]  # noqa: E731
    return {
        "agents": top(agents), "actions": top(verbs), "objects": top(objects),
        "top_triple": list(triples.most_common(1)[0][0]) if triples else [],
    }


def parse_json(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


class LLM:
    def __init__(self, model: str, host: str, cache_path: Path, enabled: bool = True, think: str = "low"):
        self.model, self.cache_path, self.think, self.client = model, cache_path, think, None
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        if enabled:
            try:
                import ollama

                self.client = ollama.Client(host=host)
                self.client.show(model)  # raises if the server is down or the model is not pulled
            except Exception as exc:  # noqa: BLE001
                log.warning("LLM disabled (%s). Is `ollama serve` running and `ollama pull %s` done?", exc, model)
                self.client = None

    @property
    def ok(self) -> bool:
        return self.client is not None

    def _complete(self, prompt: str, max_tokens=2000):
        key = hashlib.sha1((self.model + prompt).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        kwargs = dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                      options={"temperature": 0.2, "seed": 42, "num_predict": max_tokens, "num_ctx": 8192})
        try:
            try:  # gpt-oss supports reasoning effort: low / medium / high
                r = self.client.chat(think=self.think, **kwargs)
            except TypeError:  # older ollama-python without `think`
                r = self.client.chat(**kwargs)
            out = r["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM call failed: %s", exc)
            return None
        if not out or not out.strip():  # reasoning consumed the whole token budget
            log.warning("LLM returned empty content; try --ollama-think low")
            return None
        self.cache[key] = out
        self.cache_path.write_text(json.dumps(self.cache))
        return out
