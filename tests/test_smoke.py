"""End-to-end smoke test of proposition_tracker.process_dataset with a stub LLM and a stub embedder.

No Ollama, no spaCy, no model downloads: it checks that merging, linking, figures, JSON and viewer.html are written.
It says nothing about the quality of real LLM output.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import proposition_tracker as pt  # noqa: E402


class StubEmbedder:
    """Bag-of-words hashing: strings sharing words get a high cosine, identical strings cosine 1."""
    def encode(self, strings, **_):
        out = []
        for s in strings:
            v = np.zeros(64, dtype=np.float32)
            for w in s.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.array(out)


class StubLLM:
    ok = True

    def _complete(self, prompt, max_tokens=0):
        if "successive forms" in prompt:
            return json.dumps({"title": "Tram line stance", "summary": "The council plan changes over time."})
        late = "BETA" in prompt
        return json.dumps({"description": "The paragraph discusses the tram line.", "propositions": [{
            "actor": "city council", "action": "extend",
            "object": "tram line" if not late else "tram line extension",
            "orientation": "critical" if late else "supportive", "emotion": "anger" if late else "joy_hope",
            "claim": "The council extends the tram line." if not late else "The council extends the tram line later."}]})


def test_pipeline_end_to_end(tmp_path):
    para = "ALPHA text about the tram line and the council plan. It goes on for a while so that it is long enough. "
    rows = [{"id": f"d{i}", "date": f"2026-0{i + 1}-10", "title": f"Doc {i}",
             "full_text": (para if i < 2 else para.replace("ALPHA", "BETA")) * 2} for i in range(4)]
    csv = tmp_path / "corpus.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    cfg = pt.Config(lang="en", link_thr=0.5, link_min_element_cos=0.3, skip_identical=False)
    out = tmp_path / "out"
    out.mkdir()
    pt.process_dataset(csv, None, cfg, None, StubLLM(), StubEmbedder(), out)

    d = out / "corpus"
    for name in ("paragraphs.csv", "mentions.csv", "propositions.csv", "edges.csv", "narratives.csv",
                 "narrative_01.json", "narrative_01.png", "viewer.html", "narrative_summary.csv"):
        assert (d / name).exists(), name
    rec = json.loads((d / "narrative_01.json").read_text())
    assert len(rec["steps"]) == 4
    assert rec["flags"]["orientation_reversal"] is True                # supportive -> critical
    assert "id" in pd.read_csv(d / "propositions.csv").columns         # analyze_corpus.py needs it
