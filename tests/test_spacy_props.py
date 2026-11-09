"""Tests for the rule-based extractor. Skipped when the spaCy model is not installed."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

spacy = pytest.importorskip("spacy")
try:
    NLP = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    pytest.skip("en_core_web_sm not installed", allow_module_level=True)

from spacy_props import extract_propositions  # noqa: E402


def triples(text, n=5):
    return [(p["actor"], p["action"], p["object"]) for p in extract_propositions(text, NLP, "en", n)]


def test_simple_svo_with_compounds():
    assert ("city council", "cut", "budget") in triples("The city council has cut the budget.")


def test_compound_chain_and_of_complement():
    assert ("we", "accept", "cancellation of tram line extension") in triples(
        "We refuse to accept the cancellation of the tram line extension.")


def test_passive_agent_becomes_actor():
    assert ("city council", "cancel", "tram line") in triples("The tram line was cancelled by the city council.")


def test_negation_is_kept():
    assert ("council", "not extend", "tram line") in triples("The council does not extend the tram line.")


def test_pronoun_resolved_to_main_clause_subject():
    assert ("city council", "extend", "tram line") in triples(
        "The city council has announced that it will extend the tram line to the northern districts.")


def test_conjoined_verbs_share_subject():
    t = triples("Our volunteers organise repair cafes and run a tool library.")
    assert ("volunteer", "organise", "repair cafe") in t and ("volunteer", "run", "tool library") in t


def test_max_props_and_claim_is_the_sentence():
    ps = extract_propositions("The council cut the budget. The mayor opened the park. We called the office.", NLP, "en", 2)
    assert len(ps) == 2 and all(p["claim"].endswith(".") for p in ps)


def test_pipeline_runs_without_llm(tmp_path):
    import pandas as pd
    from test_smoke import StubEmbedder
    import proposition_tracker as pt
    from llm_utils import LLM

    txt = ("The city council extends the tram line to the northern districts. We welcome the extension of the tram line.\n\n"
           "Volunteers organise repair cafes in every district of the city. Neighbours bring broken devices to the library.")
    rows = [{"id": f"d{i}", "date": f"2026-0{i + 1}-10", "title": f"Doc {i}", "full_text": txt} for i in range(3)]
    csv = tmp_path / "c.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    out = tmp_path / "o"
    out.mkdir()
    cfg = pt.Config(lang="en", extractor="spacy", skip_identical=False)
    pt.process_dataset(csv, None, cfg, NLP, LLM("x", "http://localhost:1", out / "c.json", enabled=False), StubEmbedder(), out)
    props = pd.read_csv(out / "c" / "propositions.csv")
    assert len(props) > 0 and set(props.orientation) == {"neutral"} and set(props.emotion) == {"neutral"}
    assert (out / "c" / "viewer.html").exists()
