"""Fast unit tests (no LLM, no models). Run: pytest"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import build_viewer  # noqa: E402
import emotions  # noqa: E402
import proposition_tracker as pt  # noqa: E402


def test_emotion_tables_are_consistent():
    assert set(emotions.JASPER_EMOTIONS) == set(emotions.EMOTION_GROUP) - {"neutral"}
    assert set(emotions.EMOTION_DIRECTION) == set(emotions.EMOTIONS)
    assert build_viewer.VALENCE is emotions.EMOTION_DIRECTION


@pytest.mark.parametrize("raw,expected", [
    ("anger", "anger"), ("Hope", "joy_hope"), ("sadness", "grief_sorrow"), ("outrage", "outrage_indignation"),
    ("", "neutral"), (None, "neutral"), ("gibberish-label", "neutral"),
])
def test_norm_emotion(raw, expected):
    assert pt.norm_emotion(raw) == expected


def test_status_flips():
    assert pt.orientation_status("supportive", "critical") == "flip"
    assert pt.orientation_status("supportive", "neutral") == "shift"
    assert pt.orientation_status("critical", "critical") == "same"
    assert pt.emotion_status("joy_hope", "resignation") == "flip"      # encouraging <-> dampening
    assert pt.emotion_status("anger", "joy_hope") == "shift"           # anger has no direction
    assert pt.emotion_status("anger", "anger") == "same"


def test_prop_sim_ignores_unspecified_elements_and_needs_two():
    v = lambda *x: np.array(x, dtype=np.float32)  # noqa: E731
    a = {"actor": v(1, 0), "action": v(0, 1), "object": v(1, 0)}
    b = {"actor": v(1, 0), "action": v(0, 1), "object": v(0, 1)}
    s, cos = pt.prop_sim(a, b, (0.25, 0.25, 0.5))
    assert cos["object"] == pytest.approx(0.0) and s == pytest.approx(0.5)
    zero = v(0, 0)
    only_one = {"actor": v(1, 0), "action": zero, "object": zero}
    assert pt.prop_sim(a, only_one, (0.25, 0.25, 0.5))[0] == 0.0       # < 2 specified elements -> no match


def test_clean_input_parses_german_dates_unambiguously(tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame({"date": ["11.04.2026", "01.02.2026"], "full_text": ["First text.", "Second text."]}).to_csv(raw, index=False)
    out = tmp_path / "clean.csv"
    subprocess.run([sys.executable, str(ROOT / "clean_input.py"), "--input", str(raw), "--output", str(out),
                    "--date-format", "%d.%m.%Y", "--today", "2026-12-31"], check=True, capture_output=True)
    df = pd.read_csv(out)
    assert list(df["date"]) == ["2026-02-01", "2026-04-11"]            # 11 April, not 4 November
    assert list(df["id"]) == ["doc001", "doc002"]
