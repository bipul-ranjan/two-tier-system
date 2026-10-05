"""
Tests for src/learned_router.py -- the router that predicts how good Tier 1's answer is and escalates
when the prediction is below a cut-point. They use small synthetic data, so they need no Ollama,
no Claude key and no history file, and run in a couple of seconds:
    python -m pytest tests/ -q
"""
import json
import os
import sys

import joblib
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.learned_router import RouterNotTrainedError, TextProbeRouter  # noqa: E402


def make_data(n=400, seed=0):
    """Answers that mention 'specialist' score high, answers that mention 'declined' score low, and the
    retail unit scores a little lower overall. Confidence carries no information, on purpose."""
    rng = np.random.default_rng(seed)
    queries, answers, conf, units, quality = [], [], [], [], []
    for i in range(n):
        good = rng.random() < 0.5
        unit = "retail_bank" if i % 2 else "payments"
        queries.append(f"how do I {rng.choice(['cancel', 'freeze', 'replace', 'check'])} my {rng.choice(['card', 'account', 'transfer'])} {i % 7}")
        answers.append(("please contact our specialist who can help you with this" if good else "your request was declined and cannot be processed") + f" reference {i % 11}")
        conf.append(float(rng.uniform(0.5, 0.8)))
        units.append(unit)
        quality.append((4.5 if good else 2.0) - (0.3 if unit == "retail_bank" else 0.0) + float(rng.normal(0, 0.2)))
    return queries, answers, np.array(conf), units, np.array(quality)


@pytest.fixture(scope="module")
def trained():
    """One router trained on the synthetic data, shared by the tests in this file."""
    q, a, c, u, y = make_data()
    return TextProbeRouter.fit(q, a, c, u, y, cutoff=3.3, meta={"version": "test-router"}), (q, a, c, u, y)


def test_predicts_higher_quality_for_good_answers(trained):
    """An answer written like the high-scoring ones is predicted to score well above one written like
    the low-scoring ones.
    """
    router, _ = trained
    good = router.predict_one("how do I cancel my card", "please contact our specialist who can help you with this", 0.6, "payments")
    bad = router.predict_one("how do I cancel my card", "your request was declined and cannot be processed", 0.6, "payments")
    assert good > bad + 1.0


def test_single_prediction_matches_batch(trained):
    """Predicting one answer at a time gives the same numbers as predicting them as a batch."""
    router, (q, a, c, u, _) = trained
    batch = router.predict(q[:5], a[:5], c[:5], u[:5])
    one = [router.predict_one(q[i], a[i], c[i], u[i]) for i in range(5)]
    assert np.allclose(batch, one)


def test_escalates_exactly_below_the_cutpoint(trained):
    """A prediction below the cut-point escalates, one above it stays local, and one exactly at it
    stays local.
    """
    router, _ = trained
    assert router.decide(3.29) == "ESCALATE"
    assert router.decide(3.31) == "LOCAL"
    assert router.decide(3.3) == "LOCAL"          # a prediction exactly at the cut-point is kept, like the old rule's >=


def test_decisions_follow_the_prediction(trained):
    """Every decision equals "predicted quality below the cut-point", and the router separates the two
    kinds of answer.
    """
    router, (q, a, c, u, _) = trained
    preds = router.predict(q, a, c, u)
    decisions = [router.decide(p) for p in preds]
    assert decisions == ["ESCALATE" if p < 3.3 else "LOCAL" for p in preds]
    assert 0.2 < np.mean([d == "ESCALATE" for d in decisions]) < 0.8        # it separates the two kinds of answer


def test_an_unfamiliar_business_unit_still_gets_a_prediction(trained):
    """A business unit the router never saw still gets a finite prediction."""
    router, _ = trained
    p = router.predict_one("how do I check my account", "please contact our specialist", 0.6, "wealth_management")
    assert np.isfinite(p)


def test_empty_text_does_not_crash(trained):
    """An empty question and answer still give a finite prediction."""
    router, _ = trained
    assert np.isfinite(router.predict_one("", "", 0.6, "payments"))


def test_top_features_name_the_words_that_move_the_prediction(trained):
    """The words that push the prediction down or up are the ones from the matching answer template.
    """
    router, _ = trained
    top = router.top_features(k=10)
    # every word of a template predicts its quality equally well here, so the weight is shared among them:
    # check the words of each template land on the right side, not one particular word
    bad_words = {"declined", "cannot", "processed", "request", "was", "your"}
    good_words = {"specialist", "contact", "help", "who", "can", "our"}
    down = {t for w, _ in top["answer"]["down"] for t in w.split()}
    up = {t for w, _ in top["answer"]["up"] for t in w.split()}
    assert down & bad_words and not (down & good_words)
    assert up & good_words and not (up & bad_words)


def test_save_and_load_round_trip(trained, tmp_path):
    """A saved router loads back with identical predictions, cut-point and version, and writes its info
    file.
    """
    router, (q, a, c, u, _) = trained
    path = str(tmp_path / "router.joblib")
    router.save(path)
    loaded = TextProbeRouter.load(path)
    assert np.allclose(router.predict(q[:20], a[:20], c[:20], u[:20]), loaded.predict(q[:20], a[:20], c[:20], u[:20]))
    assert loaded.cutoff == router.cutoff and loaded.version == "test-router"
    info = json.load(open(str(tmp_path / "router_info.json")))
    assert info["cutoff"] == 3.3 and info["units"] == ["payments", "retail_bank"]


def test_loading_a_missing_model_says_how_to_train_it(tmp_path):
    """A missing model file raises an error that names the training script and the --threshold-router
    switch.
    """
    with pytest.raises(RouterNotTrainedError) as e:
        TextProbeRouter.load(str(tmp_path / "nope.joblib"))
    assert "scripts/train_text_router.py" in str(e.value) and "--threshold-router" in str(e.value)


def test_a_model_from_another_scikit_learn_version_is_refused(trained, tmp_path):
    """A model saved under another scikit-learn version is refused with a message to retrain."""
    router, _ = trained
    path = str(tmp_path / "router.joblib")
    router.save(path)
    saved = joblib.load(path)
    saved.meta["sklearn_version"] = "0.0.1"
    joblib.dump(saved, path)
    with pytest.raises(RouterNotTrainedError) as e:
        TextProbeRouter.load(path)
    assert "0.0.1" in str(e.value) and "Retrain" in str(e.value)


def test_a_model_without_a_cutpoint_is_refused(tmp_path):
    """A trained model that was never given a cut-point is refused."""
    q, a, c, u, y = make_data(100)
    path = str(tmp_path / "router.joblib")
    TextProbeRouter.fit(q, a, c, u, y).save(path)          # trained, but never given a cut-point
    with pytest.raises(RouterNotTrainedError) as e:
        TextProbeRouter.load(path)
    assert "cut-point" in str(e.value)


def test_a_corrupt_model_file_is_refused_clearly(tmp_path):
    """A file that is not a model is refused with a message to retrain."""
    path = tmp_path / "router.joblib"
    path.write_bytes(b"this is not a model")
    with pytest.raises(RouterNotTrainedError) as e:
        TextProbeRouter.load(str(path))
    assert "Retrain" in str(e.value)
