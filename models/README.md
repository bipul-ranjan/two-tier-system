# models/

Trained models used by the pipeline itself (not the fine-tuned Tier 1 models, which live in Ollama).

- `router/` -- the learned router. Created by `python scripts/train_text_router.py`:
  `text_probe_router.joblib` (the model), `text_probe_router_info.json` (its version, cut-point and settings)
  and `what_it_learned.txt` (the words and phrases it weighs most).

The model file is tied to the scikit-learn version that trained it. If you upgrade scikit-learn, or move to
another machine with a different version, retrain (it takes seconds).
