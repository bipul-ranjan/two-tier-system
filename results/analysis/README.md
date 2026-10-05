# results/analysis/

Saved outputs of the research scripts that were used to choose the router. They are evidence, kept so the numbers can be checked.

| File | Made by | Contains |
|---|---|---|
| `reward_model_zero_shot_Skywork-Reward-V2-Qwen3-0.6B.csv` | `scripts/probe_reward_model.py` | The reward model's score for 500 (question, Tier 1 answer) pairs. Lets the script resume and lets the report be reprinted with `--report-only` |
| `router_oof_finetune_all-MiniLM-L6-v2_L256_E3_F5_S0_N6310.csv` | `scripts/train_router.py --mode finetune` | The fine-tuned MiniLM router's held-out prediction for each of 6,310 rows. The name records the settings (256 tokens, 3 epochs, 5 folds, seed 0, 6,310 rows) |
| `stage1_fold1_training_log.txt`, `stage1_fold1_report.txt` | Terminal output saved from `scripts/train_router.py` | The progress lines of the first fine-tuning fold (about an hour), and its first report, before all five folds were run. Kept as evidence; the **final** five-fold result is in the project documents, and fold 1 alone looked better than the full run |
| `router_embeddings_all-MiniLM-L6-v2_L256.npz` | `scripts/train_router.py` (frozen mode) | A 20 MB cache of MiniLM's reading of every question and answer. **Not worth keeping in Git**: it is regenerated in about seven minutes. It is listed in `.gitignore` |

Re-print a finished report without retraining: `python scripts/train_router.py --mode finetune --report-only` (use the same settings as
the run that saved the folds).
