# Regularization screen

Ten fresh seed-0 fits compare the original MSE control with head dropout 0.2/0.5, adapter input dropout 0.1/0.3, separate head or adapter AdamW decay 0.1/1.0, and a combined setting. Defaults are no dropout and decay 0.01. All fits use 3 exposures (2250 updates), the same learning rates, split, batch size and cosine schedule. Adapter dropout affects both LoRA sets when enabled; this screen uses variant adapters only.

[Exact plan](plan.json) · [Live status](STATUS.md)

The source snapshot and resolved per-run configurations are saved under `/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1`. W&B group: `mutate_everything_regularization_v1`. Every run saves checkpoints selected by overall Spearman, doubles Spearman, doubles RMSE and doubles nDCG@30. Stage promotion uses overall Spearman; inspect other metrics separately. No external benchmark is evaluated. Confirmation across seeds is required before treating screen differences as reliable.

The regularization worker runs on GPU 2 independently of the existing tuning worker on GPU 0.
