# GRASP LoRA (MLQA + XLsum)

This folder packages the MLQA and XLsum workflows plus the GRPO controller used to learn sparsity (delete) ratios for bilingual merges (English+Arabic, English+Chinese).

## Prerequisites
- First, generate the datasets and splits: `python dataset/prepare_mlqa_xlsum_splits.py` (requires the `datasets` library and network access). This writes the MLQA and XLsum train/val/test/micro_dev CSVs under `dataset/*/input/` as expected by `dataset_constants.py`.
- Then install the Python dependencies: `pip install -r requirements.txt` (ideally inside your conda environment). If you have 2 GPUs available, set `CUDA_VISIBLE_DEVICES` accordingly and torchrun will shard the work across both instead of a single GPU.
- All scripts automatically set `PYTHONPATH` to use the local `configurations/src` and `dataset` modules.

## MLQA Pipeline
- `merge/1_finetuning_lora_MLQA.sh` — fine-tunes three adapters on MLQA in order: English → Chinese → Arabic. Outputs land in `merge/output/*_adapter`.
- `merge/1_generate_lora_MLQA.sh` — runs inference on the MLQA test split for the Arabic and Chinese adapters; generations go to `merge/output/generations/<lang>`.
- `merge/GRASP_LoRA_ar_ch_MLQA.sh` — GRPO run to learn best delete ratios (sparsity) for English+Arabic and English+Chinese MLQA merges.
- `merge/3_finetuning_merge_del_MLQA.sh` — deterministic sweeps over user-provided delete ratios (`AR_DELETE_PERCENTS` and `CH_DELETE_PERCENTS`) across multiple seeds to reproduce merge+delete variants.
- `merge/3_generate_merge_del_seeds_MLQA.sh` — generates on MLQA using the merge+delete checkpoints produced in the previous step (uses the same ratios and seeds by default).

## XLsum Pipeline
- `merge/1_finetuning_lora_XLsum.sh` — fine-tunes XLsum adapters in order: English summary → Chinese summary → Arabic summary. Outputs land in `merge/output/*_adapter`.
- `merge/1_generate_lora_XLsum.sh` — generates on the XLsum test split for the Arabic and Chinese summary adapters (single adapters, no merges).
- `merge/GRASP_LoRA_ar_ch_XLsum.sh` — GRPO run to learn best delete ratios (sparsity) for English+Arabic and English+Chinese XLsum merges.
- `merge/3_finetuning_merge_del_XLsum.sh` — deterministic sweeps over configurable delete ratios (`AR_DELETE_PERCENTS`, `CH_DELETE_PERCENTS`) across multiple seeds for summary merges.
- `merge/3_generate_merge_del_seeds_XLsum.sh` — generates on XLsum using the merge+delete checkpoints produced in the previous step (same ratios and seeds by default). An alias `merge/3_generate_lora_all_lang_XLsum.sh` points to this generation step.

## Notes
- Default model: `meta-llama/Meta-Llama-3-8B-Instruct`. Override via environment variables in each script if needed.
- Outputs stay under `merge/output` in this repository so you can keep results self-contained when pushing to GitHub.
