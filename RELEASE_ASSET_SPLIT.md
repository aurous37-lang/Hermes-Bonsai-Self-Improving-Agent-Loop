# GitHub vs Hugging Face Release Split

This project is intentionally split into:
- a lightweight GitHub repository for code, docs, and a tiny manually reviewed sample dataset
- a Hugging Face model release for exported weights and checkpoints

The rule of thumb:
- If it helps someone understand, reproduce, or run the loop, GitHub is the right home.
- If it is a large exported model artifact, Hugging Face is the right home.
- If it is raw corpus data, batch data, or generated trace material, it stays local and out of the public release.
- The full training build can exist locally, but it should not be shipped in the public repo.

## Keep on GitHub

These files are appropriate for the GitHub repository because they explain the loop, the curriculum, and the methodology without shipping the full training corpus:

### Core orchestration and training code
- `karpathy_loop.py`
- `hermes_iteration.py`
- `autonomous_loop.py`
- `distill_with_claude.py`
- `build_training_set.py`
- `build_stage2_weighted.py`
- `train_stage1.py`
- `eval_bonsai.py`
- `bonsai_turboquant_server.py`
- `analyze_sprint.py`
- `self_correction_speedrun.py`

### Documentation and methodology
- `README.md`
- `BONSAI_KARPATHY_HERMES_LOOP_WRITEUP.md`
- `hermes_bonsai_reasoning_traces_for_claudecode.md`
- `pre_finetune_analysis.md`
- `Gemini.txt`
- `notes/`

### Small public sample data
- `sample_dataset.jsonl` (manually reviewed and intentionally tiny)

### Repo hygiene
- `.gitignore`

### Local runtime state that should not be committed
- `task_queue.json`
- `bonsai_graduation_state.json`
- anything under `karpathy_loop_output/`
- anything under `karpathy_loop_metrics/`

## Keep on Hugging Face

These files are appropriate for a Hugging Face model repository because they are exported model artifacts rather than source code:

### Primary model artifacts
- Stage 2 GGUF checkpoint(s), for example:
  - `bonsai-stage2.gguf`
  - `bonsai-stage2-q4_k_m.gguf`
  - any other exported GGUF variants you want to publish

### Optional companion artifacts
- adapter weights such as `adapter_model.safetensors`
- model weights such as `model.safetensors` if you publish a non-GGUF variant
- model config files like `config.json`
- tokenizer files such as `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, and `special_tokens_map.json`
- `generation_config.json`
- a Hugging Face `README.md` / model card describing provenance, usage, and license

## Keep out of the public release entirely

These items should remain local/private and should not be uploaded to GitHub or Hugging Face as part of the public release:
- raw corpora
- generated training JSONL variants
- `raw_traces/`
- `gold_traces/`
- `training_batches/`
- machine-specific logs
- checkpoint directories
- secrets, auth files, and local environment files
- full training datasets built from private or unreviewed outputs

## Practical publishing flow

1. Publish the code/docs/sample on GitHub.
2. Publish the exported weights on Hugging Face.
3. Link the Hugging Face artifact from the GitHub README.
4. Keep raw corpora and generated traces out of both public surfaces.
