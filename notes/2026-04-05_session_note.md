# Session note — 2026-04-05

Public release split completed.

What shipped on GitHub:
- orchestration / training / dataset-builder code
- docs and methodology
- curated `sample_dataset.jsonl`
- `.gitignore` rules excluding raw corpora, traces, batches, logs, checkpoints

What shipped on Hugging Face:
- `bonsai-8b-stage2-post-curriculum-q8.gguf`
- model card describing stage 2, domains, strengths, validation, and usage

Key references:
- GitHub repo: https://github.com/aurous37-lang/Hermes-Bonsai-Self-Improving-Agent-Loop
- Hugging Face repo: https://huggingface.co/Aurous37/Hermes-Bonsai-Self-Improving-Agent-Loop

Notes for next session:
- Keep GitHub and Hugging Face content split cleanly.
- If the model card changes, mirror only the relevant inference notes back to GitHub.
- Avoid adding raw corpora or private/generated trace dumps to either public surface.
