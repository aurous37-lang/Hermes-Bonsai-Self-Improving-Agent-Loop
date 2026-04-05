# Karpathy Auto-Research Loop for Local Self-Improving Agents

A self-improving training pipeline where a small local model (Bonsai 1-bit 8B) generates its own fine-tuning data through a teacher/student loop with a stronger model (GPT-5.4-mini via Codex). The local model progressively learns structured reasoning, tool use, and self-correction — with the goal of eventually running the entire loop autonomously.

**This is active research.** The pipeline is running and producing results. Everything here is reproducible on consumer hardware (single RTX 5070 Ti).

---

## What This Is

A concrete implementation of several ideas that the field has mostly only theorized about:

- **Karpathy Auto-Research** — the model drives its own learning agenda, generating tasks in domains where it's weakest
- **Knowledge Distillation** — a strong teacher (Codex) corrects and compresses a weak student's (Bonsai) reasoning traces
- **Self-Play for Reasoning** — the student attempts, gets scored, the corrected traces become its next training set
- **Continuous Curriculum** — weakness analysis steers task generation toward the student's failure modes
- **Structured Reasoning Traces** — compact `goal/constraints/plan/checks/decision` format instead of verbose chain-of-thought

The pipeline runs on a cron loop — every 15 minutes, 20 tasks are generated, solved by the student, distilled by the teacher, scored, and compiled into training-ready JSONL. No human in the loop.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Karpathy Auto-Research Loop              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    raw trace    ┌──────────┐    gold trace    │
│  │  Bonsai  │ ──────────────> │  Codex   │ ──────────────>  │
│  │  1-bit   │                 │  GPT-5.4 │                  │
│  │   8B     │ <── fine-tune ──│ (teacher)│ <── score ──     │
│  │(student) │                 └──────────┘                  │
│  └──────────┘                                               │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────┐                                               │
│  │ Training │  ChatML + Structured JSONL                    │
│  │  Data    │  (dual format, loss-masked)                   │
│  └──────────┘                                               │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────┐                                               │
│  │  LoRA    │  Unsloth/Axolotl, rank 64                     │ 
│  │Fine-Tune │  all-linear, LR 1.5e-4                        │
│  └──────────┘                                               │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────┐                                       │
│  │ TurboQuant Cache │  V=4bit, K=3bit                       │
│  │ (inference only) │  Enables 32K-64K context              │
│  └──────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### The Three Stages

**Stage 1 — Research & Generate (Bonsai)**
The student model receives a task from the queue and produces a raw reasoning trace. This is Bonsai's genuine attempt — messy, verbose, sometimes wrong. That's expected and valuable.

**Stage 2 — Distill & Refine (Codex)**
The teacher takes the raw trace and produces a gold-standard compressed version following the structured schema. It scores the student's attempt on correctness (0-10), conciseness, schema consistency, and actionability. It also marks whether the student's raw attempt was `pass`, `partial`, or `fail` — this is the primary training signal.

**Stage 3 — Compile & Iterate**
Validated traces are compiled into training-ready JSONL in dual format (ChatML conversations + structured trace). Metrics are logged. The task queue is updated. Weak domains are identified and fed back into task generation for the next cycle.

## The Reasoning Trace Format

Instead of verbose chain-of-thought, we train on compact structured traces:

```
<think>
Goal: Find the heavier counterfeit among 12 coins in 3 weighings.
Constraints: Exactly 3 weighings, 1 heavier coin, balance scale.
Plan: Ternary search — split 12→4→2→1.
Checks: 3^3=27 outcomes covers 12. Each step eliminates 2/3.
Decision: Divide into equal groups of 4 at each stage.
</think>

Split into 3 groups of 4. Weigh A vs B — heavier side has the counterfeit;
if balanced, it's in C. Take the 4 suspects, weigh 2 vs 2. Finally weigh
1 vs 1. Done in exactly 3 weighings.
```

**Why this format:**
- Compact enough for a 1-bit 8B model (40-500 tokens by difficulty tier)
- Explicit structure makes it easy to score and validate
- Each field serves a purpose: goal = what, constraints = boundaries, plan = how, checks = verification, decision = commitment
- The `<think>` tags create a clean loss-masking boundary for training

**Token budgets:**
- Easy tasks: 40-120 tokens
- Medium tasks: 120-250 tokens
- Hard tasks: 250-500 tokens

## TurboQuant KV Cache Compression

Bonsai's 1-bit weights are only 1.1GB, but the KV cache is uncompressed and grows linearly with context length. This causes coherence drift on longer reasoning chains — the model "loses the thread."

[TurboQuant](https://github.com/mit-han-lab/TurboQuant) compresses the KV cache at inference time:

```python
from transformers import AutoModelForCausalLM
from turboquant import TurboQuantCache

model = AutoModelForCausalLM.from_pretrained("prism-ml/Bonsai-8B", torch_dtype=torch.float16, device_map="auto")
cache = TurboQuantCache(model, v_bits=4, k_bits=3)
outputs = model.generate(**inputs, past_key_values=cache, use_cache=True)
```

**Key insights:**
- V=4-bit, K=3-bit is the sweet spot — values are more sensitive than keys
- 4-bit TurboQuant on models >3B is indistinguishable from full precision
- Most effective at 4K+ token contexts
- Fixes the inference bottleneck while LoRA training fixes the weights
- Combined: a model that reasons in the right format AND sustains it across 32K+ context

## Results So Far

> **Status: Active research — numbers update as the loop runs**

| Metric | Baseline (pre-loop) | Current |
|--------|:-------------------:|:-------:|
| Training entries | 0 | 2,148+ (growing) |
| Iterations completed | 0 | 80+ |
| Tasks generated | 0 | 600+ |
| Codex validation rate | — | 95.3% |
| Avg correctness score | — | 9.3/10 |
| Bonsai raw pass rate | — | 67% |
| Format compliance | 0% | 0% (pre-training) |
| Distill failures | ~20% (self-distill) | 0% (Codex) |

### Bonsai Pass Rate by Domain

Where the student gets it right without teacher correction:

| Domain | Pass Rate | Notes |
|--------|:---------:|-------|
| refusal_redirect | 96% | Knows when to say no |
| self_correction | 84% | Can acknowledge and fix errors |
| research_synthesis | 83% | Good at comparative analysis |
| agent_routing | 82% | Picks the right tool usually |
| math | 79% | Solid on calculation, weak on word problems |
| memory_integration | 92% | Up from 62% — curriculum targeting worked |
| code_debugging | 59% | Misses edge cases |
| architecture | 56% | Weak on constraint analysis |
| devops | 39% | Gets direction but not specifics — diagnostic chunking now active |
| logic_puzzle | 57% | Up from 32% — learning systematic elimination |

The loop automatically generates more tasks in weak domains.

### Dataset Composition

- **543 ChatML pairs** — conversations format for SFT
- **543 structured traces** — goal/constraints/plan/checks/decision with metadata
- **30 hand-crafted Stage 2 exemplars** — compressed traces (25-78 tokens) as anchors
- **10 domains**, difficulty split: 20% easy, 49% medium, 30% hard

## Theoretical Foundation

### Why This Works for Small Models

The target is: **can it become a reliable agent with calibrated self-awareness and tool routing?**

A model that:
- Reasons compactly in a structured format
- Calls the right tool instead of guessing
- Routes hard tasks to stronger models when it recognizes its limits
- Self-corrects before finalizing
- Runs locally at ~270 tok/s with no API costs

That's achievable through this pipeline.

### Perception → Context → Intention

The cognitive order and frame-work for this agent is:

1. **Perceive** — what is actually in front of me? Raw signal, not assumptions.
2. **Context** — given what I perceive, which of my patterns apply? This is where memory fires.
3. **Intention** — now I know what I'm looking at and what applies. Execute.

Current agent architectures invert this — they start with intention (answer the query), then look for context (RAG retrieval), and barely perceive the actual situation. Training on traces that encode all three layers teaches the model to *read the room* rather than just answer questions.

### Braided Memory-Logic

A research direction we're exploring: memory and logic should be intertwined, not separate systems.

Current agents have three loose ropes running parallel:
- **Semantic content** (what it is) — the fact in the vector store
- **Logical affordance** (what it means) — implies, contradicts, enables, blocks
- **Activation context** (when it fires) — the trigger pattern, the situation shape

These should be braided. Each crossing point reinforces the others:
- Content x Logic = facts carry their own reasoning
- Logic x Context = reasoning knows when to apply
- Context x Content = retrieval is situational, not just similar

A heuristic is exactly this — a memory that already contains the reasoning pattern and fires when the situation matches. Current RAG retrieves a fact and then asks the model to reason about it. A braided system would retrieve *the reasoning itself*.

### Diagnostic Chunking: Solving the DevOps Wall

**The problem:** Bonsai's pass rate on DevOps tasks was 33-39% — far below its 96% on refusal or 84% on self-correction. It consistently got the "direction" right but missed the "specifics." Why?

**The insight:** In a 1-bit model, each token prediction accumulates quantization error. When Bonsai has to generate a long chain of domain-specific tokens (`/sys/fs/cgroup/memory/memory.usage_in_bytes`), the signal degrades by the time it spells out the specifics. It's like trying to remember a 20-digit number one digit at a time — by digit 10, you're guessing.

**The analogy:** Exceptional readers don't process word-by-word. They chunk 3-4 words as a single semantic unit: "service failed after deploy" arrives as two chunks ("service failed" + "after deploy"), not five isolated tokens. The meaning is captured before entropy can act on each piece.

**The reframe:** The model doesn't fail because it lacks the concept — it fails because the concept diffuses before enough related signal is bound together.

**The solution — diagnostic chunks:**

Instead of training on atomic steps:
```
Plan: Check pod status. Check container logs. Inspect cgroup memory file.
```

Train on bound diagnostic units:
```
Plan: "container alive but proxy unreachable" → check upstream config + bind address
```

Each chunk is a single retrieval/reasoning unit. The model retrieves the *pattern* whole, rather than reconstructing it token-by-token where entropy degrades the chain.

**Implemented as:**
1. **Chunked distillation prompts** — Codex now groups commands/paths into logical chunks when distilling DevOps/debug/architecture traces
2. **Diagnostic chunk library** — Stored in Hermes memory as pattern→implies→checks entries
3. **Failure mode taxonomy** — DevOps failures are annotated as: missed chunk, wrong chunk, correct chunk wrong ranking, or correct ranking weak specificity

**Diagnostic chunk examples:**
```yaml
pattern: "works locally, fails in container"
implies: [env mismatch, missing dependency, path difference, port binding]
checks: [compare env vars, verify container paths, inspect bind address]

pattern: "deploy succeeded but service broken"
implies: [migration didn't run, config not reloaded, feature flag toggled]
checks: [verify migration status, check config reload mechanism, review flag state]
```

This connects directly to the braided memory architecture — each chunk weaves content (what the pattern is), logic (what it implies), and activation context (when it fires) into a single retrievable unit.

### Domain-Specific Cognitive Modes

A key finding at ~3,000 entries: **one trace format doesn't fit all domains.** Chunked distillation helped diagnostic tasks but caused a regression in agent routing (82% → 43%) — routing is decisional dispatch, not diagnosis. Forcing the model into elaborate evidence binding when it needs fast classification actually hurts performance.

The pipeline now applies domain-specific cognitive modes:

| Mode | Domains | Trace Style | Optimizes For |
|------|---------|-------------|---------------|
| **Diagnostic** | devops, code_debugging | Chunked evidence binding, pattern→implies→checks | Preserving causal structure |
| **Routing** | agent_routing | Compressed threshold logic: classify→select→fallback | Decisive dispatch |
| **Deductive** | logic_puzzle | Explicit variable tracking, elimination steps | State maintenance across steps |
| **Constraint** | architecture | Chunk + constraint interaction tracking | Multi-constraint trade-offs |
| **Behavioral** | refusal, self_correction | Minimal structural response | Clean pattern execution |

This maps onto the deeper insight: **different forms of thought require different forms of memory activation and structuring.** Diagnosis needs bound evidence. Routing needs compressed thresholds. Deduction needs tracked variables. A mature agent architecture should match the cognitive mode to the task, not force everything through one reasoning template.

### Four-Label Failure Taxonomy

Binary pass/fail loses critical training signal. For domains where Bonsai is "partial" (right direction, wrong specifics), we annotate four distinct failure types:

1. **Correct chunk, weak specificity** — identifies the right problem class but not the exact test/fix
2. **Correct chunk, incomplete coverage** — finds the first bug/root cause but misses the second or edge case
3. **Correct direction, wrong ranking** — lists plausible causes but orders them badly
4. **Correct diagnosis, weak verification** — guesses the right issue but doesn't attach the concrete command/check

These labels transform "partial" from noise into targeted training signal. Each type suggests a different intervention: weak specificity needs vocabulary anchoring, incomplete coverage needs second-pass verification, wrong ranking needs calibration data, weak verification needs command-level chunking.

## File Structure

```
.
├── README.md                          # This file
├── CLAUDE.md                          # Project context for AI assistants
│
├── karpathy_loop.py                   # Core 3-stage pipeline
├── hermes_iteration.py                # Two-model iteration (Bonsai + Codex)
├── autonomous_loop.py                 # Self-sustaining loop with task generation
├── distill_with_claude.py             # Manual gold distillation tool
│
├── train_stage1.py                    # Unsloth LoRA training config
├── eval_bonsai.py                     # Before/after evaluation harness
├── bonsai_turboquant_server.py        # TurboQuant inference server
│
├── task_queue.json                    # Local runtime state (not committed)
├── sample_dataset.jsonl               # Curated public sample dataset
├── hermes_bonsai_reasoning_traces_for_claudecode.md  # Methodology doc
├── Gemini.txt                         # Training schema + hyperparameters
│
└── karpathy_loop_output/              # Generated locally; ignored in GitHub
    ├── training_data.jsonl            # Full local build output
    ├── train.jsonl                    # 90% split for training
    ├── eval.jsonl                     # 10% split for evaluation
    ├── raw_traces/                    # Generated Bonsai attempts
    ├── gold_traces/                   # Generated teacher traces
    └── training_batches/              # Per-iteration batch files
```

## How to Run

### Prerequisites

- GPU with 16GB+ VRAM (tested on RTX 5070 Ti)
- [Hermes Agent](https://github.com/nousresearch/hermes-agent) v0.6.0+
- A local model server (LM Studio, llama.cpp, vLLM) running a Qwen3-8B variant
- Access to a strong model API for distillation (Codex, Claude, GPT-4+)
- Python 3.11+ with: `transformers`, `torch`, `turboquant`, `openai`

### Quick Start

```bash
# 1. Start your local model server (e.g., LM Studio with Bonsai-8B)
#    Should serve on your local OpenAI-compatible endpoint

# 2. Run a single iteration (5 tasks, self-distilled via Bonsai)
python karpathy_loop.py --pilot

# 3. Run with Codex as teacher (requires Hermes OAuth login)
hermes model  # Select openai-codex, authenticate
python hermes_iteration.py --count 20

# 4. Set up the cron loop
hermes cron create "every 15m" \
  'Use the terminal tool to run the Hermes venv Python against the iteration script with the desired count' \
  --name "karpathy-loop"

# 5. Run baseline evaluation
python eval_bonsai.py --tag baseline

# 6. When ready to train
python train_stage1.py

# 7. After training, compare
python eval_bonsai.py --tag stage1
python eval_bonsai.py --compare baseline stage1
```

### Running with TurboQuant

```bash
# Start TurboQuant-enabled server (alongside or instead of LM Studio)
python bonsai_turboquant_server.py --port 1235 --v-bits 4 --k-bits 3

# Point the iteration script at the TurboQuant server
# Update BONSAI_API in the iteration script to the local TurboQuant endpoint
```

## Training Configuration

From `train_stage1.py`, optimized for Bonsai 1-bit 8B:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA Rank | 64 | Higher rank compensates for low-bit quantization |
| Target Modules | all-linear | Captures full reasoning flow |
| Learning Rate | 1.5e-4 | Prevents "Bit-Collapse" in 1-bit weights |
| Max Seq Length | 8192 (Stage 1) | Conservative start, scale to 32768 |
| Loss Masking | `<think>` + assistant only | Don't train on system/user tokens |
| Epochs | 3-5 | Format learning converges fast |
| Effective Batch | 16 | 2 x 8 gradient accumulation |

### Two-Stage Training

**Stage 1 — Format Learning** (current focus)
Teach the model to emit structured `<think>` traces reliably. Target: 80%+ format compliance.

**Stage 2 — Trace Compression**
After the model learns the format, compress traces aggressively while preserving correctness. Target: average think-block length under 80 tokens with maintained accuracy.

## Adapting for Your Own Model

The pipeline is model-agnostic. To use it with a different local model:

1. Change `BONSAI_API` and `BONSAI_MODEL` in `karpathy_loop.py`
2. Adjust token budgets in `TOKEN_TARGETS` if your model has different capacity
3. Update `MODEL_NAME` in `train_stage1.py` to your base model's HuggingFace ID
4. Adjust LoRA rank and learning rate for your quantization level

The teacher model (Codex) can be swapped for any strong model API — Claude, GPT-4, a larger local model, etc. The distillation prompts in `hermes_iteration.py` are model-agnostic.

## Research Directions

**Active:**
- Measuring Bonsai's pass rate improvement after Stage 1 fine-tuning
- Diagnostic chunking for DevOps/debug domains — does binding evidence into semantic chunks before prediction improve specificity?
- Optimal dataset mix for format learning vs domain competence
- Stage 2 trace compression — how short can traces be while preserving correctness?

**Planned:**
- Perception/Context/Intention trace format (encoding situation recognition, not just problem solving)
- Braided memory-logic annotations (memories that carry their own reasoning affordances)
- Diagnostic chunk library as retrievable reasoning units — testing whether chunk-based memory reduces the "partial" failure rate on multi-step diagnostic tasks
- Two-pass inference: evidence chunking pass → pattern matching pass → ranked differential
- Failure mode annotation: missed chunk / wrong chunk / correct chunk wrong ranking / correct ranking weak specificity
- Closing the loop: Bonsai as both student AND teacher (self-distillation quality after training)
- TurboQuant impact on long-chain reasoning accuracy
- Multi-Token Prediction (MTP) integration for 1-bit models — predicting 4-8 token chunks to reduce per-token entropy accumulation
- Tinker/Atropos RL polish pass after distillation

## Public Release Split

This repository is intentionally published as a safe, lightweight public release. The goal is to share the loop, curriculum design, graduation protocol, and methodology without shipping raw corpora or large binary artifacts.

See `RELEASE_ASSET_SPLIT.md` for the exact GitHub vs Hugging Face boundary.

GitHub gets:
- orchestration code (`karpathy_loop.py`, `hermes_iteration.py`, `autonomous_loop.py`)
- training scripts (`train_stage1.py`)
- dataset builders and evaluation scripts
- schema / methodology docs
- a small manually reviewed `sample_dataset.jsonl` for smoke tests and format demos
- the documentation needed to understand and reproduce the loop

Hugging Face gets:
- exported model checkpoints and weights
- stage 2 GGUF artifacts
- any other large companion model files produced by training/export

Neither GitHub nor Hugging Face should contain:
- raw corpora and batch files
- generated training JSONL variants
- large generated datasets and traces
- logs with machine-specific paths
- checkpoint directories and exported weights in the repo tree
- secrets, auth files, and local environment files

Top-level folder map for new contributors:
- `README.md` — project overview, loop design, and public release notes
- `karpathy_loop.py` — core orchestration loop and dataset compilation
- `hermes_iteration.py` — iterative teacher/student distillation and scoring
- `autonomous_loop.py` — autonomous task-generation loop
- `train_stage1.py` — training entrypoint
- `build_stage2_weighted.py` — dataset weighting / rebuild script
- `eval_bonsai.py` — baseline and post-training evaluation
- `sample_dataset.jsonl` — curated clean sample dataset
- `notes/` — human-readable session notes and methodology notes
- `karpathy_loop_output/` — generated artifacts only (ignored in GitHub)
- `karpathy_loop_metrics/` — generated metrics only (ignored in GitHub)

## Hugging Face Artifacts

The stage 2 GGUF checkpoint should live on Hugging Face rather than in the repo. Add the release link here when publishing:
- Stage 2 GGUF: https://huggingface.co/<your-org>/<bonsai-stage2-gguf-repo>

If you publish additional exports, keep them with the model release rather than adding them to GitHub.
## Contributing

This is early-stage research. If you're working on:
- Small model self-improvement loops
- Structured reasoning trace formats
- KV cache compression for local inference
- Agent memory architectures
- Knowledge distillation pipelines

We'd love to hear from you. Open an issue or PR.

## License

MIT

## Acknowledgments

- [Andrej Karpathy](https://karpathy.ai/) — the auto-research loop concept
- [Nous Research](https://nousresearch.com/) — Hermes Agent framework
- [Unsloth](https://github.com/unslothai/unsloth) — efficient LoRA training
- [TurboQuant](https://github.com/mit-han-lab/TurboQuant) — KV cache compression
- [Prism ML](https://huggingface.co/prism-ml) — Bonsai 1-bit 8B model
