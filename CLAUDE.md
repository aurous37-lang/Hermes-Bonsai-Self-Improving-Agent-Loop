# Hermes Agent Training Research

## Project
Karpathy Auto-Research loop for self-improving Hermes agent. Bonsai 1-bit 8B (student) generates reasoning traces, Codex GPT-5.4-mini (teacher) distills gold-standard compressed traces with domain-specific cognitive modes. 6,557 entries at iteration 178. Loop is live and autonomous. Training must happen INSIDE the loop under Hermes's control — never as a standalone pipeline.

## Architecture

**CRITICAL RULE: Hermes is always in the loop. Never drift into standalone pipelines.**
Bonsai must always be running. The cron must always be active. Training is a step IN the Karpathy cycle, not a separate process. If you find yourself writing a standalone script that bypasses Hermes, stop and redesign. Ask: "Is Hermes driving this, or did I take over?"

**TRAINING RULE: When "let's train" is requested, do NOT jump to standalone CUDA/GGUF work.**
Follow this ladder instead:
1. Define the exact behavior to teach
2. Inspect the loop-generated outputs for format consistency
3. Build a tiny high-consistency sample set (25-100 examples)
4. Run the smallest possible in-loop overfit test
5. Evaluate behavioral change
6. Only scale up after the tiny test passes

If you find yourself writing LD_LIBRARY_PATH, pkill llama-server, or running a standalone training script — STOP. You've left the loop. Two sessions have made this exact mistake.

- **Hermes**: v0.6.0, cron-driven loop (every 15m), memories in the Hermes config store — ORCHESTRATES EVERYTHING
- **Bonsai**: local OpenAI-compatible endpoint, Qwen3-8B 1-bit + TurboQuant KV compression, raw reasoning generation — ALWAYS RUNNING
- **Codex**: GPT-5.4-mini via OAuth (not API key), gold distillation with 5 cognitive modes + 3D scoring
- **Output**: karpathy_loop_output/training_data.jsonl (dual format: ChatML + structured; generated locally, not shipped)
- **Cron**: hermes cron fires hermes_iteration.py every 15 min, Hermes runs it via terminal tool
- **Python**: Hermes-managed venv Python (has openai, etc.)
- **llama-server**: local llama-server build, large context, flash-attn, full GPU offload

## Key Files
- `karpathy_loop.py` — Core pipeline: DISTILL_SYSTEM prompt, stage2_apply_gold(), compile, metrics
- `hermes_iteration.py` — Two-model iteration: codex_distill() with cognitive modes, analyze_weaknesses() with failure signatures, codex_generate_tasks() with targeted curriculum
- `autonomous_loop.py` — Self-sustaining loop with task generation (fallback when Codex offline)
- `build_training_set.py` — Weighted training set builder (50% frontier / 30% strong / 20% solved)
- `eval_bonsai.py` — Before/after evaluation harness (baseline captured: 0/0/0)
- `analyze_sprint.py` — Post-sprint analysis (3D scores, domain evolution, partial types, token anchoring)
- `bonsai_turboquant_server.py` — TurboQuant inference server (port 1235)
- `distill_with_claude.py` — Manual Claude gold distillation tool

### Artifacts from previous session (not part of the loop)
- `train_stage1.py` — Standalone Unsloth LoRA script. Violates architecture rule. Reference only.
- `unsloth_compiled_cache/` — Leftover from standalone training attempt. Can be deleted.
- `karpathy_loop_output/stage1_checkpoint/` (677MB) — LoRA adapter from standalone run. Can be deleted.

## Current State (iteration 178, 2026-04-03)

### Model Performance (500-task run with corrected pipeline)
| Metric | Value |
|--------|-------|
| Training entries | 6,557 |
| Validation rate | 92.4% |
| 3D scoring adoption | 100% (was 21.9% before fix) |
| Domain schema adoption | 100% (was 15.9% before fix) |
| Partial type classification | 98.8% (was 0.7% before fix) |
| Bonsai raw pass rate | 47% |
| Bonsai partial rate | 45% |
| Bonsai fail rate | 6% |

### 3D Scores (500-task average)
| Dimension | Average | Interpretation |
|-----------|---------|----------------|
| Direction | 9.18 | Bonsai enters the right problem family |
| Completion | 9.17 | Bonsai finishes the answer |
| Specificity | 8.21 | **Gap of ~1.0 point** — the bottleneck |

### Domain Tiers (updated from 500-task run)
- **Solved** (93-100%): math, refusal_redirect, research_synthesis — reinforcement only
- **Strong** (64-83%): memory_integration, architecture, self_correction — cognitive modes working
- **Recovered**: agent_routing — back from 40%, still somewhat unstable
- **Frontier** (43-50%): code_debugging (~40% flat), logic_puzzle (~45%), devops (~43%) — partial-heavy

## What Was Fixed (2026-04-03)

### Root cause: DISTILL_SYSTEM contradicted domain-specific prompts
The system prompt in `karpathy_loop.py` hardcoded the generic schema (`goal/constraints/plan/checks/decision`) and old 4-field scoring (`correctness/conciseness/schema_consistency/actionability`). The user prompt in `codex_distill()` specified domain-specific schemas and 3D scoring. Codex followed the system prompt ~84% of the time, causing:
- 84% generic schema fallback
- 78% old 4-field scoring
- 99.3% of partial_type set to "n/a"
- Only 7% of traces had token anchors

### Changes made
1. **DISTILL_SYSTEM** made schema-agnostic — removed hardcoded schema and old scores, added directives for partial_type and token anchors ([CMD:], [CODE:])
2. **Score normalization** in stage2_apply_gold() — remaps old 4-field scores to 3D if Codex reverts
3. **Schema validation** in codex_distill() — measures domain schema match, infers partial_type from 3D scores when Codex doesn't set it
4. **Enriched analyze_weaknesses()** — returns failure_count, top_partial_types, avg_specificity per domain (was just domain names)
5. **Targeted task generation** — codex_generate_tasks() now receives failure signatures and generates curriculum that targets specific weakness types (weak_specificity → exact command tasks, incomplete_coverage → multi-bug tasks, etc.)
6. **Verification printout** — each iteration reports 3D vs old scores, schema match rate, partial_type fill rate

### Result
After fix, 500-task validation run: 100% 3D scoring, 100% domain schemas, 98.8% partial_type classification. The feedback loop from failure analysis to task generation is now closed.

## Five Cognitive Modes
Different tasks need different trace structures. Wrong mode causes regressions (routing dropped 82%→43% under diagnostic chunking, recovered to 78% with routing mode).

1. **Diagnostic** (devops): symptom_chunks → ranked_hypotheses → exact_next_check → confirm/rule_out → diagnosis. Token-anchor commands with [CMD: ...]
2. **Diagnostic+Completion** (code_debugging): observed_bug → primary_cause → secondary_risk → edge_cases → minimal_fix → verification_test. Token-anchor code with [CODE: ...]
3. **Deductive** (logic_puzzle): knowns → unknowns → constraints → eliminations → remaining_candidates → final_solution
4. **Routing** (agent_routing): task_type → confidence → best_route → why → fallback → escalation_trigger. Compressed threshold logic, NOT analysis.
5. **Constraint** (architecture): goal → constraints → tensions → resolution → decision

## Three-Dimensional Scoring
Replaces flat correctness score:
- **Direction** (0-10): Did it enter the right problem family?
- **Specificity** (0-10): Did it give the right exact discriminator/command/edge case?
- **Completion** (0-10): Did it fully close the loop without leaving half-formed?

## Four-Label Failure Taxonomy
For partial failures (45% of all traces — the dominant category):
1. **weak_specificity**: right problem class, missing exact test/fix
2. **incomplete_coverage**: found first bug, missed second/edge case
3. **wrong_ranking**: plausible causes listed but ordered badly
4. **weak_verification**: right diagnosis, no concrete command to confirm

Now actively classified on every partial/fail trace AND fed back into task generation.

## TurboQuant KV Cache Compression
- turboquant 0.2.0, V=4-bit, K=3-bit
- Fixes inference bottleneck: 1-bit weights are 1.1GB but KV cache is uncompressed
- Enables 94K context on RTX 5070 Ti (16GB) without coherence drift
- Server: bonsai_turboquant_server.py on :1235
- llama-server runs Bonsai at localhost:1234 with --ctx-size 94208

## Research Thesis: Unified Memory-Logic

### Core claim
Reasoning quality depends on retrieving the right form of structure for the task, not just the right content. Different forms of thought require different forms of memory activation and structuring.

### Three strands braided
Strand 1 = semantic content (what it is), Strand 2 = logical affordance (what it means — implies, contradicts, enables), Strand 3 = activation context (when it fires). Current agents have three loose ropes running parallel. Braiding means each crossing reinforces the others.

### Perception → Context → Intention
From user's BJJ pedagogy. The right cognitive order: perceive raw signal → let context fire relevant patterns → form and execute intention. Current agents invert this. Trainable via traces that encode all three layers.

### Diagnostic Chunking Hypothesis
High-entropy tokens degrade in 1-bit models when processed individually. Binding related evidence into semantic chunks before reasoning preserves substance — like speed reading 3-4 words as a single unit. Confirmed by results: devops improved with chunking, research_synthesis hit 100%. The concept arrives intact before entropy diffuses the specifics.

### Cognitive Mode Controller
Discovered empirically: a light classifier that picks the right reasoning structure per task type eliminates mode-mismatch regressions and enables large gains. This is NOT just "more data" — it's matching the shape of thought to the shape of the problem.

### Specificity as Universal Bottleneck (confirmed at scale)
3D scoring across 500 traces: Direction 9.18, Completion 9.17, Specificity 8.21. The gap is ~1.0 point, consistent across all domains. Bonsai knows what to do and finishes the task, but misses the exact discriminator, command, or edge case. Training should increase exactness, not elaboration.

### Key finding: 47% pass / 45% partial / 6% fail
Nearly half of all traces are "almost right." The model rarely fails outright — it consistently enters the right problem family and produces a complete answer, but lacks precision. This is the most actionable signal for training: the delta between Bonsai's partial attempts and Codex's gold corrections IS the training signal.

## Next Steps

### 1. Accumulate corrected data
The pipeline now produces properly structured traces with 3D scores, domain schemas, and classified failure types. Let the loop keep generating local outputs and only retain the curated/public-safe subset in GitHub. The full build can grow locally as needed.

### 2. Analyze failure type distribution
With partial_type now classified on every trace, run analyze_sprint.py to see which failure types dominate each frontier domain. This will reveal whether devops is primarily weak_specificity (exact commands) vs weak_verification (confirmation steps), and similarly for code_debugging and logic_puzzle.

### 3. Domain-specific schema refinement
The GPT collaboration proposed additional schema fields that target specificity directly:
- devops: `best_discriminator`, `expected_result_if_true/false`
- code_debugging: `secondary_check`, `exact_line_or_condition`
- logic_puzzle: `critical_discriminator`, `state_update`, `consistency_check`
These could be merged into the existing schemas in codex_distill() to push Codex toward even more precise gold traces.

### 4. Inference-ready memory architecture
The biggest unrealized conceptual piece. Memory entries should support not just recall but inference — abstractions, heuristics, transfer triggers, logic hooks, confidence. This is the "three strands braided" thesis made concrete. Design the schema, then train Bonsai to produce and consume it.

### 5. Agent_routing investigation
The one domain still showing instability. Now that partial_type is classified, check whether routing failures are weak_specificity (wrong handler) or wrong_ranking (right handlers but wrong priority). This determines whether to adjust the routing schema or the task distribution.

### 6. Training step under Hermes control
When the collected outputs are large enough and the failure patterns are well-characterized, Hermes orchestrates the actual weight update through Codex — not a standalone script. The mechanism for this (how Hermes triggers and manages fine-tuning) still needs to be designed.
