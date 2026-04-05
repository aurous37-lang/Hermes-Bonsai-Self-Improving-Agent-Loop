# Pre-Fine-Tune Analysis & Recommendations

## Sprint Results (5,186 entries, 169 iterations)

### Pipeline Quality (improved across all eras)
| Era | Iterations | Validation | Correctness |
|-----|:---:|:---:|:---:|
| Pre-cognitive (<90) | 90 | 90.1% | 9.1 |
| Cognitive modes (90-115) | 26 | 94.6% | 9.1 |
| Completion discipline (116+) | 53 | 95.0% | 9.2 |

### 3D Scoring Reveals the Real Bottleneck
Direction and completion are strong across all domains. **Specificity is the universal weak point.**

| Domain | Direction | Specificity | Completion |
|--------|:---------:|:-----------:|:----------:|
| devops | 9.1 | **8.5** | 9.1 |
| logic_puzzle | **8.7** | **8.6** | 9.0 |
| code_debugging | 9.6 | **9.0** | 9.3 |
| architecture | 9.7 | **8.4** | 9.7 |
| agent_routing | 9.9 | **8.6** | 9.8 |

**Key insight:** "Bonsai's dominant bottleneck is specificity, not general reasoning structure. It usually identifies the correct task family and produces a complete answer, but loses precision at the level of exact discriminators, commands, and edge cases."

### Investigation Results

**Code debugging regression: RESOLVED — measurement artifact.** Early data was self-distilled with placeholder scores. Actual raw pass rates are flat at ~40% across all eras. The 3D scores (dir 9.4, spec 8.7, comp 9.1) confirm it's not collapsing.

**Agent routing regression: PARTIALLY EXPLAINED.** Routing mode fixed the acute collapse (40%→78% in one window), but the full-sprint aggregate still trends down (87→73→68). Likely a mix of task distribution shift (harder routing tasks) and incomplete mode robustness.

## Four-Bucket Training Strategy

### A. Safe Reinforcement (10% weight)
- refusal_redirect, math, research_synthesis
- Purpose: preserve identity, prevent regression
- These are solved — don't waste gradient budget

### B. Beneficial Structure (25% weight)
- architecture, logic_puzzle, memory_integration, self_correction
- Purpose: lock in cognitive mode gains
- These domains improved under mode engineering — reinforce that

### C. Specificity Training (50% weight)
- devops, code_debugging (the partial-heavy frontier)
- Purpose: increase precision without bloating traces
- Focus on: exact commands, exact edge cases, exact discriminators
- Use token-anchored traces ([CMD:], [CODE:])
- Prioritize examples where Bonsai was "partial" — the delta between its attempt and Codex's correction IS the training signal

### D. Investigation Bucket (15% weight)
- agent_routing
- Purpose: stabilize routing mode, don't over-train on unstable data
- Lower weight until routing mode is more robust

## Training Objective (Precision-First)
1. Preserve high direction scores (already 8.7-10.0)
2. Preserve high completion scores (already 9.0-10.0)
3. **Increase specificity** without bloating traces

This means: do NOT train toward longer, more elaborate answers. Train toward *sharper* answers at the same or shorter length.

## Pre-Finetune Checklist
- [x] 5,186 entries accumulated
- [x] 3D scoring propagating (42/50 recent traces)
- [x] Code debugging regression investigated — measurement artifact
- [x] Baseline eval captured (0% format compliance pre-training)
- [ ] Fix partial_type propagation (in pipeline, not yet in gold traces)
- [ ] Build weighted training set with build_training_set.py
- [ ] Freeze a 200-entry validation set by domain for consistent measurement
- [ ] Run Stage 1 fine-tune
- [ ] Run post-training eval and compare
