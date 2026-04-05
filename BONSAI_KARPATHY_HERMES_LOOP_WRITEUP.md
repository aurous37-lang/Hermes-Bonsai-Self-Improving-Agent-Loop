# Bonsai–Karpathy–Hermes Loop Write-Up

This experiment was a fun, practical exploration of a self-improving local-agent loop: Bonsai generated traces, Hermes orchestrated the workflow, and a stronger teacher model distilled and scored the outputs.

The goal was not just to make a model answer better, but to improve the entire loop: data generation, trace quality, curriculum selection, validation, and fallback recovery when the training stack hit platform-specific problems.

## System we used

- Hermes as the orchestrator and loop controller
- Bonsai as the local student model generating raw reasoning traces
- Codex as the teacher / evaluator for distillation and scoring
- Karpathy-style auto-research framing to let the system discover and attack its own weak spots
- Weighted datasets and staged training artifacts to turn validated traces into training-ready corpora
- TurboQuant / local inference serving to keep the student model usable on consumer hardware
- A manual recovery path for CUDA / Unsloth / export issues when the default path became unreliable

## What worked

1. Loop-native data generation
   - The system could generate its own tasks, attempt them, score them, and recycle the best traces back into training.
   - This made the data flywheel much more useful than a hand-crafted sample set alone.

2. Teacher-guided correction
   - Codex-as-teacher was effective at compressing and correcting traces.
   - The strongest gains came from turning raw passes, hint passes, and full corrections into structured supervision.

3. Schema and scoring fixes
   - Moving to the domain-specific trace schema and the 3D scoring scheme improved consistency.
   - The loop became much better at distinguishing direction, specificity, and completion instead of flattening everything into one score.

4. Failure-taxonomy feedback
   - Labeling partial failures made the curriculum smarter.
   - Instead of treating every miss the same, the loop could focus on weak specificity, incomplete coverage, wrong ranking, and weak verification separately.

5. Manual fallback recovery
   - When the CUDA / Unsloth path became unreliable on the target hardware, the manual build-and-convert path kept the project moving.
   - Training could still progress by saving the adapter checkpoint and performing GGUF conversion separately.

6. Rebalanced datasets
   - Rebalancing the dataset improved coverage across domains that were otherwise underrepresented.
   - This was especially important when one domain became over-dominant in the raw-pass pool.

## What did not work well

1. The default CUDA / training stack path
   - On the target GPU architecture, the standard training stack hit compatibility problems.
   - That made the “just run the normal training script” path unreliable enough that a fallback became necessary.

2. Overly rigid early schemas
   - Hardcoded generic trace schemas and older score formats conflicted with the newer domain-specific structure.
   - That caused the teacher to follow the wrong pattern too often until the prompt and validation logic were fixed.

3. Raw-pass scarcity in some windows
   - Some batches had much lower raw-pass yield than expected.
   - That meant the dataset needed careful sampling and rebalancing instead of naive accumulation.

4. Domain skew
   - The later sprint data was heavily skewed toward self-correction.
   - Without reweighting, the final dataset would have overfit that one behavior class.

5. Checklist presence as a success signal
   - We learned not to treat checklist-shaped output as proof of success by itself.
   - It was often a correction-path marker, not a raw-pass marker.

## Improvements we made

1. Closed the teacher/student feedback loop
   - Weakness analysis now feeds directly into the next round of task generation.
   - That makes the curriculum adaptive instead of static.

2. Added domain-sensitive trace handling
   - Different task families now use different reasoning structures.
   - This reduced mode mismatch and improved consistency across task types.

3. Added a richer evaluation lens
   - The 3D scoring model gave a more useful picture of model behavior.
   - Direction, specificity, and completion exposed different failure modes that a single score would hide.

4. Improved partial-failure handling
   - Partial outputs are now classified instead of being discarded as generic failures.
   - That turned “almost right” examples into useful training signal.

5. Rebalanced the training dataset
   - We deduplicated overlapping files and projected the distribution across raw passes, hint passes, and full corrections.
   - That helped prevent the dataset from being dominated by one narrow domain.

6. Established a safer fallback build path
   - Instead of waiting on a brittle stack to behave, we documented a manual recovery path.
   - The fallback keeps the loop moving without abandoning the overall architecture.

## Sanitization and publication notes

For any public release, the following should stay redacted or generalized:

- API keys, tokens, and auth material
- local absolute file paths
- personal email addresses
- machine-specific usernames and workspace paths
- private config file locations
- any raw training examples that contain real contact details or other sensitive identifiers

The goal is to share the method, not the private environment.

## Why this matters

The main takeaway is that useful agent improvement does not come only from bigger models or more data. It also comes from:

- better task selection
- better feedback labels
- better trace structure
- better fallback recovery
- better dataset balance

In other words, the loop itself is part of the model.

## Credit

This work builds on ideas and tooling from:

- Andrej Karpathy and the broader Karpathy auto-research framing
- The Bonsai model/runtime effort that supplied the local student model
- The Hermes project and its contributors/maintainers
- OpenAI / Codex as the teacher and evaluator in the distillation loop
- The open-source ML and inference ecosystem that made the local stack possible

## Closing note

This experiment was done purely for fun, curiosity, and learning. If the community wants to continue it, the most useful next step is to keep the loop small, keep the evaluation honest, and keep the fallback path documented so others do not lose time repeating the same environment failures.
