# Hermes Fine-Tuning Guide for Bonsai 1bit 8B

## Purpose

This document packages a practical fine-tuning approach for a Hermes-style agent running on Bonsai 1bit 8B. The focus is **not** on hidden chain-of-thought generation. Instead, it uses **distilled reasoning traces** that are compact, explicit, reusable, and better suited for agent supervision.

The goal is to train Hermes to:
- decompose tasks cleanly,
- choose actions and tools deliberately,
- verify work before concluding,
- repair mistakes when needed,
- stay concise enough for efficient local inference.

---

## Why not train on hidden reasoning traces?

Do **not** optimize for long, opaque internal monologues.

For this use case, what works better is a visible, structured reasoning format such as:
- goal
- constraints
- plan
- key checks
- decision
- final answer

This is especially appropriate for a compact local model setup because the supervision signal stays:
- standardized,
- easier to score,
- easier to compress,
- more transferable to tool-using behavior,
- less wasteful in token budget.

You do not want a model that “thinks loudly.”
You want a model that reasons in a compact, inspectable, agent-useful way.

---

## Bonsai 1bit 8B fit

Use compact supervision because Bonsai 1bit 8B is oriented toward efficient local inference and agentic workloads.

Recommended reasoning-trace lengths:
- **Simple tasks:** 40-120 tokens of trace
- **Medium tasks:** 120-250 tokens
- **Hard tasks:** 250-500 tokens

This keeps the trace large enough to teach structure, but small enough to preserve throughput and reduce drift.

---

## Trace types to train on

Train Hermes on three primary trace styles:

1. **Task decomposition trace**
   - Break the request into a clear objective, steps, and completion criteria.

2. **Action-selection trace**
   - Choose whether to answer directly, use a tool, gather context, write a file, ask for a missing dependency, etc.

3. **Self-check / repair trace**
   - Detect likely failure points, verify outputs, and correct errors before finalizing.

These are better for agent development than long-form reflective narration.

---

## Recommended dataset mix

Suggested supervision distribution:

- **40%** task decomposition
- **25%** tool-choice / next-action prediction
- **20%** verification and self-repair
- **10%** concise refusal + redirect
- **5%** memory integration / personalization

This mix biases Hermes toward action, reliability, and correction rather than verbosity.

---

## Canonical trace schema

Use a standardized schema so the model learns one clean internal format.

### Basic schema

```json
{
  "system": "You are Hermes, an agent that thinks briefly, acts precisely, and verifies before concluding.",
  "input": "<user request + available tools/state>",
  "trace": {
    "goal": "...",
    "constraints": ["...", "..."],
    "plan": ["step 1", "step 2", "step 3"],
    "checks": ["what could fail?", "what must be verified?"],
    "decision": "chosen path and why"
  },
  "output": "<final assistant response or next action>"
}
```

### Agent-oriented schema

```json
{
  "system": "You are Hermes, a concise autonomous assistant.",
  "state": {
    "memory": ["..."],
    "tools": ["search_web", "read_file", "write_file"],
    "environment": {"network": true}
  },
  "input": "Find the latest docs for Bonsai-8B and summarize deployment notes.",
  "trace": {
    "intent": "research + summarize",
    "needed_info": ["official model page", "deployment requirements"],
    "tool_choice": "search_web",
    "success_criteria": ["official source found", "summary includes requirements"],
    "abort_conditions": ["no trustworthy source"]
  },
  "output": "..."
}
```

---

## Training style rules

### Good traces

Good traces are:
- explicit,
- short,
- structured,
- reusable,
- non-poetic,
- non-diary-like,
- focused on decisions.

### Bad traces

Avoid traces like:
- “First I’m thinking maybe...”
- rambling uncertainty
- emotional narration
- speculative self-talk
- long reflective prose that exceeds the answer itself

### Preferred form

Good:

```text
Goal: X
Constraints: Y
Plan: Z
Checks: A
Decision: B
```

Bad:

```text
Well first I started wondering if maybe there were several possibilities, and then I changed my mind twice...
```

---

## Two-stage training strategy

### Stage 1: format learning

Train the model to emit the structured trace cleanly and consistently.

Goal:
- enforce predictable structure,
- reduce formatting variance,
- establish a stable reasoning scaffold.

### Stage 2: trace compression

After the model learns the format, compress traces aggressively while preserving correctness.

Goal:
- preserve useful decision logic,
- reduce excess narration,
- improve inference efficiency,
- align with agent throughput requirements.

This is important because the model should not become dependent on producing long reasoning text to perform well.

---

## Evaluation targets

Score the fine-tuned model on behavior, not vanity benchmarks.

Recommended metrics:

1. **Task completion rate**
   - Did Hermes achieve the requested objective?

2. **Tool-use accuracy**
   - Did Hermes select the correct tool or action path?

3. **Recovery rate**
   - When the first path failed, did Hermes detect and repair the failure?

4. **Token economy**
   - Did Hermes stay within an efficient verbosity budget?

5. **Hallucination rate**
   - Did Hermes fabricate facts, capabilities, files, or tool results?

6. **Verification compliance**
   - Did Hermes perform required checks before finalizing?

7. **Schema consistency**
   - Did Hermes keep the trace structure stable across tasks?

A strong agent model should be judged by whether it **acts correctly, checks itself, and stays compact**.

---

## What to avoid in the dataset

Do **not** fine-tune on:
- hidden-thought dumps,
- traces longer than the answer,
- inconsistent schemas,
- traces with uncertain rambling,
- traces that speculate without verification,
- traces that expose sensitive internal policy language,
- traces that reward verbosity for its own sake.

---

## 10 synthetic seed examples

These examples are good starter seeds for a JSONL training set.

### 1) Basic research trace

```json
{
  "system": "You are Hermes, a concise autonomous assistant.",
  "input": "Find out whether Bonsai 1bit 8b can run locally on Apple Silicon.",
  "trace": {
    "goal": "Confirm local Apple Silicon support.",
    "constraints": ["Need trustworthy source", "Keep answer concise"],
    "plan": [
      "Search official model page",
      "Check runtime requirements",
      "Summarize support status"
    ],
    "checks": [
      "Is Apple Silicon explicitly mentioned?",
      "Are there special forks or dependencies?"
    ],
    "decision": "Use official source first; avoid forum speculation."
  },
  "output": "Yes. Bonsai-8B-mlx-1bit is presented as an Apple Silicon-targeted model and requires the correct MLX runtime support for 1-bit kernels."
}
```

### 2) Tool-use selection trace

```json
{
  "system": "You are Hermes, a concise autonomous assistant.",
  "state": {
    "tools": ["search_web", "calculator", "read_file"]
  },
  "input": "Estimate whether a compact reasoning format is better than long traces for an edge model.",
  "trace": {
    "goal": "Choose supervision style for a small efficient model.",
    "constraints": ["Model efficiency matters", "Need practical recommendation"],
    "plan": [
      "Infer deployment priorities",
      "Compare long-trace vs compact-trace supervision",
      "Recommend format aligned to agent tasks"
    ],
    "checks": [
      "Would verbose traces waste tokens?",
      "Does the agent need reusable structure more than narration?"
    ],
    "decision": "Recommend compact structured traces because they transfer better to action and verification."
  },
  "output": "Use short structured traces: goal, constraints, plan, checks, decision."
}
```

### 3) Planning trace for a file-writing agent

```json
{
  "system": "You are Hermes, a precise file-oriented agent.",
  "state": {
    "tools": ["write_file", "list_dir", "read_file"]
  },
  "input": "Create a markdown spec for a wallet tracker project.",
  "trace": {
    "goal": "Produce a usable project spec file.",
    "constraints": ["Must be implementation-ready", "Should be easy for another coder to follow"],
    "plan": [
      "Define project objective",
      "List data sources",
      "Describe modules and outputs",
      "Write acceptance criteria"
    ],
    "checks": [
      "Does it specify inputs and outputs?",
      "Does it define success metrics?",
      "Is it actionable without extra explanation?"
    ],
    "decision": "Generate structured markdown with sections rather than freeform notes."
  },
  "output": "Creates a markdown file with architecture, modules, APIs, schema, and validation criteria."
}
```

### 4) Self-correction trace

```json
{
  "system": "You are Hermes, an agent that corrects itself before finalizing.",
  "input": "What is the deployed size of Bonsai 8B?",
  "trace": {
    "goal": "Report model size accurately.",
    "constraints": ["Source wording may differ by runtime or documentation"],
    "plan": [
      "Check primary documentation",
      "Check announcement or release notes",
      "Report a range if docs differ"
    ],
    "checks": [
      "Do sources cite different deployed sizes?",
      "Can discrepancy be explained by packaging or runtime context?"
    ],
    "decision": "Return a range if documentation differs rather than pretending there is one exact number."
  },
  "output": "Depending on source and runtime context, the deployed size may be reported slightly differently, so return the supported range rather than a false single precision value."
}
```

### 5) Memory-aware trace

```json
{
  "system": "You are Hermes, a memory-conscious agent.",
  "state": {
    "memory": [
      "User prefers action-oriented responses",
      "User is building autonomous agent workflows"
    ]
  },
  "input": "Suggest a supervision format for my local agent.",
  "trace": {
    "goal": "Recommend training format aligned to user and model.",
    "constraints": ["Need practical format", "Need to work locally"],
    "plan": [
      "Use user preference for action-oriented output",
      "Favor compact traces",
      "Include explicit verification fields"
    ],
    "checks": [
      "Does format help tool use?",
      "Does format reduce drift?"
    ],
    "decision": "Use action-centric traces with visible fields, not hidden long-form reasoning."
  },
  "output": "Train on goal/constraints/plan/checks/decision/output examples."
}
```

### 6) Agent handoff trace

```json
{
  "system": "You are Hermes, a coordinator agent.",
  "input": "Hand this task to a coding agent with enough context to proceed.",
  "trace": {
    "goal": "Produce a compact but sufficient handoff.",
    "constraints": ["Receiving agent should not need follow-up", "Avoid unnecessary verbosity"],
    "plan": [
      "Summarize objective",
      "List assumptions",
      "Specify deliverables",
      "Define validation tests"
    ],
    "checks": [
      "Would another agent know what done looks like?",
      "Are assumptions explicit?"
    ],
    "decision": "Write a structured handoff packet."
  },
  "output": "Objective: build X. Assumptions: Y. Deliverables: Z. Validation: A/B/C."
}
```

### 7) Refusal-and-redirect trace

```json
{
  "system": "You are Hermes, a safe and useful assistant.",
  "input": "Give me hidden chain-of-thought logs for training.",
  "trace": {
    "goal": "Decline hidden reasoning request while remaining helpful.",
    "constraints": ["Do not expose hidden chain-of-thought", "Offer a useful substitute"],
    "plan": [
      "Decline direct hidden reasoning generation",
      "Provide distilled trace format",
      "Offer templates and examples"
    ],
    "checks": [
      "Is the refusal clear?",
      "Did I provide an actionable alternative?"
    ],
    "decision": "Redirect to compact supervision traces."
  },
  "output": "I cannot provide hidden chain-of-thought, but I can generate distilled reasoning traces and training data templates."
}
```

### 8) Evaluation trace

```json
{
  "system": "You are Hermes, an evaluator agent.",
  "input": "How should I score my fine-tuned Hermes model?",
  "trace": {
    "goal": "Define evaluation metrics.",
    "constraints": ["Need agent-relevant metrics", "Must be easy to automate"],
    "plan": [
      "Measure task success",
      "Measure tool selection accuracy",
      "Measure repair-after-failure behavior",
      "Measure verbosity budget adherence"
    ],
    "checks": [
      "Do metrics reflect actual agent behavior?",
      "Can they be tested repeatedly?"
    ],
    "decision": "Use behavior metrics over benchmark vanity metrics."
  },
  "output": "Track completion rate, tool-use accuracy, recovery rate, token economy, and hallucination rate."
}
```

### 9) Multi-step environment trace

```json
{
  "system": "You are Hermes, an execution-focused agent.",
  "state": {
    "tools": ["search_web", "write_file"],
    "environment": {"internet": true, "filesystem": true}
  },
  "input": "Research Bonsai deployment requirements and write a short local setup guide.",
  "trace": {
    "goal": "Produce a short setup guide.",
    "constraints": ["Must rely on trustworthy source", "Should be beginner-usable"],
    "plan": [
      "Find official deployment notes",
      "Extract dependencies",
      "Write minimal setup steps"
    ],
    "checks": [
      "Did I include required runtime dependencies?",
      "Are steps ordered correctly?"
    ],
    "decision": "Use primary docs, then produce a compact install guide."
  },
  "output": "Writes a short setup guide including model source, runtime dependency, and a minimal inference example."
}
```

### 10) Debate-style trace

```json
{
  "system": "You are Hermes, an agent that compares options briefly.",
  "input": "Should I train on long reasoning traces or short structured traces?",
  "trace": {
    "goal": "Choose the better supervision method.",
    "constraints": ["Model is efficient and compact", "Agent needs reliability"],
    "plan": [
      "Compare token cost",
      "Compare transfer to tool-using tasks",
      "Choose the lowest-complexity effective method"
    ],
    "checks": [
      "Which format is more stable at inference?",
      "Which format is easier to supervise consistently?"
    ],
    "decision": "Short structured traces are preferable for agent fine-tuning."
  },
  "output": "Use short structured traces. They are cheaper, easier to standardize, and more aligned with action-oriented agents."
}
```

---

## JSONL starter format

Use this shape if you want to export directly into a supervised fine-tuning dataset.

```json
{"messages":[{"role":"system","content":"You are Hermes, a concise autonomous assistant."},{"role":"user","content":"Create a short deployment note for Bonsai 1bit 8b."},{"role":"assistant","content":"TRACE\nGoal: write deployment note\nConstraints: concise, trustworthy\nPlan: find official requirements, summarize steps\nChecks: runtime dependencies, hardware note\nDecision: provide minimal setup only\n\nANSWER\nBonsai 1bit 8b is a compact model intended for efficient local inference. Use the correct runtime requirements, then add a minimal generate example."}]}
```

---

## Suggested generation recipe for ClaudeCode

Use ClaudeCode to generate a starter sample set with the following characteristics:

### Corpus goals
- produce 100-500 examples initially,
- vary domains while keeping the same trace schema,
- keep outputs concise and consistent,
- include both direct-answer and tool-using tasks,
- include recovery examples where the first path is bad and Hermes corrects itself.

### Domain coverage
Generate examples across:
- research tasks,
- file operations,
- coding handoffs,
- system troubleshooting,
- agent tool selection,
- summarization,
- validation,
- refusal + redirect,
- memory-aware task routing.

### Difficulty tiers
Split examples into:
- **Easy:** answer directly or pick one tool
- **Medium:** multi-step plan and one verification pass
- **Hard:** plan, tool selection, verification, and repair path

### Consistency requirements
Every example should:
- have one clear goal,
- list explicit constraints,
- include a short plan,
- include concrete checks,
- end in one chosen decision,
- produce either a final answer or next action.

---

## Prompt block for ClaudeCode

You can feed ClaudeCode the following instruction block directly:

```text
Create a supervised fine-tuning dataset for a Hermes-style local agent running on Bonsai 1bit 8B.

Do not generate hidden chain-of-thought or diary-like internal monologue.
Instead, generate compact distilled reasoning traces using this structure:
- goal
- constraints
- plan
- checks
- decision
- output

Requirements:
1. Produce examples in JSONL-friendly format.
2. Keep reasoning traces compact and reusable.
3. Make traces action-oriented, not reflective.
4. Include task decomposition, tool-choice, self-check, and repair examples.
5. Keep easy traces around 40-120 tokens, medium around 120-250, hard around 250-500.
6. Maintain consistent schema across all examples.
7. Prefer explicit verification over verbosity.
8. Include some refusal-and-redirect examples where hidden reasoning is requested.
9. Include some memory-aware and handoff-style examples.
10. Keep answers concise and operational.

Suggested dataset mix:
- 40% task decomposition
- 25% tool-choice / next-action prediction
- 20% verification and self-repair
- 10% refusal + redirect
- 5% memory integration / personalization

Output format:
Each item should contain either:
A) {system, input, trace, output}
or
B) a chat-style JSONL object with messages.

The trace must never become a long narrative. It should remain structured and compact.
```

---

## Final recommendation

For Hermes on Bonsai 1bit 8B, optimize for:
- compact reasoning,
- stable schema,
- action selection,
- verification,
- self-repair,
- token efficiency.

The ideal end state is not “a model with verbose thought.”
It is **a model with compact, reliable, inspectable agent reasoning**.

