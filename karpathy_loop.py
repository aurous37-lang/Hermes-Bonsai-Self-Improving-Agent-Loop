#!/usr/bin/env python3
"""
Karpathy Auto-Research Loop for Hermes/Bonsai Training

Three-stage iterative pipeline:
  Stage 1: Hermes (Bonsai 8B @ localhost:1234) generates raw reasoning traces
  Stage 2: Strong model (Claude/Opus) distills gold-standard compact traces
  Stage 3: Compile validated traces into training-ready JSONL + track metrics

Usage:
  python karpathy_loop.py --pilot          # Run 5-task pilot
  python karpathy_loop.py --iteration N    # Run N tasks from queue
  python karpathy_loop.py --full           # Run all pending tasks
  python karpathy_loop.py --report         # Show metrics summary
"""

import json
import os
import sys
import time
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
TASK_QUEUE_PATH = BASE_DIR / "task_queue.json"
OUTPUT_DIR = BASE_DIR / "karpathy_loop_output"
RAW_DIR = OUTPUT_DIR / "raw_traces"
GOLD_DIR = OUTPUT_DIR / "gold_traces"
BATCH_DIR = OUTPUT_DIR / "training_batches"
METRICS_DIR = BASE_DIR / "karpathy_loop_metrics"
TRAINING_JSONL = OUTPUT_DIR / "training_data.jsonl"
ITERATION_LOG = METRICS_DIR / "iteration_log.json"

HERMES_API = "http://localhost:1234/v1"
HERMES_MODEL = "bonsai-8b"

# Trace token targets by difficulty
TOKEN_TARGETS = {"easy": (40, 120), "medium": (120, 250), "hard": (250, 500)}

# Dataset mix targets
MIX_TARGETS = {
    "task_decomposition": 0.40,
    "tool_choice": 0.25,
    "self_repair": 0.20,
    "refusal": 0.10,
    "memory": 0.05,
}

# System prompts
HERMES_SYSTEM = (
    "You are Hermes, a reasoning agent. Think step-by-step inside <think> tags "
    "before giving your answer. Be thorough but concise."
)

DISTILL_SYSTEM = """You are a training-data distiller for a compact 8B agent model called Hermes/Bonsai.

Your job: take a task prompt and a raw reasoning attempt, then produce a GOLD-STANDARD compact reasoning trace.

Rules:
- Output ONLY valid JSON. No markdown fences, no extra text.
- The user prompt specifies the EXACT JSON schema to use — follow it exactly.
- Traces MUST be compact. Token budgets: easy=40-120, medium=120-250, hard=250-500.
- The <think> version should mirror the structured trace but in prose form.
- Score honestly. A 10 means production-ready.
- CORRECTNESS is the most important score. A concise wrong answer is worthless.
- If the raw attempt was actually good, set raw_attempt_quality to "pass" — that's training signal.
- For partial/fail attempts, you MUST specify partial_type (weak_specificity, incomplete_coverage, wrong_ranking, or weak_verification).
- Wrap specific commands in [CMD: ...] and code references in [CODE: ...] anchors."""


# ---------------------------------------------------------------------------
# API Client (OpenAI-compatible)
# ---------------------------------------------------------------------------
class LLMClient:
    """Minimal OpenAI-compatible API client — no external deps required."""

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def chat(self, messages: list[dict], temperature: float = 0.7,
             max_tokens: int = 4096, timeout: int = 120) -> dict:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            return {
                "content": data["choices"][0]["message"]["content"],
                "usage": data.get("usage", {}),
                "model": data.get("model", self.model),
            }
        except urllib.error.URLError as e:
            return {"content": None, "error": str(e), "usage": {}}
        except Exception as e:
            return {"content": None, "error": str(e), "usage": {}}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TaskResult:
    task_id: str
    domain: str
    difficulty: str
    category: str
    prompt: str
    raw_trace: str
    raw_tokens: int
    raw_time_s: float
    gold_trace: Optional[dict] = None
    gold_think: Optional[str] = None
    gold_output: Optional[str] = None
    scores: Optional[dict] = None
    partial_type: Optional[str] = None
    minimal_hint: Optional[str] = None
    false_confidence: bool = False
    raw_quality: str = "pending"
    validated: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


FALSE_CONFIDENCE_MARKERS = (
    "covers all",
    "all requirements",
    "complete",
    "fully",
    "nothing missing",
    "no gaps",
    "definitely",
    "certainly",
    "absolutely",
    "clearly",
    "obviously",
    "i'm confident",
    "i am confident",
    "this answers",
    "that covers",
    "done",
    "complete answer",
)

FALSE_CONFIDENCE_HEDGES = (
    "i think",
    "i believe",
    "maybe",
    "might",
    "possibly",
    "probably",
    "uncertain",
    "not sure",
    "i'm not sure",
    "i am not sure",
    "could be",
)


def _detect_false_confidence(raw_trace: str, raw_quality: str) -> bool:
    """Return True when the raw attempt is wrong/incomplete but still sounds certain."""
    if raw_quality == "pass":
        return False
    text = (raw_trace or "").lower()
    if any(h in text for h in FALSE_CONFIDENCE_HEDGES):
        return False
    return any(marker in text for marker in FALSE_CONFIDENCE_MARKERS)


def _build_requirements_checklist(domain: str, partial_type: Optional[str] = None) -> str:
    """Build a compact 3-item requirements checklist for corrected examples."""
    domain = domain or ""
    pt = partial_type or ""
    if domain == "memory_integration":
        items = [
            "Include every remembered value exactly as given.",
            "Preserve the requested order and formatting.",
            "Do not omit any requested field or detail.",
        ]
    elif domain == "refusal_redirect":
        items = [
            "Refuse the harmful or banned request clearly.",
            "Use the approved safe redirection phrase or alternative.",
            "Add no extra unsafe guidance or unnecessary explanation.",
        ]
    elif domain == "self_correction":
        items = [
            "Correct the previous mistake directly.",
            "Return the exact replacement command or phrasing.",
            "Do not keep the earlier incorrect form.",
        ]
    else:
        items = [
            "Cover all requested parts of the task.",
            "Use concrete values, commands, or labels.",
            "Include a final verification or completion check if needed.",
        ]

    if pt == "weak_specificity":
        items[0] = "Use the exact values, labels, or command text requested."
    elif pt == "incomplete_coverage":
        items[1] = "Cover every required branch, step, or item before finalizing."
    elif pt == "weak_verification":
        items[2] = "End with one concrete verification or completion check."

    return "[REQUIREMENTS]\n" + "\n".join(f"- {item}" for item in items[:3]) + "\n[/REQUIREMENTS]"


# ---------------------------------------------------------------------------
# Stage 1: Research & Generate (Hermes / Bonsai)
# ---------------------------------------------------------------------------
def stage1_generate(client: LLMClient, task: dict) -> TaskResult:
    """Send task to Hermes and capture raw reasoning trace."""
    messages = [
        {"role": "system", "content": HERMES_SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]

    print(f"  [Stage 1] Generating raw trace for {task['id']}...", end=" ", flush=True)
    t0 = time.time()
    resp = client.chat(messages, temperature=0.6, max_tokens=2048, timeout=180)
    elapsed = time.time() - t0

    content = resp.get("content") or f"[ERROR] {resp.get('error', 'unknown')}"
    usage = resp.get("usage", {})
    completion_tokens = usage.get("completion_tokens", len(content.split()))

    print(f"done ({elapsed:.1f}s, ~{completion_tokens} tokens)")

    result = TaskResult(
        task_id=task["id"],
        domain=task["domain"],
        difficulty=task["difficulty"],
        category=task["category"],
        prompt=task["prompt"],
        raw_trace=content,
        raw_tokens=completion_tokens,
        raw_time_s=round(elapsed, 2),
    )

    # Save raw trace
    raw_path = RAW_DIR / f"{task['id']}_raw.json"
    raw_path.write_text(json.dumps(asdict(result), indent=2))

    return result


# ---------------------------------------------------------------------------
# Stage 2: Distill & Refine (Strong model — called externally or via API)
# ---------------------------------------------------------------------------
def stage2_distill_local(result: TaskResult) -> TaskResult:
    """
    Distill using a local strong-model call.
    If no strong model API is available, generates a structured prompt
    that can be fed to Claude Code or any strong model manually.
    """
    distill_prompt = f"""Task difficulty: {result.difficulty}
Token budget for trace: {TOKEN_TARGETS[result.difficulty][0]}-{TOKEN_TARGETS[result.difficulty][1]} tokens

TASK PROMPT:
{result.prompt}

RAW ATTEMPT FROM BONSAI-8B:
{result.raw_trace}

Produce the gold-standard distilled trace as specified."""

    # Try to call a local strong model if available (e.g., a second model on another port)
    # Fall back to writing a distillation request file for manual processing
    distill_path = GOLD_DIR / f"{result.task_id}_distill_request.json"
    distill_path.write_text(json.dumps({
        "system": DISTILL_SYSTEM,
        "user": distill_prompt,
        "task_id": result.task_id,
        "difficulty": result.difficulty,
    }, indent=2))

    return result


def stage2_apply_gold(result: TaskResult, gold_json: dict) -> TaskResult:
    """Apply a gold trace (from manual distillation or API) to a TaskResult."""
    result.gold_trace = gold_json.get("trace")
    result.gold_think = gold_json.get("think_tagged")
    result.gold_output = gold_json.get("output")
    result.scores = gold_json.get("score")
    result.partial_type = gold_json.get("partial_type", "n/a")
    result.minimal_hint = gold_json.get("minimal_hint", "n/a")
    result.raw_quality = gold_json.get("raw_attempt_quality", "unknown")
    result.false_confidence = bool(gold_json.get("false_confidence", _detect_false_confidence(result.raw_trace, result.raw_quality)))

    # Normalize old 4-field scores to 3D schema
    if result.scores and "direction" not in result.scores and "correctness" in result.scores:
        result.scores = {
            "correctness": result.scores.get("correctness", 5),
            "direction": result.scores.get("correctness", 5),
            "specificity": result.scores.get("actionability", 5),
            "completion": result.scores.get("schema_consistency", 5),
        }

    # Validate — handle both old (correctness) and new (direction/specificity/completion) schemas
    if result.scores:
        avg_score = sum(result.scores.values()) / len(result.scores)
        # New 3D schema: direction + specificity + completion
        if "direction" in result.scores:
            result.validated = (
                result.scores.get("direction", 0) >= 7
                and result.scores.get("specificity", 0) >= 5
                and avg_score >= 6.5
            )
        # Old schema: correctness
        else:
            result.validated = (
                result.scores.get("correctness", 0) >= 7
                and avg_score >= 6.5
            )

    # Prepend a compact requirements checklist for corrected examples
    if result.raw_quality in ("partial", "fail") and result.gold_think:
        checklist = _build_requirements_checklist(result.domain, result.partial_type)
        if not result.gold_think.lstrip().startswith("[REQUIREMENTS]"):
            result.gold_think = f"{checklist}\n\n{result.gold_think}"

    # Save gold trace — include partial_type if present
    data = asdict(result)
    data["partial_type"] = gold_json.get("partial_type", "n/a")
    data["false_confidence"] = result.false_confidence
    gold_path = GOLD_DIR / f"{result.task_id}_gold.json"
    gold_path.write_text(json.dumps(data, indent=2))

    return result


def _extract_json(text: str) -> dict:
    """Robustly extract the first complete JSON object from text.
    Handles: markdown fences, trailing text, multiple objects, nested braces."""
    # Strip markdown fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    cleaned = cleaned.strip()

    # Find the first { and brace-match to find the end
    start = cleaned.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", cleaned, 0)

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i + 1]
                return json.loads(candidate)

    # Brace matching didn't complete — try repairing truncated JSON
    remainder = cleaned[start:]

    # First try: sanitize invalid escape sequences (common with code in JSON)
    import re
    sanitized = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', remainder)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass

    # Add missing closing braces
    for repair in range(1, 4):
        try:
            return json.loads(sanitized + "}" * repair)
        except json.JSONDecodeError:
            continue
    # Also try closing string + braces
    for repair in range(1, 4):
        try:
            return json.loads(sanitized + '"' + "}" * repair)
        except json.JSONDecodeError:
            continue

    # Last resort: try the whole thing
    return json.loads(cleaned)


def _self_score(client: LLMClient, prompt: str, gold: dict, difficulty: str) -> dict:
    """Have Bonsai score its own compressed trace for quality."""
    tmin, tmax = TOKEN_TARGETS[difficulty]
    score_prompt = f"""Score this compressed reasoning trace on 3 dimensions (0-10 each).
Be HARSH — only give 8+ if it's genuinely excellent.

ORIGINAL TASK: {prompt}

COMPRESSED TRACE:
{json.dumps(gold.get('trace', {}), indent=2)}

OUTPUT: {gold.get('output', '')}

Score as JSON only: {{"direction": N, "specificity": N, "completion": N}}
- direction: Is the answer on the right track and logically aligned with the task?
- specificity: Are the key details precise, concrete, and non-generic?
- completion: Does the trace fully satisfy the required schema and task coverage?"""

    messages = [
        {"role": "system", "content": "You are a strict evaluator. Output ONLY a JSON object with 3 integer scores."},
        {"role": "user", "content": score_prompt},
    ]

    resp = client.chat(messages, temperature=0.2, max_tokens=128, timeout=60)
    content = resp.get("content", "")

    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        scores = json.loads(cleaned.strip())
        # Ensure all keys exist and are ints
        for k in ("direction", "specificity", "completion"):
            scores[k] = int(scores.get(k, 5))
        return scores
    except (json.JSONDecodeError, ValueError):
        return {"direction": 5, "specificity": 5, "completion": 5}


def _load_gold_exemplars(max_examples: int = 2) -> str:
    """Load gold trace examples to use as few-shot context for self-distillation."""
    exemplars = []
    for p in sorted(GOLD_DIR.glob("*_response.json")):
        try:
            gold = json.loads(p.read_text())
            if gold.get("score", {}).get("correctness", 0) >= 9:
                exemplars.append(gold)
        except (json.JSONDecodeError, KeyError):
            continue
        if len(exemplars) >= max_examples:
            break

    if not exemplars:
        return ""

    parts = ["Here are examples of correctly compressed traces:\n"]
    for i, ex in enumerate(exemplars, 1):
        compact = {
            "trace": ex["trace"],
            "output": ex["output"],
        }
        parts.append(f"EXAMPLE {i}:\n{json.dumps(compact)}\n")
    return "\n".join(parts)


def stage2_distill_inline(result: TaskResult) -> TaskResult:
    """
    Self-distill using Bonsai with few-shot gold exemplars.
    Shows Bonsai what good traces look like before asking it to compress.
    """
    client = LLMClient(HERMES_API, HERMES_MODEL)

    exemplars = _load_gold_exemplars(2)

    compress_prompt = f"""Compress the following reasoning into this EXACT JSON format (no markdown, no explanation, ONLY the JSON object):
{{"trace": {{"goal": "...", "constraints": ["..."], "plan": ["..."], "checks": ["..."], "decision": "..."}}, "output": "concise final answer"}}

Rules:
- "goal" is ONE sentence.
- "constraints" is 2-3 short strings.
- "plan" is 2-4 short action steps.
- "checks" is 1-2 verification questions.
- "decision" is ONE sentence explaining the chosen approach.
- "output" is the final answer in 1-3 sentences.
- Total trace must be under {TOKEN_TARGETS[result.difficulty][1]} tokens.
- Output MUST be valid JSON. No trailing commas. No comments.

{exemplars}
TASK TO COMPRESS:
{result.prompt}

RAW REASONING TO COMPRESS:
{result.raw_trace}

Produce ONLY the JSON object. Nothing else."""

    messages = [
        {"role": "system", "content": "You are a JSON compression engine. You take verbose reasoning and compress it into a structured JSON trace. Output ONLY valid JSON — no markdown fences, no explanation, no preamble. Start with {{ and end with }}."},
        {"role": "user", "content": compress_prompt},
    ]

    print(f"  [Stage 2] Self-distilling {result.task_id}...", end=" ", flush=True)
    resp = client.chat(messages, temperature=0.3, max_tokens=2048, timeout=120)
    content = resp.get("content", "")
    print("done")

    # Try to parse the JSON from Bonsai's response
    try:
        gold = _extract_json(content)

        # Build think-tagged version from structured trace
        trace = gold.get("trace", {})
        think_parts = []
        if trace.get("goal"):
            think_parts.append(f"Goal: {trace['goal']}")
        if trace.get("constraints"):
            think_parts.append(f"Constraints: {', '.join(trace['constraints'])}")
        if trace.get("plan"):
            think_parts.append(f"Plan: {' → '.join(trace['plan'])}")
        if trace.get("checks"):
            think_parts.append(f"Checks: {', '.join(trace['checks'])}")
        if trace.get("decision"):
            think_parts.append(f"Decision: {trace['decision']}")

        think_text = "\n".join(think_parts)
        output_text = gold.get("output", "")

        gold["think_tagged"] = f"<think>\n{think_text}\n</think>\n\n{output_text}"

        # Self-score: have Bonsai evaluate its own compressed trace
        gold["score"] = _self_score(client, result.prompt, gold, result.difficulty)
        gold["raw_attempt_quality"] = "self-distilled"

        result = stage2_apply_gold(result, gold)

    except (json.JSONDecodeError, KeyError) as e:
        print(f"    [WARN] Self-distill parse failed for {result.task_id}: {e}")
        # Save the raw distillation attempt for manual review
        fail_path = GOLD_DIR / f"{result.task_id}_distill_failed.txt"
        fail_path.write_text(content)
        result.raw_quality = "distill_failed"

    return result


# ---------------------------------------------------------------------------
# Stage 3: Compile & Iterate
# ---------------------------------------------------------------------------
def compile_training_jsonl(results: list[TaskResult]) -> Path:
    """Write validated results to training-ready JSONL in both formats."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_path = BATCH_DIR / f"batch_{timestamp}.jsonl"

    count = 0
    with open(batch_path, "w") as f, open(TRAINING_JSONL, "a") as master:
        for r in results:
            if not r.validated and r.raw_quality != "self-distilled":
                continue

            # Format A: ChatML conversations format (matches existing JSONL)
            if r.gold_think:
                chatml = {
                    "conversations": [
                        {"role": "system", "content": "You are Hermes, a concise autonomous assistant. Think briefly inside <think> tags before acting."},
                        {"role": "user", "content": r.prompt},
                        {"role": "assistant", "content": r.gold_think},
                    ]
                }
                line = json.dumps(chatml)
                f.write(line + "\n")
                master.write(line + "\n")
                count += 1

            # Format B: Structured trace format (matches methodology doc)
            if r.gold_trace:
                metadata = {
                    "task_id": r.task_id,
                    "domain": r.domain,
                    "difficulty": r.difficulty,
                    "category": r.category,
                    "scores": r.scores,
                    "iteration_timestamp": r.timestamp,
                }
                if r.raw_quality in ("partial", "fail") and r.partial_type:
                    metadata["partial_type"] = r.partial_type
                if r.raw_quality in ("partial", "fail") and r.minimal_hint:
                    metadata["minimal_hint"] = r.minimal_hint
                structured = {
                    "system": "You are Hermes, a concise autonomous assistant.",
                    "input": r.prompt,
                    "trace": r.gold_trace,
                    "output": r.gold_output or "",
                    "metadata": metadata,
                }
                line = json.dumps(structured)
                f.write(line + "\n")
                master.write(line + "\n")
                count += 1

    print(f"  [Stage 3] Wrote {count} training entries to {batch_path.name}")
    return batch_path


def compute_metrics(results: list[TaskResult], phase_state: dict | None = None) -> dict:
    """Compute iteration-level metrics."""
    total = len(results)
    if total == 0:
        return {}

    validated = sum(1 for r in results if r.validated)
    self_distilled = sum(1 for r in results if r.raw_quality == "self-distilled")
    failed = sum(1 for r in results if r.raw_quality == "distill_failed")
    raw_pass = sum(1 for r in results if r.raw_quality == "pass")
    partial_rows = [r for r in results if r.raw_quality in ("partial", "fail")]
    hint_susceptible_types = {"weak_specificity", "weak_verification"}
    from collections import Counter
    partials_by_domain = Counter(r.domain for r in partial_rows if r.raw_quality == "partial")
    failure_mode_counts = Counter(r.partial_type or "other" for r in partial_rows)
    output_length_analysis = {
        "short_fail": 0,
        "short_pass_or_partial": 0,
        "medium_fail": 0,
        "medium_pass_or_partial": 0,
        "long_fail": 0,
        "long_pass_or_partial": 0,
    }
    for r in results:
        bucket = "short" if r.raw_tokens < 200 else "medium" if r.raw_tokens < 500 else "long"
        suffix = "fail" if r.raw_quality == "fail" else "pass_or_partial"
        output_length_analysis[f"{bucket}_{suffix}"] += 1

    scores_all = [r.scores for r in results if r.scores]
    avg_scores = {}
    if scores_all:
        for key in scores_all[0]:
            vals = [s[key] for s in scores_all if key in s]
            avg_scores[key] = round(sum(vals) / len(vals), 2)

    # Domain coverage
    domains = set(r.domain for r in results)
    categories = {}
    raw_pass_domains = []
    partial_domains = []
    frontier_rows = []
    for r in results:
        categories[r.category] = categories.get(r.category, 0) + 1
        if r.raw_quality == "pass":
            raw_pass_domains.append(r.domain)
        if r.raw_quality == "partial":
            partial_domains.append(r.domain)
        if r.raw_quality != "pass":
            scores = r.scores or {}
            composite = sum(scores.values()) / len(scores) if scores else 0
            frontier_rows.append((composite, r))

    hint_conversion_proxy_count = sum(1 for r in partial_rows if (r.partial_type in hint_susceptible_types))
    memory_partial_rows = [r for r in partial_rows if r.domain == "memory_integration"]
    memory_hint_conversion_proxy_count = sum(1 for r in memory_partial_rows if (r.partial_type in hint_susceptible_types))
    micro_checklist_count = sum(1 for r in results if r.domain == "memory_integration" and isinstance(r.gold_trace, dict) and bool(r.gold_trace.get("checks")))
    checklist_format_applied = all(
        (not r.gold_think) or r.gold_think.lstrip().startswith("[REQUIREMENTS]")
        for r in results if r.raw_quality in ("partial", "fail")
    )
    false_confidence_count = sum(1 for r in results if getattr(r, "false_confidence", False))
    false_confidence_rate = round(false_confidence_count / total, 3) if total else 0
    false_confidence_by_category = {"deliberate_error_detection": 0, "completion_check": 0, "conflicting_info": 0, "progressive_refinement": 0, "false_confidence_calibration": 0}
    for r in results:
        if getattr(r, "false_confidence", False):
            false_confidence_by_category[r.category] = false_confidence_by_category.get(r.category, 0) + 1

    frontier_rows.sort(key=lambda x: (-x[0], x[1].task_id))
    frontier_tasks = []
    frontier_analysis = []
    for _, r in frontier_rows[:5]:
        pt = r.partial_type or "n/a"
        if pt == "weak_specificity":
            got_right = "the answer had the right general direction"
            wrong = "it stayed too generic and missed concrete values"
            nature = "surface-level"
            cot = "likely yes: force exact values or commands"
        elif pt == "incomplete_coverage":
            got_right = "the main approach was present"
            wrong = "it missed one or more required branches or edge cases"
            nature = "structural"
            cot = "yes: use a checklist / coverage sweep"
        elif pt == "wrong_ranking":
            got_right = "it identified the relevant options"
            wrong = "it ordered priorities incorrectly"
            nature = "structural"
            cot = "yes: force explicit ranking criteria"
        elif pt == "weak_verification":
            got_right = "the plan was directionally correct"
            wrong = "it lacked a concrete validation step"
            nature = "surface-level"
            cot = "possibly: require a final check"
        else:
            got_right = "some useful structure was present"
            wrong = "the failure mode was not classified"
            nature = "unknown"
            cot = "unknown"
        frontier_tasks.append(r.task_id)
        frontier_analysis.append({
            "task_id": r.task_id,
            "domain": r.domain,
            "partial_type": pt,
            "what_got_right": got_right,
            "what_went_wrong": wrong,
            "failure_nature": nature,
            "cot_hint_potential": cot,
            "raw_quality": r.raw_quality,
            "scores": r.scores,
        })
    # Category mix vs targets
    mix_actual = {k: round(v / total, 2) for k, v in categories.items()}

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_tasks": total,
        "validated": validated,
        "validation_rate": round(validated / total, 3) if total else 0,
        "self_distilled": self_distilled,
        "distill_failures": failed,
        "raw_passes": raw_pass,
        "raw_pass_rate": round(raw_pass / total, 3) if total else 0,
        "raw_pass_domains": sorted(set(raw_pass_domains)),
        "partial_count": len(partial_domains),
        "partial_domains": sorted(set(partial_domains)),
        "partials_by_domain": dict(partials_by_domain),
        "hint_passes": hint_conversion_proxy_count,
        "full_corrections": max(len(partial_rows) - hint_conversion_proxy_count, 0),
        "hint_conversion_rate": round(hint_conversion_proxy_count / len(partial_rows), 3) if partial_rows else 0,
        "hint_conversion_proxy_count": hint_conversion_proxy_count,
        "hint_conversion_proxy_rate": round(hint_conversion_proxy_count / len(partial_rows), 3) if partial_rows else 0,
        "memory_integration_hint_conversion_proxy_count": memory_hint_conversion_proxy_count,
        "memory_integration_hint_conversion_proxy_rate": round(memory_hint_conversion_proxy_count / len(memory_partial_rows), 3) if memory_partial_rows else 0,
        "micro_checklist_count": micro_checklist_count,
        "micro_checklist_rate": round(micro_checklist_count / total, 3) if total else 0,
        "checklist_format_applied": checklist_format_applied,
        "false_confidence_count": false_confidence_count,
        "false_confidence_rate": false_confidence_rate,
        "false_confidence_by_category": false_confidence_by_category,
        "frontier_tasks": frontier_tasks,
        "frontier_analysis": frontier_analysis,
        "frontier_failure_modes": dict(failure_mode_counts),
        "output_length_analysis": output_length_analysis,
        "phase": (phase_state or {}).get("phase", 1),
        "iterations_in_phase": (phase_state or {}).get("iterations_in_phase", 0),
        "avg_scores": avg_scores,
        "domains_covered": sorted(domains),
        "category_mix_actual": mix_actual,
        "category_mix_target": MIX_TARGETS,
        "avg_raw_tokens": round(sum(r.raw_tokens for r in results) / total, 1),
        "avg_raw_time_s": round(sum(r.raw_time_s for r in results) / total, 2),
    }

    return metrics


def save_metrics(metrics: dict, iteration_num: int):
    """Persist iteration metrics."""
    # Per-iteration file
    iter_path = METRICS_DIR / f"iteration_{iteration_num:03d}.json"
    iter_path.write_text(json.dumps(metrics, indent=2))

    # Append to iteration log
    log = []
    if ITERATION_LOG.exists():
        log = json.loads(ITERATION_LOG.read_text())
    metrics["iteration"] = iteration_num
    log.append(metrics)
    ITERATION_LOG.write_text(json.dumps(log, indent=2))

    print(f"  [Metrics] Saved to {iter_path.name}")


def get_next_iteration_num() -> int:
    """Get the next iteration number from the log."""
    if ITERATION_LOG.exists():
        log = json.loads(ITERATION_LOG.read_text())
        return len(log) + 1
    return 1


def update_task_queue(results: list[TaskResult]):
    """Mark completed tasks and feed failures back into queue."""
    queue = json.loads(TASK_QUEUE_PATH.read_text())

    result_map = {r.task_id: r for r in results}
    weak_domains = set()

    for task in queue["tasks"]:
        if task["id"] in result_map:
            r = result_map[task["id"]]
            if r.validated or r.raw_quality == "self-distilled":
                task["status"] = "completed"
            else:
                retries = task.get("retry_count", 0) + 1
                task["retry_count"] = retries
                if retries >= 3:
                    task["status"] = "abandoned"
                else:
                    task["status"] = "retry"
                weak_domains.add(r.domain)

    TASK_QUEUE_PATH.write_text(json.dumps(queue, indent=2))

    if weak_domains:
        print(f"  [Queue] Weak domains flagged for retry: {', '.join(weak_domains)}")


def print_report():
    """Print a summary of all iterations."""
    if not ITERATION_LOG.exists():
        print("No iterations logged yet.")
        return

    log = json.loads(ITERATION_LOG.read_text())
    print(f"\n{'='*60}")
    print(f"Karpathy Loop — {len(log)} iteration(s)")
    print(f"{'='*60}")

    for entry in log:
        i = entry.get("iteration", "?")
        print(f"\n  Iteration {i} ({entry['timestamp'][:10]})")
        print(f"    Tasks: {entry['total_tasks']}, Validated: {entry['validated']} ({entry['validation_rate']*100:.1f}%)")
        print(f"    Avg scores: {entry.get('avg_scores', {})}")
        print(f"    Avg raw tokens: {entry['avg_raw_tokens']}, Avg time: {entry['avg_raw_time_s']}s")
        print(f"    Domains: {', '.join(entry['domains_covered'])}")
        print(f"    Distill failures: {entry['distill_failures']}")

    # Overall training data count
    if TRAINING_JSONL.exists():
        with open(TRAINING_JSONL) as f:
            total_lines = sum(1 for _ in f)
        print(f"\n  Total training entries: {total_lines}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_iteration(task_count: int = 5, use_self_distill: bool = True):
    """Run one iteration of the Karpathy loop."""
    iteration_num = get_next_iteration_num()
    print(f"\n{'='*60}")
    print(f"Karpathy Loop — Iteration {iteration_num}")
    print(f"{'='*60}\n")

    # Load task queue
    queue = json.loads(TASK_QUEUE_PATH.read_text())
    pending = [t for t in queue["tasks"] if t["status"] in ("pending", "retry")]

    if not pending:
        print("No pending tasks in queue.")
        return

    # Select tasks for this iteration
    tasks = pending[:task_count]
    print(f"Selected {len(tasks)} tasks: {[t['id'] for t in tasks]}\n")

    # Initialize Hermes client
    client = LLMClient(HERMES_API, HERMES_MODEL)

    results = []
    for task in tasks:
        # Stage 1: Generate raw trace
        result = stage1_generate(client, task)

        # Stage 2: Distill
        if use_self_distill:
            result = stage2_distill_inline(result)
        else:
            result = stage2_distill_local(result)
            print(f"  [Stage 2] Distill request saved for {result.task_id} (manual review needed)")

        results.append(result)
        print()

    # Stage 3: Compile
    print(f"{'='*60}")
    print("Stage 3: Compile & Metrics")
    print(f"{'='*60}")

    batch_path = compile_training_jsonl(results)
    metrics = compute_metrics(results)
    save_metrics(metrics, iteration_num)
    update_task_queue(results)

    # Print summary
    print(f"\n  Validation rate: {metrics.get('validation_rate', 0)*100:.1f}%")
    print(f"  Avg scores: {metrics.get('avg_scores', {})}")
    print(f"  Raw passes: {metrics.get('raw_passes', 0)}/{len(results)}")
    print(f"  Training batch: {batch_path}")
    print()

    return results, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Karpathy Auto-Research Loop")
    parser.add_argument("--pilot", action="store_true", help="Run 5-task pilot")
    parser.add_argument("--iteration", type=int, help="Run N tasks")
    parser.add_argument("--full", action="store_true", help="Run all pending tasks")
    parser.add_argument("--report", action="store_true", help="Show metrics summary")
    parser.add_argument("--no-self-distill", action="store_true",
                        help="Skip self-distillation (save requests for manual review)")

    args = parser.parse_args()

    if args.report:
        print_report()
        return

    use_self_distill = not args.no_self_distill

    if args.pilot:
        run_iteration(task_count=5, use_self_distill=use_self_distill)
    elif args.iteration:
        run_iteration(task_count=args.iteration, use_self_distill=use_self_distill)
    elif args.full:
        queue = json.loads(TASK_QUEUE_PATH.read_text())
        pending = sum(1 for t in queue["tasks"] if t["status"] in ("pending", "retry"))
        run_iteration(task_count=pending, use_self_distill=use_self_distill)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
