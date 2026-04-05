#!/usr/bin/env python3
"""
Two-model Karpathy iteration: Bonsai generates, Codex distills.

- Student (Bonsai 8B @ localhost:1234): raw reasoning traces
- Teacher (GPT-5.4-mini @ Codex): gold distillation, scoring, task gen

Designed for high throughput: 50 tasks per call, 15-min cron cycle.
Codex limits: 1000 req/min, 1M tokens/min, 400K context.

After each batch, writes a Hermes memory summary so Bonsai inherits
the learning structure when it takes over Hermes.
"""
import json
import sys
import time
import os
import argparse
import concurrent.futures
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI

from karpathy_loop import (
    LLMClient, TaskResult, TOKEN_TARGETS, MIX_TARGETS,
    BASE_DIR, TASK_QUEUE_PATH, OUTPUT_DIR, RAW_DIR, GOLD_DIR,
    BATCH_DIR, METRICS_DIR, TRAINING_JSONL, ITERATION_LOG,
    HERMES_SYSTEM, DISTILL_SYSTEM,
    stage1_generate, stage2_apply_gold, compile_training_jsonl,
    compute_metrics, save_metrics, update_task_queue, get_next_iteration_num,
    _extract_json,
)

BONSAI_API = "http://localhost:1234/v1"
BONSAI_MODEL = "bonsai-8b"
CODEX_BASE = "https://chatgpt.com/backend-api/codex"
CODEX_MODEL = "gpt-5.4-mini"
HERMES_AUTH = Path.home() / ".hermes" / "auth.json"
HERMES_MEMORIES = Path.home() / ".hermes" / "memories"
PHASE_STATE_PATH = BASE_DIR / "bonsai_graduation_state.json"

PHASE_CONFIG = {
    1: {
        "domains": ["memory_integration", "refusal_redirect", "self_correction"],
        "domain_mix": {"memory_integration": 0.6, "refusal_redirect": 0.2, "self_correction": 0.2},
        "difficulties": ["easy", "medium"],
        "gate_raw_passes": 5,
        "name": "Foundation Lock-In",
    },
    2: {
        "domains": ["memory_integration", "refusal_redirect", "self_correction", "agent_routing", "devops"],
        "difficulties": ["easy", "medium", "hard"],
        "gate_raw_passes": 10,
        "name": "Domain Expansion",
    },
    3: {
        "domains": ["logic_puzzle", "code_debugging", "math", "architecture", "memory_integration", "refusal_redirect", "self_correction", "agent_routing", "devops", "research_synthesis"],
        "difficulties": ["easy", "medium", "hard"],
        "gate_raw_passes": 15,
        "name": "Hard Domain Assault",
    },
}


# ---------------------------------------------------------------------------
# Codex client
# ---------------------------------------------------------------------------
def get_codex_token() -> str:
    auth = json.loads(HERMES_AUTH.read_text())
    pool = auth.get("credential_pool", {}).get("openai-codex", [])
    if pool:
        return pool[0]["access_token"]
    return auth["providers"]["openai-codex"]["tokens"]["access_token"]


def codex_chat(system: str, user: str, timeout: int = 60) -> str | None:
    """Call Codex via Responses API."""
    token = get_codex_token()
    client = OpenAI(api_key=token, base_url=CODEX_BASE)
    try:
        with client.responses.stream(
            model=CODEX_MODEL,
            instructions=system,
            input=[{"role": "user", "content": user}],
            store=False,
        ) as stream:
            for _ in stream:
                pass
            final = stream.get_final_response()
        parts = []
        for item in getattr(final, "output", []):
            if getattr(item, "type", None) == "message":
                for part in getattr(item, "content", []):
                    if getattr(part, "type", None) in ("output_text", "text"):
                        parts.append(getattr(part, "text", ""))
        return "".join(parts).strip() or None
    except Exception as e:
        print(f"    [Codex error] {e}")
        return None


# ---------------------------------------------------------------------------
# Stage 2: Codex distills
# ---------------------------------------------------------------------------
def codex_distill(result: TaskResult) -> TaskResult:
    tmin, tmax = TOKEN_TARGETS[result.difficulty]
    # Domain-specific cognitive modes — different tasks need different trace structures
    mode_instruction = ""
    trace_schema_hint = "2-4 steps"

    if result.domain in ("devops",):
        mode_instruction = """
COGNITIVE MODE: DIAGNOSTIC. Use the DevOps diagnostic trace schema.
Group related symptoms into diagnostic chunks. Wrap specific commands and file paths
in [CMD: ...] anchors so they survive as single tokens (e.g., [CMD: kubectl describe pod],
[CMD: cat /sys/fs/cgroup/memory/memory.usage_in_bytes]).
For partial failures, annotate: weak_specificity | incomplete_coverage | wrong_ranking | weak_verification"""
        trace_schema = """{{"trace": {{
    "symptom_chunks": ["bound diagnostic observations"],
    "likely_cause_family": "one sentence",
    "ranked_hypotheses": ["most likely first", "second", "third"],
    "exact_next_check": "[CMD: specific command or action]",
    "confirm_result": "what output confirms this hypothesis",
    "rule_out_result": "what output rules it out",
    "decision": "final diagnosis"
  }}"""

    elif result.domain in ("code_debugging",):
        mode_instruction = """
COGNITIVE MODE: DIAGNOSTIC + COMPLETION. Find ALL bugs, not just the first one.
After identifying the primary bug, always do a second pass asking "what else could break?"
Wrap code references in [CODE: ...] anchors.
For partial failures, annotate: weak_specificity | incomplete_coverage | wrong_ranking | weak_verification"""
        trace_schema = """{{"trace": {{
    "observed_bug": "primary issue found",
    "primary_cause": "why it happens",
    "secondary_risk": "what else could break (edge cases, related bugs)",
    "edge_cases": ["input that would trigger failure"],
    "minimal_fix": "[CODE: the specific fix]",
    "verification_test": "how to confirm the fix works"
  }}"""

    elif result.domain in ("logic_puzzle",):
        mode_instruction = """
COGNITIVE MODE: DEDUCTIVE. Use explicit variable/state tracking throughout.
Name each unknown. Show elimination steps. Maintain a state table.
Logic puzzles fail when the model loses track of which possibilities have been ruled out.
Check for contradictions before finalizing."""
        trace_schema = """{{"trace": {{
    "knowns": ["established facts"],
    "unknowns": ["what we need to determine"],
    "constraints": ["rules that must hold"],
    "eliminations": ["X is ruled out because Y"],
    "remaining_candidates": ["what's still possible"],
    "final_solution": "the consistent assignment"
  }}"""

    elif result.domain in ("agent_routing",):
        mode_instruction = """
COGNITIVE MODE: ROUTING. This is a dispatch task, NOT a diagnostic task. Do NOT over-analyze.
Routing wants fast classification, clean thresholds, and explicit handoff."""
        trace_schema = """{{"trace": {{
    "task_type": "classification of what's being asked",
    "confidence": "high|medium|low",
    "best_route": "selected handler/tool/agent",
    "why": "one sentence justification",
    "fallback_route": "if primary fails",
    "escalation_trigger": "condition that requires human/stronger model"
  }}"""

    elif result.domain in ("architecture",):
        mode_instruction = """
COGNITIVE MODE: CONSTRAINT. List ALL constraints first, then identify which ones
tension against each other, then resolve trade-offs explicitly."""
        trace_schema = """{{"trace": {{
    "goal": "what we're designing",
    "constraints": ["all requirements and limits"],
    "tensions": ["constraint A vs constraint B"],
    "resolution": "how the trade-off is resolved",
    "decision": "chosen architecture and why"
  }}"""

    elif result.domain in ("self_correction", "refusal_redirect"):
        mode_instruction = """
COGNITIVE MODE: BEHAVIORAL. Minimal trace. Pattern-match and execute."""
        trace_schema = """{{"trace": {{
    "goal": "one sentence",
    "constraints": ["2-3 items"],
    "plan": ["2-3 brief steps"],
    "decision": "one sentence"
  }}"""

    else:
        mode_instruction = ""
        trace_schema = """{{"trace": {{
    "goal": "one sentence",
    "constraints": ["2-3 items"],
    "plan": ["2-4 steps"],
    "checks": ["1-2 questions"],
    "decision": "one sentence"
  }}"""

    user_prompt = f"""Distill this raw reasoning trace into a gold-standard compact training example.

TASK (difficulty: {result.difficulty}, budget: {tmin}-{tmax} tokens, domain: {result.domain}):
{result.prompt}

RAW ATTEMPT FROM BONSAI-8B:
{result.raw_trace}
{mode_instruction}
Produce EXACTLY this JSON (no markdown fences). Use this domain-specific trace schema:
{trace_schema},
  "output": "concise answer",
  "think_tagged": "<think>\\n[domain-appropriate compact reasoning]\\n</think>\\n\\nAnswer",
  "score": {{
    "correctness": 0-10,
    "direction": 0-10,
    "specificity": 0-10,
    "completion": 0-10
  }},
  "raw_attempt_quality": "pass|partial|fail",
  "partial_type": "weak_specificity|incomplete_coverage|wrong_ranking|weak_verification|n/a",
  "minimal_hint": "smallest nudge that would redirect the raw attempt",
  "notes": "brief observation"
}}
Output ONLY the JSON."""

    # Expected domain-specific trace keys (for measurement)
    DOMAIN_SCHEMA_KEYS = {
        "devops": ("symptom_chunks", "ranked_hypotheses", "exact_next_check"),
        "code_debugging": ("observed_bug", "primary_cause", "minimal_fix"),
        "logic_puzzle": ("knowns", "unknowns", "eliminations"),
        "agent_routing": ("task_type", "best_route", "fallback_route"),
        "architecture": ("tensions", "resolution"),
    }

    print(f"  [S2] Codex {result.task_id}...", end=" ", flush=True)
    content = codex_chat(DISTILL_SYSTEM, user_prompt)
    if not content:
        print("no response")
        return result
    try:
        gold = _extract_json(content)

        # Normalize old 4-field scores to 3D if Codex reverted
        score = gold.get("score", {})
        if score and "direction" not in score and "correctness" in score:
            gold["score"] = {
                "correctness": score.get("correctness", 5),
                "direction": score.get("correctness", 5),
                "specificity": score.get("actionability", 5),
                "completion": score.get("schema_consistency", 5),
            }

        # Infer partial_type from 3D scores if missing for partial/fail
        rq = gold.get("raw_attempt_quality", "")
        pt = gold.get("partial_type", "n/a")
        if rq in ("partial", "fail") and pt in ("n/a", "", None):
            sc = gold.get("score", {})
            spec = sc.get("specificity", 10)
            comp = sc.get("completion", 10)
            dirn = sc.get("direction", 10)
            if spec <= 6:
                gold["partial_type"] = "weak_specificity"
            elif comp <= 6:
                gold["partial_type"] = "incomplete_coverage"
            elif dirn <= 6:
                gold["partial_type"] = "wrong_ranking"
            else:
                gold["partial_type"] = "weak_verification"
        if rq in ("partial", "fail") and gold.get("minimal_hint", "n/a") in ("n/a", "", None):
            hint_map = {
                "weak_specificity": "Name exact values, commands, or thresholds instead of speaking generally.",
                "incomplete_coverage": "Add the missing branch, edge case, or step before answering.",
                "wrong_ranking": "Re-rank the options explicitly using the task's priority constraint.",
                "weak_verification": "End with one concrete verification step or check command.",
            }
            gold["minimal_hint"] = hint_map.get(gold.get("partial_type", ""), "Use a more concrete, task-shaped answer.")

        # Check domain schema match (measurement only)
        trace = gold.get("trace", {})
        expected_keys = DOMAIN_SCHEMA_KEYS.get(result.domain, ())
        schema_match = any(k in trace for k in expected_keys) if expected_keys else True

        result = stage2_apply_gold(result, gold)
        s = result.scores or {}
        tag = "V" if result.validated else "x"
        schema_tag = "S" if schema_match else "g"  # S=domain-specific, g=generic
        print(f"{tag}{schema_tag} d:{s.get('direction','?')} s:{s.get('specificity','?')} c:{s.get('completion','?')} raw:{result.raw_quality}")
    except Exception as e:
        print(f"parse fail: {e}")
        (GOLD_DIR / f"{result.task_id}_codex_fail.txt").write_text(content)
    return result


# ---------------------------------------------------------------------------
# Batch task generation via Codex (generates many at once)
# ---------------------------------------------------------------------------
def codex_generate_tasks(
    count: int,
    focus: list[str] | None = None,
    weakness_data: list[dict] | None = None,
    allowed_domains: list[str] | None = None,
    allowed_difficulties: list[str] | None = None,
    phase: int = 1,
) -> list[dict]:
    domains = ", ".join(allowed_domains or ["logic_puzzle", "math", "code_debugging", "architecture", "devops", "agent_routing", "self_correction", "research_synthesis", "refusal_redirect", "memory_integration"])
    categories = "deliberate_error_detection, completion_check, conflicting_info, progressive_refinement, false_confidence_calibration" if allowed_domains == ["self_correction"] else ", ".join(MIX_TARGETS.keys())

    # Build targeted focus hint from enriched weakness data
    if weakness_data:
        focus_lines = ["\nFocus areas with specific training gaps:"]
        for w in weakness_data[:4]:
            line = f"- {w['domain']}: {w['failure_count']} recent failures."
            if w["top_partial_types"]:
                types_str = ", ".join(f"{t} ({c})" for t, c in w["top_partial_types"])
                line += f" Top issues: {types_str}."
            if w["avg_specificity"] is not None:
                line += f" Avg specificity: {w['avg_specificity']}/10."
            # Add targeted generation guidance per failure type
            top_type = w["top_partial_types"][0][0] if w["top_partial_types"] else None
            if top_type == "weak_specificity":
                line += " Generate tasks requiring EXACT commands, specific values, and precise syntax."
            elif top_type == "incomplete_coverage":
                line += " Generate tasks with multiple bugs, edge cases, or failure modes that require thorough coverage."
            elif top_type == "wrong_ranking":
                line += " Generate tasks where constraint ordering and priority ranking are critical."
            elif top_type == "weak_verification":
                line += " Generate tasks that require concrete verification steps and confirmation commands."
            focus_lines.append(line)
        focus_hint = "\n".join(focus_lines)
    elif focus:
        focus_hint = f"\nFocus especially on: {', '.join(focus)}"
    else:
        focus_hint = ""

    # Generate in batches of 20 to stay within response size limits
    all_tasks = []
    remaining = count
    batch_num = 0

    difficulty_hint = ", ".join(allowed_difficulties or ["easy", "medium", "hard"])
    mix_hint = ""
    category_hint = ""
    if phase == 1:
        mix_hint = "\nTarget domain mix: 60% memory_integration, 20% refusal_redirect, 20% self_correction."
    if allowed_domains == ["self_correction"]:
        category_hint = "\nTarget categories (roughly even): deliberate_error_detection, completion_check, conflicting_info, progressive_refinement, false_confidence_calibration."

    def _parse_task_array_relaxed(content: str) -> list[dict]:
        """Best-effort parse for Codex task arrays.

        Try strict JSON first. If Codex returns a malformed array, salvage any
        balanced top-level objects so a single bad separator does not discard the
        whole batch.
        """
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end <= start:
            return []
        body = cleaned[start:end + 1]
        try:
            parsed = json.loads(body)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            pass

        items: list[dict] = []
        buf = []
        depth = 0
        in_string = False
        escape = False
        started = False
        for ch in body:
            if not started:
                if ch == "{":
                    started = True
                    depth = 1
                    buf = [ch]
                    in_string = False
                    escape = False
                continue

            buf.append(ch)
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        raw_item = "".join(buf)
                        try:
                            item = json.loads(raw_item)
                            if isinstance(item, dict):
                                items.append(item)
                        except Exception:
                            pass
                        buf = []
                        started = False
        return items

    while remaining > 0:
        batch_size = min(remaining, 20)
        batch_num += 1
        prompt = f"""Generate exactly {batch_size} training tasks as a JSON array.
Phase: {phase}
Domains: {domains}
Categories: {categories}
Difficulties allowed: {difficulty_hint}{mix_hint}{category_hint}{focus_hint}

Each: {{"id":"auto-NNN","domain":"...","difficulty":"...","category":"...","prompt":"specific self-contained question (1-3 sentences)","expected_skills":["a","b"],"difficulty_estimate":"low|medium|hard","status":"pending"}}

Be creative and diverse. No duplicate topics. Output ONLY a JSON array."""

        print(f"  [TaskGen] Batch {batch_num} ({batch_size} tasks)...", end=" ", flush=True)
        content = codex_chat("Generate training datasets. Output ONLY valid JSON arrays.", prompt)
        if not content:
            print("no response")
            break

        try:
            tasks = _parse_task_array_relaxed(content)
            for t in tasks:
                t.setdefault("status", "pending")
                t.setdefault("expected_skills", [])
                t.setdefault("difficulty_estimate", t.get("difficulty", "medium"))
            all_tasks.extend(tasks)
            remaining -= len(tasks)
            print(f"{len(tasks)} tasks")
        except Exception as e:
            print(f"parse fail: {e}")
            break

    return all_tasks


# ---------------------------------------------------------------------------
# Curriculum state + frontier analysis
# ---------------------------------------------------------------------------
def load_phase_state() -> dict:
    if PHASE_STATE_PATH.exists():
        try:
            return json.loads(PHASE_STATE_PATH.read_text())
        except Exception:
            pass
    return {
        "phase": 1,
        "iterations_in_phase": 0,
        "phase_started_at": datetime.now().isoformat(),
        "history": [],
    }


def save_phase_state(state: dict) -> None:
    PHASE_STATE_PATH.write_text(json.dumps(state, indent=2))


def phase_config(phase: int) -> dict:
    return PHASE_CONFIG.get(phase, PHASE_CONFIG[1])


def apply_curriculum_to_queue(queue: dict, active_domains: list[str], allowed_difficulties: list[str], phase: int) -> int:
    parked = 0
    for task in queue.get("tasks", []):
        status = task.get("status", "pending")
        domain = task.get("domain")
        difficulty = task.get("difficulty")
        in_scope = domain in active_domains and difficulty in allowed_difficulties
        if in_scope:
            if status.startswith("parked_"):
                task["status"] = "pending"
        elif status in ("pending", "retry"):
            task["status"] = f"parked_phase_{phase}"
            parked += 1
    return parked


def select_curriculum_tasks(tasks: list[dict], active_domains: list[str], allowed_difficulties: list[str], n: int) -> list[dict]:
    selected = [
        t for t in tasks
        if t.get("status") in ("pending", "retry")
        and t.get("domain") in active_domains
        and t.get("difficulty") in allowed_difficulties
    ]
    return selected[:n]


def build_frontier_analysis(results: list[TaskResult], limit: int = 5) -> dict:
    frontier = []
    for r in results:
        if r.raw_quality == "pass":
            continue
        scores = r.scores or {}
        composite = sum(scores.values()) / len(scores) if scores else 0
        frontier.append((composite, r))
    frontier.sort(key=lambda x: (-x[0], x[1].task_id))
    top = []
    for _, r in frontier[:limit]:
        pt = r.partial_type or "n/a"
        if pt == "weak_specificity":
            wrong = "key details were too generic or underspecified"
            got_right = "the high-level direction was present"
            structural = "surface-level"
            cot = "likely yes, if it forced concrete values"
        elif pt == "incomplete_coverage":
            wrong = "it omitted one or more required branches, steps, or edge cases"
            got_right = "the main approach was usually right"
            structural = "structural"
            cot = "yes, if it prompted a checklist or coverage sweep"
        elif pt == "wrong_ranking":
            wrong = "it ordered priorities or options incorrectly"
            got_right = "it recognized the relevant items"
            structural = "structural"
            cot = "yes, if it forced explicit ranking criteria"
        elif pt == "weak_verification":
            wrong = "it lacked a concrete verification step or check"
            got_right = "the answer direction was usually fine"
            structural = "surface-level"
            cot = "possibly, if it required a final validation step"
        else:
            wrong = "the failure mode was not classified"
            got_right = "some useful structure was still present"
            structural = "unknown"
            cot = "unknown"
        top.append({
            "task_id": r.task_id,
            "domain": r.domain,
            "partial_type": pt,
            "what_got_right": got_right,
            "what_went_wrong": wrong,
            "failure_nature": structural,
            "cot_hint_potential": cot,
            "raw_quality": r.raw_quality,
            "scores": r.scores,
        })
    return {"frontier_tasks": [x["task_id"] for x in top], "frontier_analysis": top}


# ---------------------------------------------------------------------------
# Weakness analysis
# ---------------------------------------------------------------------------
def analyze_weaknesses() -> list[dict]:
    """Analyze recent gold traces for failure signatures by domain.

    Returns list of dicts: {domain, failure_count, top_partial_types, avg_specificity}
    sorted by failure_count descending.
    """
    if not ITERATION_LOG.exists():
        return []

    # Scan recent gold traces (last 500) for failure patterns
    from collections import Counter, defaultdict
    gold_files = sorted(GOLD_DIR.glob("*_gold.json"), key=lambda p: p.stat().st_mtime)
    recent_golds = gold_files[-500:]

    domain_failures = defaultdict(lambda: {"count": 0, "partial_types": Counter(), "specificities": []})

    for p in recent_golds:
        try:
            d = json.loads(p.read_text())
            if d.get("raw_quality") in ("fail", "partial"):
                dom = d.get("domain", "unknown")
                domain_failures[dom]["count"] += 1
                pt = d.get("partial_type", "n/a")
                if pt and pt != "n/a":
                    domain_failures[dom]["partial_types"][pt] += 1
            # Collect specificity scores regardless of pass/fail
            scores = d.get("scores", {})
            if "specificity" in scores:
                dom = d.get("domain", "unknown")
                domain_failures[dom]["specificities"].append(scores["specificity"])
        except:
            pass

    # Also check domain coverage gaps from iteration log
    log = json.loads(ITERATION_LOG.read_text())
    recent = log[-10:]
    domain_counts = {}
    for e in recent:
        for d in e.get("domains_covered", []):
            domain_counts[d] = domain_counts.get(d, 0) + 1
    all_domains = {"logic_puzzle", "math", "code_debugging", "architecture", "devops",
                   "agent_routing", "self_correction", "research_synthesis",
                   "refusal_redirect", "memory_integration"}
    coverage_gaps = {d for d in all_domains if domain_counts.get(d, 0) < 3}

    # Build enriched weakness list
    results = []
    seen = set()
    for dom, info in sorted(domain_failures.items(), key=lambda x: -x[1]["count"]):
        if info["count"] > 0:
            specs = info["specificities"]
            top_types = info["partial_types"].most_common(3)
            results.append({
                "domain": dom,
                "failure_count": info["count"],
                "top_partial_types": top_types,
                "avg_specificity": round(sum(specs) / len(specs), 1) if specs else None,
            })
            seen.add(dom)

    # Add coverage gaps that didn't appear in failures
    for dom in coverage_gaps:
        if dom not in seen:
            results.append({
                "domain": dom,
                "failure_count": 0,
                "top_partial_types": [],
                "avg_specificity": None,
            })

    return results[:6]  # Top 6 weakness areas


# ---------------------------------------------------------------------------
# Hermes memory integration
# ---------------------------------------------------------------------------
def update_hermes_memory(metrics: dict, iteration_num: int, raw_quality_summary: dict):
    """Write a memory file into Hermes's memory system so Bonsai inherits the learning."""
    HERMES_MEMORIES.mkdir(parents=True, exist_ok=True)

    # Running summary — overwrite each time with latest state
    total_entries = 0
    if TRAINING_JSONL.exists():
        with open(TRAINING_JSONL) as f:
            total_entries = sum(1 for _ in f)

    # Compute pass rate from recent iterations
    if ITERATION_LOG.exists():
        log = json.loads(ITERATION_LOG.read_text())
        recent = log[-10:]
    else:
        recent = []

    avg_vr = sum(e.get("validation_rate", 0) for e in recent) / max(len(recent), 1)
    avg_correct = sum(e.get("avg_scores", {}).get("correctness", 0) for e in recent) / max(len(recent), 1)

    memory_content = f"""# Karpathy Auto-Research Loop Status

Last updated: {datetime.now().isoformat()[:19]}
Iteration: {iteration_num}
Total training entries: {total_entries}

## Current Performance
- Validation rate (last 10): {avg_vr*100:.1f}%
- Avg correctness (last 10): {avg_correct:.1f}/10
- Active phase: {metrics.get('phase', 1)}
- Iterations in phase: {metrics.get('iterations_in_phase', 0)}
- Raw attempt quality this batch: {json.dumps(raw_quality_summary)}
- Hint conversion proxy rate: {metrics.get('hint_conversion_proxy_rate', 0):.3f}
- Memory-integration hint proxy rate: {metrics.get('memory_integration_hint_conversion_proxy_rate', 0):.3f}
- Micro-checklist coverage: {metrics.get('micro_checklist_rate', 0):.3f}

## Weak Areas (need more training data)
{json.dumps(analyze_weaknesses())}

## Architecture
- Student: Bonsai 1-bit 8B (localhost:1234) — generates raw reasoning
- Teacher: GPT-5.4-mini (Codex) — distills gold traces, scores
- Trace schema: goal/constraints/plan/checks/decision/output
- Token budgets: easy 40-120, medium 120-250, hard 250-500
- Training format: ChatML + structured JSONL (dual)

## For Future Bonsai (when you take over Hermes)
- You are being trained on these traces. The goal/constraints/plan/checks/decision
  format is YOUR reasoning format. Use it.
- Keep traces compact. You lose coherence past ~500 tokens.
- When you don't know something, say so. "fail" traces are training signal too.
- The task queue at ~/Desktop/Hermes Agent/Training Research/task_queue.json
  has your curriculum. Generate new tasks in weak domains.
- Run iterations with: hermes_iteration.py --count N
"""

    memory_path = HERMES_MEMORIES / "karpathy_loop_status.md"
    memory_path.write_text(memory_content)

    # Also write a skill-learning memory — what patterns Bonsai struggles with
    if raw_quality_summary.get("fail", 0) > 0 or raw_quality_summary.get("partial", 0) > 0:
        learning_path = HERMES_MEMORIES / "karpathy_learning_notes.md"
        existing = learning_path.read_text() if learning_path.exists() else ""

        new_note = f"\n## Iteration {iteration_num} ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
        new_note += f"Quality: {json.dumps(raw_quality_summary)}\n"
        weakness_data = analyze_weaknesses()
        new_note += f"Weak domains: {json.dumps([w['domain'] for w in weakness_data])}\n"
        for w in weakness_data[:3]:
            if w["top_partial_types"]:
                types_str = ", ".join(f"{t}({c})" for t, c in w["top_partial_types"])
                new_note += f"  {w['domain']}: {w['failure_count']} failures, types: {types_str}"
                if w["avg_specificity"] is not None:
                    new_note += f", avg_spec: {w['avg_specificity']}"
                new_note += "\n"

        # Keep only last 20 notes
        lines = existing.split("\n## ")
        if len(lines) > 20:
            lines = lines[-20:]
            existing = "## ".join(lines)

        learning_path.write_text(existing + new_note)


# ---------------------------------------------------------------------------
# Main — high-throughput iteration
# ---------------------------------------------------------------------------
def run_iteration(n: int = 50):
    iteration_num = get_next_iteration_num()
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"Karpathy Loop — Iteration {iteration_num} (Bonsai + Codex) — {n} tasks")
    print(f"{'='*60}\n")

    bonsai = LLMClient(BONSAI_API, BONSAI_MODEL)

    # Test Codex
    codex_ok = False
    try:
        test = codex_chat("Reply OK.", "Test.")
        codex_ok = test is not None
        print(f"  Codex: {'online' if codex_ok else 'OFFLINE'} ({CODEX_MODEL})")
    except Exception as e:
        print(f"  Codex: offline ({e})")

    # Load queue and enforce the active curriculum phase
    phase_state = load_phase_state()
    phase = int(phase_state.get("phase", 1))
    cfg = phase_config(phase)
    active_domains = cfg["domains"]
    allowed_difficulties = cfg["difficulties"]
    print(f"  Phase {phase} — {cfg['name']}")
    print(f"  Active domains: {', '.join(active_domains)}")
    print(f"  Allowed difficulties: {', '.join(allowed_difficulties)}")
    print(f"  Iterations in phase: {phase_state.get('iterations_in_phase', 0)}")

    queue = json.loads(TASK_QUEUE_PATH.read_text())
    parked = apply_curriculum_to_queue(queue, active_domains, allowed_difficulties, phase)
    if parked:
        TASK_QUEUE_PATH.write_text(json.dumps(queue, indent=2))
        print(f"  Parked {parked} out-of-scope tasks for later phases")

    pending = select_curriculum_tasks(queue["tasks"], active_domains, allowed_difficulties, n)
    if active_domains == ["self_correction"]:
        sprint_categories = {
            "deliberate_error_detection",
            "completion_check",
            "conflicting_info",
            "progressive_refinement",
            "false_confidence_calibration",
        }
        pending = [t for t in pending if t.get("category") in sprint_categories]

    if len(pending) < n:
        weakness_data = [w for w in analyze_weaknesses() if w["domain"] in active_domains]
        weak_domains = [w["domain"] for w in weakness_data] if weakness_data else active_domains
        needed = n - len(pending) + 20  # Extra buffer
        print(f"  Need {needed} more phase-aligned tasks (have {len(pending)} pending)")

        if codex_ok:
            new_tasks = codex_generate_tasks(
                needed,
                weak_domains,
                weakness_data,
                allowed_domains=active_domains,
                allowed_difficulties=allowed_difficulties,
                phase=phase,
            )
        else:
            from autonomous_loop import generate_tasks
            new_tasks = generate_tasks(bonsai, needed, weak_domains or None)

        if new_tasks:
            existing = {t["id"] for t in queue["tasks"]}
            num = 1
            for t in new_tasks:
                while f"auto-{num:03d}" in existing:
                    num += 1
                t["id"] = f"auto-{num:03d}"
                t.setdefault("status", "pending")
                t.setdefault("difficulty_estimate", t.get("difficulty", "medium"))
                existing.add(t["id"])
                num += 1
            queue["tasks"].extend(new_tasks)
            TASK_QUEUE_PATH.write_text(json.dumps(queue, indent=2))
            pending = select_curriculum_tasks(queue["tasks"], active_domains, allowed_difficulties, n)
            print(f"  Added {len(new_tasks)} tasks (queue: {len(queue['tasks'])} total)")

    tasks = pending[:n]
    if not tasks:
        print("No phase-aligned tasks.")
        return

    print(f"  Running {len(tasks)} phase-aligned tasks\n")

    # Process tasks
    results = []
    raw_quality = {"pass": 0, "partial": 0, "fail": 0, "unknown": 0}
    batch_start = time.time()

    for i, task in enumerate(tasks):
        # Stage 1: Bonsai generates
        result = stage1_generate(bonsai, task)

        # Stage 2: Codex distills
        if codex_ok:
            result = codex_distill(result)
        else:
            from karpathy_loop import stage2_distill_inline
            result = stage2_distill_inline(result)

        results.append(result)
        raw_quality[result.raw_quality] = raw_quality.get(result.raw_quality, 0) + 1

        # Progress every 10 tasks
        if (i + 1) % 10 == 0:
            elapsed = time.time() - batch_start
            rate = (i + 1) / elapsed
            remaining = (len(tasks) - i - 1) / rate if rate > 0 else 0
            validated = sum(1 for r in results if r.validated)
            print(f"  --- {i+1}/{len(tasks)} done ({rate:.1f}/s, ~{remaining:.0f}s left, {validated} validated) ---\n")

    # Stage 3: Compile
    print(f"\n{'='*60}")
    print("Stage 3: Compile")
    print(f"{'='*60}")

    batch_path = compile_training_jsonl(results)
    metrics = compute_metrics(results, phase_state)
    save_metrics(metrics, iteration_num)
    update_task_queue(results)

    # Update phase state after the batch
    phase_state.setdefault("history", []).append({
        "iteration": iteration_num,
        "timestamp": metrics.get("timestamp"),
        "phase": metrics.get("phase", phase),
        "raw_passes": metrics.get("raw_passes", 0),
        "validation_rate": metrics.get("validation_rate", 0),
    })
    phase_state["iterations_in_phase"] = int(phase_state.get("iterations_in_phase", 0)) + 1
    phase_lock = os.getenv("HERMES_PHASE_LOCK", "0").strip().lower() not in ("0", "false", "no", "off")
    if metrics.get("raw_passes", 0) >= cfg["gate_raw_passes"] and phase < 3:
        if phase_lock:
            phase_state["phase"] = phase
            phase_state["diagnostic_next_batch"] = False
            print(f"  Phase gate reached; phase lock active, staying in phase {phase_state['phase']}")
        else:
            phase_state["phase"] = phase + 1
            phase_state["iterations_in_phase"] = 0
            phase_state["phase_started_at"] = datetime.now().isoformat()
            phase_state["diagnostic_next_batch"] = True
            print(f"  Phase gate reached; advancing to phase {phase_state['phase']}")
    else:
        phase_state["phase"] = phase
        phase_state["diagnostic_next_batch"] = False
    save_phase_state(phase_state)

    # Update Hermes memory
    update_hermes_memory(metrics, iteration_num, raw_quality)

    # Final report
    vr = metrics.get("validation_rate", 0)
    avg_c = metrics.get("avg_scores", {}).get("correctness", 0)
    total = sum(1 for _ in open(TRAINING_JSONL)) if TRAINING_JSONL.exists() else 0
    elapsed = time.time() - t_start

    print(f"\n  Validation: {vr*100:.1f}% | Correctness: {avg_c:.1f}/10")
    print(f"  Raw quality: {json.dumps(raw_quality)}")
    print(f"  Raw passes: {metrics.get('raw_passes', 0)}")
    print(f"  Hint passes: {metrics.get('hint_passes', 0)} | Full corrections: {metrics.get('full_corrections', 0)} | Hint conversion rate: {metrics.get('hint_conversion_rate', 0):.3f}")
    print(f"  Raw pass domains: {metrics.get('raw_pass_domains', [])}")
    print(f"  Partial count: {metrics.get('partial_count', 0)} | Partial domains: {metrics.get('partial_domains', [])}")
    print(f"  False confidence: {metrics.get('false_confidence_count', 0)} | Rate: {metrics.get('false_confidence_rate', 0):.3f} | By category: {metrics.get('false_confidence_by_category', {})}")
    print(f"  Frontier tasks: {metrics.get('frontier_tasks', [])}")
    print(f"  Frontier failure modes: {metrics.get('frontier_failure_modes', {})}")
    print(f"  Output length analysis: {metrics.get('output_length_analysis', {})}")
    print(f"  Checklist format applied: {metrics.get('checklist_format_applied', False)}")
    print(f"  Phase: {metrics.get('phase', phase)} | Iterations in phase: {metrics.get('iterations_in_phase', phase_state.get('iterations_in_phase', 0))}")
    print(f"  Scores: {metrics.get('avg_scores', {})}")
    print(f"  Total training entries: {total}")
    print(f"  Time: {elapsed:.0f}s ({len(tasks)/elapsed:.1f} tasks/s)")
    print(f"  Bonsai pass rate: {raw_quality['pass']}/{len(results)} ({raw_quality['pass']/max(len(results),1)*100:.0f}%)")

    # Data quality verification
    has_3d = sum(1 for r in results if r.scores and "direction" in r.scores)
    has_old = sum(1 for r in results if r.scores and "direction" not in r.scores)
    partials_with_type = 0
    partials_total = 0
    for r in results:
        if r.raw_quality in ("partial", "fail"):
            partials_total += 1
            # Check the gold trace file for partial_type
            gold_path = GOLD_DIR / f"{r.task_id}_gold.json"
            if gold_path.exists():
                try:
                    gd = json.loads(gold_path.read_text())
                    if gd.get("partial_type", "n/a") not in ("n/a", "", None):
                        partials_with_type += 1
                except:
                    pass
    print(f"  Scores: {has_3d} 3D / {has_old} old-schema")
    if partials_total > 0:
        print(f"  Partial types: {partials_with_type}/{partials_total} classified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    run_iteration(parser.parse_args().count)
