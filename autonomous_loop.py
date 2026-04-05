#!/usr/bin/env python3
"""
Autonomous Karpathy Loop — Hermes drives everything.

This script lets Bonsai:
  1. Generate its own task queue (new tasks in weak domains)
  2. Solve the tasks (raw reasoning)
  3. Self-distill into compact traces
  4. Self-score and validate
  5. Compile training data
  6. Identify weak areas and generate targeted follow-up tasks
  7. Repeat indefinitely

The only external dependency is the Hermes API at localhost:1234.

Usage:
  python autonomous_loop.py --iterations 5    # Run 5 full cycles
  python autonomous_loop.py --forever          # Run until stopped
  python autonomous_loop.py --generate-tasks 10  # Just generate 10 new tasks
"""

import json
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# Import the core pipeline
sys.path.insert(0, str(Path(__file__).parent))
from karpathy_loop import (
    LLMClient, TaskResult, HERMES_API, HERMES_MODEL, TOKEN_TARGETS,
    MIX_TARGETS, BASE_DIR, TASK_QUEUE_PATH, OUTPUT_DIR, RAW_DIR, GOLD_DIR,
    BATCH_DIR, METRICS_DIR, TRAINING_JSONL, ITERATION_LOG,
    stage1_generate, stage2_distill_inline, compile_training_jsonl,
    compute_metrics, save_metrics, update_task_queue, get_next_iteration_num,
    _extract_json,
)


# ---------------------------------------------------------------------------
# Task Generation — Bonsai creates its own training tasks
# ---------------------------------------------------------------------------
DOMAINS = [
    "logic_puzzle", "math", "code_debugging", "architecture", "devops",
    "agent_routing", "self_correction", "research_synthesis",
    "refusal_redirect", "memory_integration",
]

CATEGORIES = list(MIX_TARGETS.keys())
DIFFICULTIES = ["easy", "medium", "hard"]

TASK_GEN_SYSTEM = """You are a training task generator for an AI agent called Hermes.
Generate diverse, specific tasks that test reasoning ability.

Output ONLY valid JSON — an array of task objects. No markdown, no explanation.
Each task must have: id, domain, difficulty, category, prompt, expected_skills.

Rules:
- Prompts must be specific and self-contained (no external data needed)
- Easy tasks: single-step, factual, one tool choice
- Medium tasks: multi-step, require verification
- Hard tasks: require planning, constraint analysis, trade-off reasoning
- Each prompt should be 1-3 sentences maximum
- expected_skills is a list of 2-3 skill tags"""


def generate_tasks(client: LLMClient, count: int = 5,
                   focus_domains: list[str] | None = None,
                   focus_categories: list[str] | None = None) -> list[dict]:
    """Have Bonsai generate new tasks, optionally focused on weak areas."""

    domain_hint = ""
    if focus_domains:
        domain_hint = f"\nFOCUS on these domains (they need more coverage): {', '.join(focus_domains)}"

    category_hint = ""
    if focus_categories:
        category_hint = f"\nFOCUS on these categories (underrepresented): {', '.join(focus_categories)}"

    prompt = f"""Generate exactly {count} training tasks as a JSON array.

Available domains: {', '.join(DOMAINS)}
Available categories: {', '.join(CATEGORIES)}
Available difficulties: {', '.join(DIFFICULTIES)}
{domain_hint}
{category_hint}

Use IDs like "auto-001", "auto-002", etc. starting from the next available number.

Example task:
{{"id": "auto-001", "domain": "math", "difficulty": "easy", "category": "task_decomposition", "prompt": "A store has a 20% off sale. An item costs $45. What is the sale price?", "expected_skills": ["percentage calculation", "discount application"]}}

Generate {count} tasks as a JSON array. Output ONLY the JSON array."""

    messages = [
        {"role": "system", "content": TASK_GEN_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    print(f"  [TaskGen] Asking Bonsai to generate {count} tasks...", end=" ", flush=True)
    resp = client.chat(messages, temperature=0.8, max_tokens=4096, timeout=180)
    content = resp.get("content", "")
    print("done")

    # Parse the array — handle JSON array, JSONL (newline-delimited), or single object
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        cleaned = cleaned.strip()

        tasks = None

        # Try 1: JSON array
        start = cleaned.find("[")
        if start != -1:
            end = cleaned.rfind("]")
            if end > start:
                try:
                    tasks = json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError:
                    pass

        # Try 2: Newline-delimited JSON objects (JSONL style)
        if tasks is None:
            tasks = []
            for line in cleaned.split("\n"):
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    obj = _extract_json(line)
                    tasks.append(obj)
                except (json.JSONDecodeError, Exception):
                    continue

        # Try 3: Single object
        if not tasks:
            obj = _extract_json(cleaned)
            tasks = [obj]

        # Validate and fix tasks
        valid_tasks = []
        for t in tasks:
            if not isinstance(t, dict):
                continue
            if "prompt" not in t:
                continue
            # Ensure required fields
            t.setdefault("id", f"auto-{len(valid_tasks) + 1:03d}")
            t.setdefault("domain", "logic_puzzle")
            t.setdefault("difficulty", "medium")
            t.setdefault("category", "task_decomposition")
            t.setdefault("expected_skills", [])
            t["status"] = "pending"
            valid_tasks.append(t)

        print(f"  [TaskGen] Generated {len(valid_tasks)} valid tasks")
        return valid_tasks

    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [TaskGen] Parse failed: {e}")
        # Save for debugging
        fail_path = METRICS_DIR / f"taskgen_failed_{datetime.now().strftime('%H%M%S')}.txt"
        fail_path.write_text(content)
        return []


def get_existing_task_ids() -> set[str]:
    """Get all existing task IDs from the queue."""
    if TASK_QUEUE_PATH.exists():
        queue = json.loads(TASK_QUEUE_PATH.read_text())
        return {t["id"] for t in queue["tasks"]}
    return set()


def add_tasks_to_queue(new_tasks: list[dict]):
    """Add new tasks to the queue, avoiding ID collisions."""
    queue = json.loads(TASK_QUEUE_PATH.read_text())
    existing_ids = {t["id"] for t in queue["tasks"]}

    # Renumber to avoid collisions
    next_num = 1
    for t in new_tasks:
        while f"auto-{next_num:03d}" in existing_ids:
            next_num += 1
        t["id"] = f"auto-{next_num:03d}"
        existing_ids.add(t["id"])
        next_num += 1

    queue["tasks"].extend(new_tasks)
    TASK_QUEUE_PATH.write_text(json.dumps(queue, indent=2))
    print(f"  [Queue] Added {len(new_tasks)} tasks (total: {len(queue['tasks'])})")


def analyze_weaknesses() -> tuple[list[str], list[str]]:
    """Analyze metrics to find weak domains and underrepresented categories."""
    if not ITERATION_LOG.exists():
        return [], []

    log = json.loads(ITERATION_LOG.read_text())
    if not log:
        return [], []

    # Aggregate recent iterations
    recent = log[-3:]  # Last 3 iterations

    # Find domains with low validation
    domain_results = {}
    for entry in recent:
        for domain in entry.get("domains_covered", []):
            domain_results.setdefault(domain, []).append(entry["validation_rate"])

    weak_domains = [d for d, rates in domain_results.items()
                    if sum(rates) / len(rates) < 0.7]

    # Find underrepresented categories
    total_mix = {}
    total_count = 0
    for entry in recent:
        for cat, frac in entry.get("category_mix_actual", {}).items():
            total_mix[cat] = total_mix.get(cat, 0) + frac * entry["total_tasks"]
        total_count += entry["total_tasks"]

    if total_count > 0:
        actual_mix = {k: v / total_count for k, v in total_mix.items()}
    else:
        actual_mix = {}

    under_cats = []
    for cat, target in MIX_TARGETS.items():
        actual = actual_mix.get(cat, 0)
        if actual < target * 0.5:  # More than 50% below target
            under_cats.append(cat)

    return weak_domains, under_cats


# ---------------------------------------------------------------------------
# Autonomous Loop
# ---------------------------------------------------------------------------
def run_autonomous_iteration(client: LLMClient, tasks_per_iter: int = 5):
    """Run one complete autonomous iteration."""
    iteration_num = get_next_iteration_num()

    print(f"\n{'='*60}")
    print(f"AUTONOMOUS ITERATION {iteration_num}")
    print(f"{'='*60}\n")

    # Step 1: Analyze weaknesses and generate targeted tasks if needed
    queue = json.loads(TASK_QUEUE_PATH.read_text())
    pending = [t for t in queue["tasks"] if t["status"] in ("pending", "retry")]

    if len(pending) < tasks_per_iter:
        print("[Phase: Task Generation]")
        weak_domains, under_cats = analyze_weaknesses()
        if weak_domains:
            print(f"  Weak domains: {weak_domains}")
        if under_cats:
            print(f"  Underrepresented categories: {under_cats}")

        needed = tasks_per_iter - len(pending) + 5  # Generate extras
        new_tasks = generate_tasks(
            client, count=needed,
            focus_domains=weak_domains or None,
            focus_categories=under_cats or None,
        )
        if new_tasks:
            add_tasks_to_queue(new_tasks)

        # Reload queue
        queue = json.loads(TASK_QUEUE_PATH.read_text())
        pending = [t for t in queue["tasks"] if t["status"] in ("pending", "retry")]

    # Step 2: Select tasks for this iteration
    tasks = pending[:tasks_per_iter]
    if not tasks:
        print("  No tasks available. Generating fresh batch...")
        new_tasks = generate_tasks(client, count=tasks_per_iter)
        if new_tasks:
            add_tasks_to_queue(new_tasks)
            tasks = new_tasks[:tasks_per_iter]

    if not tasks:
        print("  Failed to generate tasks. Stopping.")
        return None, None

    print(f"\n[Phase: Research & Generate]")
    print(f"  Running {len(tasks)} tasks: {[t['id'] for t in tasks]}\n")

    # Step 3: Generate raw traces and self-distill
    results = []
    for task in tasks:
        result = stage1_generate(client, task)
        result = stage2_distill_inline(result)
        results.append(result)
        print()

    # Step 4: Compile and measure
    print(f"\n[Phase: Compile & Measure]")
    batch_path = compile_training_jsonl(results)
    metrics = compute_metrics(results)
    save_metrics(metrics, iteration_num)
    update_task_queue(results)

    # Step 5: Report
    vr = metrics.get("validation_rate", 0)
    avg_c = metrics.get("avg_scores", {}).get("correctness", 0)
    fails = metrics.get("distill_failures", 0)

    print(f"\n  Results:")
    print(f"    Validation: {vr*100:.1f}% | Correctness: {avg_c:.1f}/10 | Distill failures: {fails}")
    print(f"    Training batch: {batch_path.name}")

    # Count total training data
    total_entries = 0
    if TRAINING_JSONL.exists():
        with open(TRAINING_JSONL) as f:
            total_entries = sum(1 for _ in f)
    print(f"    Total training entries: {total_entries}")

    return results, metrics


def run_loop(max_iterations: int = -1, tasks_per_iter: int = 5,
             target_entries: int = 500):
    """Run the autonomous loop until target is reached or max iterations hit."""
    client = LLMClient(HERMES_API, HERMES_MODEL)
    iteration = 0

    print(f"\n{'#'*60}")
    print(f"# Karpathy Autonomous Loop")
    print(f"# Target: {target_entries} training entries")
    print(f"# Tasks per iteration: {tasks_per_iter}")
    print(f"{'#'*60}\n")

    while max_iterations == -1 or iteration < max_iterations:
        iteration += 1

        results, metrics = run_autonomous_iteration(client, tasks_per_iter)
        if results is None:
            print("Loop stopped — no results.")
            break

        # Check if we've hit the target
        total_entries = 0
        if TRAINING_JSONL.exists():
            with open(TRAINING_JSONL) as f:
                total_entries = sum(1 for _ in f)

        if total_entries >= target_entries:
            print(f"\n{'='*60}")
            print(f"TARGET REACHED: {total_entries} entries (goal: {target_entries})")
            print(f"{'='*60}\n")
            break

        # Brief pause between iterations
        print(f"\n  --- Cooling down 2s before next iteration ---\n")
        time.sleep(2)

    # Final report
    from karpathy_loop import print_report
    print_report()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Autonomous Karpathy Loop")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of iterations to run (default: 5)")
    parser.add_argument("--forever", action="store_true",
                        help="Run until stopped (Ctrl+C)")
    parser.add_argument("--tasks-per-iter", type=int, default=5,
                        help="Tasks per iteration (default: 5)")
    parser.add_argument("--target", type=int, default=500,
                        help="Target number of training entries (default: 500)")
    parser.add_argument("--generate-tasks", type=int,
                        help="Just generate N new tasks and exit")

    args = parser.parse_args()

    if args.generate_tasks:
        client = LLMClient(HERMES_API, HERMES_MODEL)
        weak_domains, under_cats = analyze_weaknesses()
        tasks = generate_tasks(client, args.generate_tasks, weak_domains, under_cats)
        if tasks:
            add_tasks_to_queue(tasks)
            for t in tasks:
                print(f"  [{t['id']}] ({t['difficulty']}/{t['domain']}) {t['prompt'][:80]}...")
        return

    max_iter = -1 if args.forever else args.iterations
    try:
        run_loop(max_iter, args.tasks_per_iter, args.target)
    except KeyboardInterrupt:
        print("\n\nLoop interrupted by user.")
        from karpathy_loop import print_report
        print_report()


if __name__ == "__main__":
    main()
