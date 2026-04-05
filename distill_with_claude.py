#!/usr/bin/env python3
"""
Gold-standard distillation using Claude/Opus via Claude Code.

This script reads raw traces from karpathy_loop_output/raw_traces/,
generates distillation prompts, and can either:
  1. Print prompts for manual Claude Code processing
  2. Apply gold traces from completed distillation files

Usage:
  python distill_with_claude.py --generate     # Print distillation prompts
  python distill_with_claude.py --apply         # Apply completed gold files
  python distill_with_claude.py --interactive   # One-by-one interactive mode
"""

import json
import sys
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "karpathy_loop_output" / "raw_traces"
GOLD_DIR = BASE_DIR / "karpathy_loop_output" / "gold_traces"
OUTPUT_DIR = BASE_DIR / "karpathy_loop_output"
TRAINING_JSONL = OUTPUT_DIR / "training_data.jsonl"

TOKEN_TARGETS = {"easy": (40, 120), "medium": (120, 250), "hard": (250, 500)}

DISTILL_TEMPLATE = """## Gold Trace Distillation Request

**Task ID:** {task_id}
**Difficulty:** {difficulty} (trace budget: {token_min}-{token_max} tokens)
**Domain:** {domain}
**Category:** {category}

### Task Prompt
{prompt}

### Bonsai Raw Attempt
```
{raw_trace}
```

### Instructions
Produce a gold-standard compact reasoning trace using this schema:

```json
{{
  "trace": {{
    "goal": "one clear objective",
    "constraints": ["constraint 1", "constraint 2"],
    "plan": ["step 1", "step 2", "step 3"],
    "checks": ["what could fail?", "what to verify?"],
    "decision": "chosen path and why"
  }},
  "output": "final concise answer",
  "think_tagged": "<think>\\nGoal: ...\\nConstraints: ...\\nPlan: ...\\nChecks: ...\\nDecision: ...\\n</think>\\n\\nAnswer...",
  "score": {{
    "direction": 0-10,
    "specificity": 0-10,
    "completion": 0-10
  }},
  "score_legacy": {{
    "correctness": 0-10,
    "conciseness": 0-10,
    "schema_consistency": 0-10,
    "actionability": 0-10
  }},
  "raw_attempt_quality": "pass|partial|fail",
  "notes": "observations"
}}
```

**Keep the trace within {token_min}-{token_max} tokens.** Correctness > conciseness > everything else.
Was the raw attempt correct? If so, mark raw_attempt_quality as "pass".
"""


def load_raw_traces() -> list[dict]:
    """Load all raw trace files."""
    traces = []
    for p in sorted(RAW_DIR.glob("*_raw.json")):
        traces.append(json.loads(p.read_text()))
    return traces


def get_undistilled() -> list[dict]:
    """Find raw traces that don't have gold counterparts yet."""
    raw = load_raw_traces()
    undistilled = []
    for r in raw:
        gold_path = GOLD_DIR / f"{r['task_id']}_gold.json"
        if not gold_path.exists():
            undistilled.append(r)
    return undistilled


def generate_prompts():
    """Print distillation prompts for all undistilled traces."""
    undistilled = get_undistilled()
    if not undistilled:
        print("All traces have been distilled.")
        return

    print(f"Found {len(undistilled)} undistilled traces.\n")

    for r in undistilled:
        tmin, tmax = TOKEN_TARGETS.get(r["difficulty"], (40, 500))
        prompt = DISTILL_TEMPLATE.format(
            task_id=r["task_id"],
            difficulty=r["difficulty"],
            token_min=tmin,
            token_max=tmax,
            domain=r["domain"],
            category=r["category"],
            prompt=r["prompt"],
            raw_trace=r["raw_trace"],
        )
        print(prompt)
        print("\n" + "=" * 70 + "\n")


def apply_gold_file(gold_file: Path, raw_data: dict) -> bool:
    """Apply a gold JSON file to create the full gold trace record."""
    try:
        gold = json.loads(gold_file.read_text())
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Invalid JSON in {gold_file.name}: {e}")
        return False

    # Merge raw data with gold distillation
    merged = {**raw_data}
    merged["gold_trace"] = gold.get("trace")
    merged["gold_think"] = gold.get("think_tagged")
    merged["gold_output"] = gold.get("output")
    merged["scores"] = gold.get("score")
    merged["raw_quality"] = gold.get("raw_attempt_quality", "unknown")

    # Validate
    if merged["scores"]:
        avg = sum(merged["scores"].values()) / len(merged["scores"])
        merged["validated"] = merged["scores"].get("correctness", 0) >= 7 and avg >= 6.5
    else:
        merged["validated"] = False

    # Save
    out_path = GOLD_DIR / f"{raw_data['task_id']}_gold.json"
    out_path.write_text(json.dumps(merged, indent=2))

    # Append to training JSONL if validated
    if merged["validated"] and merged["gold_think"]:
        chatml = {
            "conversations": [
                {"role": "system", "content": "You are Hermes, a concise autonomous assistant. Think briefly inside <think> tags before acting."},
                {"role": "user", "content": raw_data["prompt"]},
                {"role": "assistant", "content": merged["gold_think"]},
            ]
        }
        with open(TRAINING_JSONL, "a") as f:
            f.write(json.dumps(chatml) + "\n")

        structured = {
            "system": "You are Hermes, a concise autonomous assistant.",
            "input": raw_data["prompt"],
            "trace": merged["gold_trace"],
            "output": merged["gold_output"] or "",
            "metadata": {
                "task_id": raw_data["task_id"],
                "domain": raw_data["domain"],
                "difficulty": raw_data["difficulty"],
                "category": raw_data["category"],
                "scores": merged["scores"],
                "source": "claude_gold",
            },
        }
        with open(TRAINING_JSONL, "a") as f:
            f.write(json.dumps(structured) + "\n")

    status = "VALIDATED" if merged["validated"] else "NOT VALIDATED"
    print(f"  [{status}] {raw_data['task_id']} — scores: {merged['scores']}")
    return merged["validated"]


def apply_all():
    """Look for manually-created gold JSON files and apply them."""
    raw_traces = {r["task_id"]: r for r in load_raw_traces()}

    # Look for gold response files (user drops them as {task_id}_response.json)
    applied = 0
    for resp_file in sorted(GOLD_DIR.glob("*_response.json")):
        task_id = resp_file.stem.replace("_response", "")
        if task_id in raw_traces:
            if apply_gold_file(resp_file, raw_traces[task_id]):
                applied += 1

    print(f"\nApplied {applied} gold traces.")


def interactive():
    """Interactive mode: show one trace at a time, paste gold JSON."""
    undistilled = get_undistilled()
    if not undistilled:
        print("All traces distilled.")
        return

    raw_map = {r["task_id"]: r for r in load_raw_traces()}

    for r in undistilled:
        tmin, tmax = TOKEN_TARGETS.get(r["difficulty"], (40, 500))
        print(f"\n{'='*60}")
        print(f"Task: {r['task_id']} ({r['difficulty']}, {r['domain']})")
        print(f"Budget: {tmin}-{tmax} tokens")
        print(f"{'='*60}")
        print(f"\nPrompt: {r['prompt'][:200]}...")
        print(f"\nRaw trace preview: {r['raw_trace'][:300]}...")
        print(f"\nPaste gold JSON (or 'skip' / 'quit'):")

        lines = []
        while True:
            line = input()
            if line.strip().lower() in ("skip", "quit"):
                if line.strip().lower() == "quit":
                    return
                break
            lines.append(line)
            # Try to parse — if valid JSON, we're done
            try:
                json.loads("\n".join(lines))
                break
            except json.JSONDecodeError:
                continue

        if lines:
            text = "\n".join(lines)
            try:
                gold = json.loads(text)
                resp_path = GOLD_DIR / f"{r['task_id']}_response.json"
                resp_path.write_text(json.dumps(gold, indent=2))
                apply_gold_file(resp_path, raw_map[r["task_id"]])
            except json.JSONDecodeError as e:
                print(f"  [ERROR] Invalid JSON: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Claude-powered gold trace distillation")
    parser.add_argument("--generate", action="store_true", help="Print distillation prompts")
    parser.add_argument("--apply", action="store_true", help="Apply completed gold files")
    parser.add_argument("--interactive", action="store_true", help="Interactive paste mode")
    parser.add_argument("--status", action="store_true", help="Show distillation status")

    args = parser.parse_args()

    if args.generate:
        generate_prompts()
    elif args.apply:
        apply_all()
    elif args.interactive:
        interactive()
    elif args.status:
        raw = load_raw_traces()
        undistilled = get_undistilled()
        print(f"Raw traces: {len(raw)}")
        print(f"Distilled:  {len(raw) - len(undistilled)}")
        print(f"Pending:    {len(undistilled)}")
        if undistilled:
            print(f"Next: {[u['task_id'] for u in undistilled[:5]]}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
