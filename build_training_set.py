#!/usr/bin/env python3
"""
Build a weighted training set from the 4,000 entries.

Weighting:
  50% frontier tier (devops, code_debugging, logic_puzzle)
  30% strong tier (architecture, agent_routing, memory_integration, self_correction)
  20% solved tier (math, refusal_redirect, research_synthesis)

Also produces separate frontier-only and stage2-compressed sets.

Usage:
  python build_training_set.py              # Build weighted train/eval split
  python build_training_set.py --frontier   # Frontier-only set for targeted training
  python build_training_set.py --stats      # Show dataset composition
"""
import json
import random
import argparse
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent
TRAINING_JSONL = BASE_DIR / "karpathy_loop_output" / "training_data.jsonl"
OUTPUT_DIR = BASE_DIR / "karpathy_loop_output"

TIERS = {
    "frontier": ["devops", "code_debugging", "logic_puzzle"],
    "strong": ["architecture", "agent_routing", "memory_integration", "self_correction"],
    "solved": ["math", "refusal_redirect", "research_synthesis"],
}

TIER_WEIGHTS = {"frontier": 0.50, "strong": 0.30, "solved": 0.20}


def get_domain(entry: dict) -> str:
    """Extract domain from either ChatML or structured format."""
    meta = entry.get("metadata", {})
    if meta.get("domain"):
        return meta["domain"]
    # For ChatML entries without metadata, try to infer from paired structured entry
    return meta.get("domain", "unknown")


def get_tier(domain: str) -> str:
    for tier, domains in TIERS.items():
        if domain in domains:
            return tier
    return "unknown"


def build_weighted(target_size: int = 2000, eval_ratio: float = 0.1):
    """Build a weighted training set."""
    with open(TRAINING_JSONL) as f:
        all_entries = [json.loads(l) for l in f if l.strip()]

    # Separate ChatML entries (what we train on) from structured
    chatml = [e for e in all_entries if "conversations" in e]
    structured = [e for e in all_entries if "trace" in e]

    # Build paired: each ChatML entry gets its domain from the structured partner
    # They're written in pairs, so entry i and i+1 are the same task
    paired_chatml = []
    for i, entry in enumerate(all_entries):
        if "conversations" not in entry:
            continue
        # Look for the next structured entry to get domain
        domain = "unknown"
        if i + 1 < len(all_entries) and "trace" in all_entries[i + 1]:
            domain = all_entries[i + 1].get("metadata", {}).get("domain", "unknown")
        elif i > 0 and "trace" in all_entries[i - 1]:
            domain = all_entries[i - 1].get("metadata", {}).get("domain", "unknown")
        paired_chatml.append((entry, domain))

    # Bucket by tier
    buckets = {"frontier": [], "strong": [], "solved": [], "unknown": []}
    for entry, domain in paired_chatml:
        tier = get_tier(domain)
        buckets[tier].append(entry)

    print(f"Raw bucket sizes:")
    for tier, entries in buckets.items():
        print(f"  {tier}: {len(entries)}")

    # Sample according to weights
    random.seed(42)
    train_set = []
    for tier, weight in TIER_WEIGHTS.items():
        n = int(target_size * weight)
        pool = buckets[tier]
        if len(pool) >= n:
            sampled = random.sample(pool, n)
        else:
            # Oversample if not enough
            sampled = pool * (n // len(pool) + 1)
            sampled = random.sample(sampled, n)
        train_set.extend(sampled)
        print(f"  Sampled {len(sampled)} from {tier} (target: {n})")

    # Add unknown bucket entries
    train_set.extend(buckets["unknown"])

    random.shuffle(train_set)

    # Split train/eval
    split = int(len(train_set) * (1 - eval_ratio))
    train = train_set[:split]
    eval_set = train_set[split:]

    # Write
    train_path = OUTPUT_DIR / "train_weighted.jsonl"
    eval_path = OUTPUT_DIR / "eval_weighted.jsonl"

    with open(train_path, "w") as f:
        for entry in train:
            f.write(json.dumps(entry) + "\n")
    with open(eval_path, "w") as f:
        for entry in eval_set:
            f.write(json.dumps(entry) + "\n")

    print(f"\nWeighted dataset:")
    print(f"  Train: {len(train)} -> {train_path.name}")
    print(f"  Eval:  {len(eval_set)} -> {eval_path.name}")
    return train, eval_set


def build_frontier_only():
    """Build a frontier-only set for targeted completion-discipline training."""
    with open(TRAINING_JSONL) as f:
        all_entries = [json.loads(l) for l in f if l.strip()]

    frontier_domains = set(TIERS["frontier"])
    frontier = []

    for i, entry in enumerate(all_entries):
        if "conversations" not in entry:
            continue
        domain = "unknown"
        if i + 1 < len(all_entries) and "trace" in all_entries[i + 1]:
            domain = all_entries[i + 1].get("metadata", {}).get("domain", "unknown")
        if domain in frontier_domains:
            frontier.append(entry)

    random.seed(42)
    random.shuffle(frontier)
    split = int(len(frontier) * 0.9)

    train_path = OUTPUT_DIR / "train_frontier.jsonl"
    eval_path = OUTPUT_DIR / "eval_frontier.jsonl"

    with open(train_path, "w") as f:
        for entry in frontier[:split]:
            f.write(json.dumps(entry) + "\n")
    with open(eval_path, "w") as f:
        for entry in frontier[split:]:
            f.write(json.dumps(entry) + "\n")

    print(f"Frontier-only dataset:")
    print(f"  Train: {split} -> {train_path.name}")
    print(f"  Eval:  {len(frontier) - split} -> {eval_path.name}")


def show_stats():
    """Show dataset composition."""
    with open(TRAINING_JSONL) as f:
        all_entries = [json.loads(l) for l in f if l.strip()]

    chatml = sum(1 for e in all_entries if "conversations" in e)
    structured = sum(1 for e in all_entries if "trace" in e)

    domains = Counter()
    for entry in all_entries:
        meta = entry.get("metadata", {})
        dom = meta.get("domain", "unknown")
        if dom != "unknown":
            domains[dom] += 1

    tiers_count = {"frontier": 0, "strong": 0, "solved": 0, "unknown": 0}
    for dom, count in domains.items():
        tier = get_tier(dom)
        tiers_count[tier] += count

    print(f"Total: {len(all_entries)} ({chatml} ChatML + {structured} structured)")
    print(f"\nBy tier:")
    for tier, count in tiers_count.items():
        pct = count / max(len(all_entries), 1) * 100
        print(f"  {tier}: {count} ({pct:.0f}%)")
    print(f"\nBy domain:")
    for dom, count in domains.most_common():
        print(f"  {dom}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--target", type=int, default=2000)
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.frontier:
        build_frontier_only()
    else:
        build_weighted(args.target)
