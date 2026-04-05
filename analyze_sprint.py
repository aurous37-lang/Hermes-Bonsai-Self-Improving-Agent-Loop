#!/usr/bin/env python3
"""
Post-sprint analysis. Run after a training sprint to get the full picture.

Compares:
  - Domain pass rates across eras (pre-cognitive, cognitive, completion-discipline)
  - 3D scoring breakdown (direction/specificity/completion) if available
  - Partial failure type distribution
  - Token anchoring presence in new traces
  - Dataset composition and readiness for Stage 1

Usage:
  python analyze_sprint.py
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path(__file__).parent
TRAINING = BASE / "karpathy_loop_output" / "training_data.jsonl"
GOLD_DIR = BASE / "karpathy_loop_output" / "gold_traces"
METRICS = BASE / "karpathy_loop_metrics" / "iteration_log.json"

TIERS = {
    "frontier": ["devops", "code_debugging", "logic_puzzle"],
    "strong": ["architecture", "agent_routing", "memory_integration", "self_correction"],
    "solved": ["math", "refusal_redirect", "research_synthesis"],
}


def get_tier(domain):
    for tier, domains in TIERS.items():
        if domain in domains:
            return tier
    return "unknown"


def analyze():
    print(f"\n{'#'*60}")
    print(f"# POST-SPRINT ANALYSIS")
    print(f"{'#'*60}\n")

    # --- Dataset size ---
    with open(TRAINING) as f:
        total = sum(1 for _ in f)
    print(f"Total training entries: {total}")

    # --- Iteration log ---
    log = json.loads(open(METRICS).read())
    print(f"Total iterations: {len(log)}")

    # Split eras
    era1 = [e for e in log if e["iteration"] < 90]       # pre-cognitive
    era2 = [e for e in log if 90 <= e["iteration"] < 116] # cognitive modes
    era3 = [e for e in log if e["iteration"] >= 116]      # completion discipline

    for name, era in [("Pre-cognitive (<90)", era1), ("Cognitive modes (90-115)", era2), ("Completion discipline (116+)", era3)]:
        if not era:
            continue
        vrs = [e["validation_rate"] for e in era]
        cors = [e["avg_scores"]["correctness"] for e in era if e.get("avg_scores")]
        print(f"\n  {name}: {len(era)} iterations")
        print(f"    Validation: {sum(vrs)/len(vrs)*100:.1f}%")
        if cors:
            print(f"    Correctness: {sum(cors)/len(cors):.1f}")

    # --- Gold trace analysis by era ---
    all_golds = sorted(GOLD_DIR.glob("*_gold.json"), key=lambda p: p.stat().st_mtime)

    # Split golds into thirds roughly
    third = len(all_golds) // 3
    eras_golds = {
        "Early": all_golds[:third],
        "Middle": all_golds[third:2*third],
        "Recent": all_golds[2*third:],
    }

    print(f"\n{'='*60}")
    print("DOMAIN PASS RATES BY ERA")
    print(f"{'='*60}")

    era_domain_stats = {}
    for era_name, golds in eras_golds.items():
        stats = {}
        for p in golds:
            try:
                d = json.loads(p.read_text())
                dom = d.get("domain", "?")
                q = d.get("raw_quality", "?")
                if dom not in stats:
                    stats[dom] = {"pass": 0, "partial": 0, "fail": 0}
                if q in ("pass", "partial", "fail"):
                    stats[dom][q] += 1
            except:
                pass
        era_domain_stats[era_name] = stats

    all_domains = sorted(set(
        d for stats in era_domain_stats.values() for d in stats.keys()
    ))

    print(f"\n  {'Domain':<25s} {'Early':>8s} {'Middle':>8s} {'Recent':>8s}  Trend")
    print(f"  {'—'*65}")
    for dom in all_domains:
        rates = []
        for era_name in ["Early", "Middle", "Recent"]:
            s = era_domain_stats.get(era_name, {}).get(dom, {"pass": 0, "partial": 0, "fail": 0})
            t = sum(s.values())
            rate = s["pass"] / t * 100 if t > 0 else 0
            rates.append((rate, t))

        trend = ""
        if rates[2][0] > rates[0][0] + 10:
            trend = "  IMPROVING"
        elif rates[2][0] < rates[0][0] - 10:
            trend = "  REGRESSED"
        else:
            trend = "  stable"

        parts = []
        for rate, n in rates:
            parts.append(f"{rate:5.0f}%({n:2d})")
        print(f"  {dom:<25s} {parts[0]:>8s} {parts[1]:>8s} {parts[2]:>8s}{trend}")

    # --- 3D Scoring (if available in recent traces) ---
    print(f"\n{'='*60}")
    print("3D SCORING BREAKDOWN (recent traces)")
    print(f"{'='*60}")

    recent_golds = all_golds[-100:]
    scores_3d = defaultdict(lambda: {"direction": [], "specificity": [], "completion": []})
    has_3d = False

    for p in recent_golds:
        try:
            d = json.loads(p.read_text())
            s = d.get("scores", {})
            dom = d.get("domain", "?")
            if "direction" in s:
                has_3d = True
                scores_3d[dom]["direction"].append(s["direction"])
                scores_3d[dom]["specificity"].append(s["specificity"])
                scores_3d[dom]["completion"].append(s["completion"])
        except:
            pass

    if has_3d:
        print(f"\n  {'Domain':<25s} {'Direction':>10s} {'Specificity':>12s} {'Completion':>11s}")
        print(f"  {'—'*60}")
        for dom in sorted(scores_3d.keys()):
            s = scores_3d[dom]
            d_avg = sum(s["direction"]) / len(s["direction"]) if s["direction"] else 0
            sp_avg = sum(s["specificity"]) / len(s["specificity"]) if s["specificity"] else 0
            c_avg = sum(s["completion"]) / len(s["completion"]) if s["completion"] else 0
            print(f"  {dom:<25s} {d_avg:>10.1f} {sp_avg:>12.1f} {c_avg:>11.1f}")
    else:
        print("  No 3D scoring data yet (new schema may not have propagated)")

    # --- Partial failure types ---
    print(f"\n{'='*60}")
    print("PARTIAL FAILURE TYPES (recent traces)")
    print(f"{'='*60}")

    partial_types = Counter()
    partial_by_domain = defaultdict(Counter)
    for p in recent_golds:
        try:
            d = json.loads(p.read_text())
            pt = d.get("partial_type", d.get("raw_quality", "unknown"))
            dom = d.get("domain", "?")
            if d.get("raw_quality") == "partial" or pt not in ("pass", "fail", "n/a", "unknown"):
                partial_types[pt] += 1
                partial_by_domain[dom][pt] += 1
        except:
            pass

    if partial_types:
        print(f"\n  Overall: {dict(partial_types)}")
        for dom in sorted(partial_by_domain.keys()):
            print(f"  {dom}: {dict(partial_by_domain[dom])}")
    else:
        print("  No partial type annotations yet")

    # --- Token anchoring check ---
    print(f"\n{'='*60}")
    print("TOKEN ANCHORING ([CMD:], [CODE:]) in recent traces")
    print(f"{'='*60}")

    cmd_count = 0
    code_count = 0
    for p in recent_golds:
        try:
            d = json.loads(p.read_text())
            text = json.dumps(d)
            if "[CMD:" in text:
                cmd_count += 1
            if "[CODE:" in text:
                code_count += 1
        except:
            pass

    print(f"  [CMD: ...] anchors found in {cmd_count}/{len(recent_golds)} recent traces")
    print(f"  [CODE: ...] anchors found in {code_count}/{len(recent_golds)} recent traces")

    # --- Dataset readiness ---
    print(f"\n{'='*60}")
    print("DATASET READINESS FOR STAGE 1")
    print(f"{'='*60}")

    with open(TRAINING) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    chatml = sum(1 for e in entries if "conversations" in e)
    structured = sum(1 for e in entries if "trace" in e)
    stage2 = sum(1 for e in entries if e.get("metadata", {}).get("stage") == 2)

    tier_counts = {"frontier": 0, "strong": 0, "solved": 0, "unknown": 0}
    for e in entries:
        dom = e.get("metadata", {}).get("domain", "unknown")
        tier_counts[get_tier(dom)] += 1

    print(f"\n  Total: {len(entries)} ({chatml} ChatML + {structured} structured)")
    print(f"  Stage 2 compressed exemplars: {stage2}")
    print(f"  Tier split: {json.dumps(tier_counts)}")
    print(f"  Recommended: run build_training_set.py to generate weighted split")

    print(f"\n{'#'*60}")
    print(f"# ANALYSIS COMPLETE")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    analyze()
