import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent
GOLD_DIR = BASE / 'karpathy_loop_output' / 'gold_traces'
OUT_TRAIN = BASE / 'karpathy_loop_output' / 'train_stage2_weighted.jsonl'
OUT_EVAL = BASE / 'karpathy_loop_output' / 'eval_stage2_weighted.jsonl'
MANIFEST = BASE / 'karpathy_loop_metrics' / 'stage2_weighted_manifest.json'

SEED = 42
random.seed(SEED)

SYSTEM = 'You are Hermes, a concise autonomous assistant. Think briefly inside <think> tags before acting.'
TARGET_TRAIN = 600
TARGET_EVAL = 80
DOMAIN_WEIGHTS = {
    'self_correction': 0.40,
    'memory_integration': 0.30,
    'other': 0.30,
}

SELF_CORRECTION_CATEGORIES = {
    'deliberate_error_detection',
    'completion_check',
    'conflicting_info',
    'progressive_refinement',
    'false_confidence_calibration',
}


def strip_requirements(text: str) -> str:
    if not text:
        return ''
    txt = text.strip()
    if txt.startswith('[REQUIREMENTS]'):
        lines = txt.splitlines()
        out = []
        skip = True
        for line in lines:
            if line.strip() == '[/REQUIREMENTS]':
                skip = False
                continue
            if skip:
                continue
            out.append(line)
        txt = '\n'.join(out).strip()
    return txt


def wrap_think(trace: dict | None, think_tagged: str | None, output: str | None) -> str:
    if think_tagged:
        cleaned = strip_requirements(think_tagged)
        return cleaned
    trace_txt = json.dumps(trace or {}, indent=2, ensure_ascii=False)
    out = output or ''
    return f"<think>\n{trace_txt}\n</think>\n\n{out}".strip()


def build_conv(prompt: str, gold: dict) -> dict:
    assistant = wrap_think(gold.get('gold_trace'), gold.get('gold_think'), gold.get('gold_output'))
    return {
        'conversations': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': assistant},
        ]
    }


def build_struct(prompt: str, gold: dict) -> dict:
    return {
        'system': 'You are Hermes, a concise autonomous assistant.',
        'input': prompt,
        'trace': gold.get('gold_trace') or {},
        'output': gold.get('gold_output') or '',
        'metadata': {
            'task_id': gold.get('task_id'),
            'domain': gold.get('domain'),
            'difficulty': gold.get('difficulty'),
            'category': gold.get('category'),
            'scores': gold.get('scores') or {},
            'iteration_timestamp': gold.get('timestamp') or gold.get('iteration_timestamp') or '',
            'false_confidence': bool(gold.get('false_confidence', False)),
            'source': 'gold_trace',
        },
    }


def domain_bucket(domain: str) -> str:
    return domain if domain in ('self_correction', 'memory_integration') else 'other'


def main():
    gold_files = sorted(GOLD_DIR.glob('*_gold.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    pool = []
    for p in gold_files[:1200]:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        # Focus on directly useful correction examples.
        if d.get('raw_quality') not in ('partial', 'fail', 'pass'):
            continue
        prompt = d.get('prompt') or d.get('input') or ''
        if not prompt:
            continue
        pool.append(d)

    buckets = defaultdict(list)
    for d in pool:
        buckets[domain_bucket(d.get('domain', 'other'))].append(d)

    # Within self_correction, bias toward the sprint categories if we have them.
    self_pool = [d for d in buckets['self_correction'] if d.get('category') in SELF_CORRECTION_CATEGORIES]
    if self_pool:
        buckets['self_correction'] = self_pool

    # Shuffle deterministically within buckets.
    for k in list(buckets.keys()):
        random.shuffle(buckets[k])

    train_target = TARGET_TRAIN
    eval_target = TARGET_EVAL
    train_by_bucket = {
        'self_correction': int(round(train_target * DOMAIN_WEIGHTS['self_correction'])),
        'memory_integration': int(round(train_target * DOMAIN_WEIGHTS['memory_integration'])),
    }
    train_by_bucket['other'] = train_target - train_by_bucket['self_correction'] - train_by_bucket['memory_integration']

    eval_by_bucket = {
        'self_correction': int(round(eval_target * DOMAIN_WEIGHTS['self_correction'])),
        'memory_integration': int(round(eval_target * DOMAIN_WEIGHTS['memory_integration'])),
    }
    eval_by_bucket['other'] = eval_target - eval_by_bucket['self_correction'] - eval_by_bucket['memory_integration']

    selected_train = []
    selected_eval = []
    manifest = {
        'seed': SEED,
        'gold_pool_size': len(pool),
        'bucket_sizes': {k: len(v) for k, v in buckets.items()},
        'train_target': train_target,
        'eval_target': eval_target,
        'train_by_bucket': train_by_bucket,
        'eval_by_bucket': eval_by_bucket,
        'train_counts': {},
        'eval_counts': {},
    }

    def take(bucket_name: str, n: int):
        out = []
        src = buckets.get(bucket_name, [])
        while src and len(out) < n:
            out.append(src.pop())
        return out

    for bucket_name, n in train_by_bucket.items():
        selected_train.extend(take(bucket_name, n))
    # fill shortfalls from remaining pool, prioritizing target buckets in order
    while len(selected_train) < train_target:
        any_added = False
        for bucket_name in ('self_correction', 'memory_integration', 'other'):
            if len(selected_train) >= train_target:
                break
            extra = take(bucket_name, 1)
            if extra:
                selected_train.extend(extra)
                any_added = True
        if not any_added:
            break

    for bucket_name, n in eval_by_bucket.items():
        selected_eval.extend(take(bucket_name, n))
    while len(selected_eval) < eval_target:
        any_added = False
        for bucket_name in ('self_correction', 'memory_integration', 'other'):
            if len(selected_eval) >= eval_target:
                break
            extra = take(bucket_name, 1)
            if extra:
                selected_eval.extend(extra)
                any_added = True
        if not any_added:
            break

    def write_jsonl(path: Path, examples: list[dict]):
        lines = []
        for d in examples:
            prompt = d.get('prompt') or d.get('input') or ''
            conv = build_conv(prompt, d)
            struct = build_struct(prompt, d)
            lines.append(json.dumps(conv, ensure_ascii=False))
            lines.append(json.dumps(struct, ensure_ascii=False))
        path.write_text('\n'.join(lines) + ('\n' if lines else ''))

    write_jsonl(OUT_TRAIN, selected_train)
    write_jsonl(OUT_EVAL, selected_eval)

    def count_by(examples):
        c = defaultdict(int)
        for d in examples:
            c[domain_bucket(d.get('domain', 'other'))] += 1
        return dict(c)

    manifest['train_counts'] = count_by(selected_train)
    manifest['eval_counts'] = count_by(selected_eval)
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    print(f'Wrote {OUT_TRAIN}')
    print(f'Wrote {OUT_EVAL}')


if __name__ == '__main__':
    main()
