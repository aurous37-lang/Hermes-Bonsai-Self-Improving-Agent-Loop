import os
import time
from pathlib import Path

# Run from the Hermes training environment.
os.environ.setdefault("HERMES_PHASE_LOCK", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from hermes_iteration import PHASE_CONFIG, load_phase_state, save_phase_state, run_iteration

SELF_ONLY = {
    "domains": ["self_correction"],
    "domain_mix": {"self_correction": 1.0},
    "difficulties": ["medium"],
    "gate_raw_passes": 10,
    "name": "Self-Correction Speed Run",
}

# Patch every phase config so the loop cannot drift into other domains while this sprint runs.
for phase in list(PHASE_CONFIG.keys()):
    PHASE_CONFIG[phase].update(SELF_ONLY)

state = load_phase_state()
state["phase"] = 2
state["iterations_in_phase"] = int(state.get("iterations_in_phase", 0))
state["diagnostic_next_batch"] = False
save_phase_state(state)

END_AT = time.time() + 30 * 60
iter_num = 0
print("[sprint] self_correction-only speed run started", flush=True)
print("[sprint] phase config patched to self_correction-only / medium only", flush=True)
print("[sprint] time budget: 30 minutes", flush=True)

while time.time() < END_AT:
    iter_num += 1
    remaining = int(END_AT - time.time())
    print(f"[sprint] iteration {iter_num} start | time_remaining={remaining}s", flush=True)
    run_iteration(10)
    print(f"[sprint] iteration {iter_num} end", flush=True)

print("[sprint] time budget complete", flush=True)
