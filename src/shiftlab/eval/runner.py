import os, time, json
import random
from dataclasses import replace

from shiftlab.core.types import ShiftReport

def run(source, target, metrics, adapter, out_dir, shift_op=None, seed: int = 0):
    rng = random.Random(seed)

    if shift_op is not None:
        target = shift_op.apply(target, rng)   # <-- use target, not tgt/undefined

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(out_dir, ts)
    os.makedirs(run_dir, exist_ok=True)

    # apply adaptation to target texts
    adapted_texts = adapter.apply(target.texts)

    # IMPORTANT: keep constraints/metadata by creating a new DomainData
    adapted_target = replace(target, texts=adapted_texts)

    results = {}
    for m in metrics:
        results[m.name] = m.compute(source, adapted_target)  # <-- pass objects

    report = ShiftReport(metrics=results)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(report.metrics, f, indent=2)

    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write(f"# ShiftLab run {ts}\n\n")
        f.write(f"- adapter: {adapter.name}\n\n")
        if shift_op is not None:
            f.write(f"- shift: {getattr(shift_op, 'name', shift_op.__class__.__name__)}\n\n")
        for k, v in report.metrics.items():
            f.write(f"- {k}: {v:.6f}\n")

    return run_dir, report.metrics