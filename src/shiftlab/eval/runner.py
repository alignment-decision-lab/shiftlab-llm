import os, time, json
from shiftlab.core.types import ShiftReport

def run(source, target, metrics, adapter, out_dir: str):
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(out_dir, ts)
    os.makedirs(run_dir, exist_ok=True)

    # apply adaptation only to target (simple convention)
    adapted_target = adapter.apply(target.texts)

    results = {}
    for m in metrics:
        results[m.name] = m.compute(source.texts, adapted_target)

    report = ShiftReport(metrics=results)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(report.metrics, f, indent=2)

    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write(f"# ShiftLab run {ts}\n\n")
        f.write(f"- adapter: {adapter.name}\n\n")
        for k, v in report.metrics.items():
            f.write(f"- {k}: {v:.6f}\n")

    return run_dir, report.metrics
