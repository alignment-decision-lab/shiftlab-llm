import argparse
import yaml
from pathlib import Path
from datetime import datetime

from shiftlab.core.registry import (
    DATASETS,
    SHIFT_METRICS,
    ADAPTERS,
    SHIFT_OPERATORS,
)


from shiftlab.eval.runner import run
from shiftlab.shift.constraint_shift import make_tighten_budget, make_change_penalty  # noqa: F401

# ensure modules are imported so they register
from shiftlab.data.toy_domains import make_toy_domains  # noqa: F401
from shiftlab.shift.metrics import make_token_jsd, make_oov_rate  # noqa: F401
from shiftlab.adapt.adapters import make_no_adapt, make_normalize_punct  # noqa: F401
from shiftlab.shift.operators import make_no_shift  # noqa: F401
from shiftlab.shift.data_shift import make_inject_punct  # noqa: F401
from shiftlab.shift.constraint_shift import make_tighten_budget, make_change_penalty  # noqa: F401
from shiftlab.adapt.recalibration import make_temperature_scaling  # noqa: F401

def _default_outdir() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / ts


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--outdir", default=None)

    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 0))

    # ----------------------------
    # Dataset creation
    # ----------------------------
    src, tgt = DATASETS.create(
        cfg["data"]["name"],
        seed=seed,
        **cfg["data"].get("params", {})
    )

    # ----------------------------
    # Shift operator (default: no_shift)
    # ----------------------------
    shift_cfg = cfg.get("shift_operator", {"name": "no_shift", "params": {}})
    shift_op = SHIFT_OPERATORS.create(
        shift_cfg["name"],
        **shift_cfg.get("params", {})
    )

    # ----------------------------
    # Metrics
    # ----------------------------
    metrics = [
        SHIFT_METRICS.create(mm["name"], **mm.get("params", {}))
        for mm in cfg["shift_metrics"]
    ]

    # ----------------------------
    # Adapter
    # ----------------------------
    adapter = ADAPTERS.create(
        cfg["adapter"]["name"],
        **cfg["adapter"].get("params", {})
    )

    # ----------------------------
    # Output directory
    # ----------------------------
    cfg_out = (cfg.get("report", {}) or {}).get("out_dir", None)

    if args.outdir:
        out_dir = Path(args.outdir)
    elif cfg_out:
        out_dir = Path(cfg_out)
    else:
        out_dir = _default_outdir()

    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Run experiment
    # ----------------------------
    run_dir, metrics_out = run(
        source=src,
        target=tgt,
        metrics=metrics,
        adapter=adapter,
        out_dir=str(out_dir),
        shift_op=shift_op,
        seed=seed,
    )

    print("Done. Metrics:", metrics_out)
    print("Artifacts in:", run_dir)


if __name__ == "__main__":
    main()