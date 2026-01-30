import argparse, yaml
from shiftlab.core.registry import DATASETS, SHIFT_METRICS, ADAPTERS
from shiftlab.eval.runner import run

# ensure modules are imported so they register
from shiftlab.data.toy_domains import make_toy_domains  # noqa: F401
from shiftlab.shift.metrics import make_token_jsd, make_oov_rate  # noqa: F401
from shiftlab.adapt.adapters import make_no_adapt, make_normalize_punct  # noqa: F401

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 0))

    src, tgt = DATASETS.create(cfg["data"]["name"], seed=seed, **cfg["data"].get("params", {}))

    metrics = []
    for mm in cfg["shift_metrics"]:
        metrics.append(SHIFT_METRICS.create(mm["name"], **mm.get("params", {})))

    adapter = ADAPTERS.create(cfg["adapter"]["name"], **cfg["adapter"].get("params", {}))

    run_dir, metrics_out = run(src, tgt, metrics, adapter, out_dir=cfg["report"]["out_dir"])
    print("Done. Metrics:", metrics_out)
    print("Artifacts in:", run_dir)

if __name__ == "__main__":
    main()
