"""Create the empty folder skeleton for the model-bank Hugging Face repo.

Pushes a placeholder README into every (dataset x lambda) subfolder that
models_bank.py will eventually save a real checkpoint into, plus a top-level
model card and a metadata CSV stub -- so the intended structure is visible
and browsable before any training has run.

The grid of datasets and lambdas is read directly from the source of truth
(models_bank.py's datasets_config dict, configs/diagnostic/models_bank.yaml's
lambda list) instead of being duplicated here, so this script can't silently
drift out of sync with what train_one_bank_model actually produces.

Usage:
    python scripts/create_hf_model_bank_skeleton.py --dry-run
    python scripts/create_hf_model_bank_skeleton.py
"""
import argparse
import ast
from pathlib import Path

import yaml
from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "alignment-decision-lab/robustness-model-bank"
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_BANK_PY = REPO_ROOT / "diagnostic_experiment" / "models_bank.py"
MODELS_BANK_YAML = REPO_ROOT / "configs" / "diagnostic" / "models_bank.yaml"


def get_dataset_names():
    """Statically extract the top-level keys of datasets_config from
    models_bank.py, without importing the module (which needs torch)."""
    tree = ast.parse(MODELS_BANK_PY.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "datasets_config" for t in node.targets
        ):
            return [key.value for key in node.value.keys]

    raise RuntimeError(f"Could not find datasets_config in {MODELS_BANK_PY}")


def get_lambdas():
    """Read the trained lambda grid (bracket endpoints, not the finer
    lambda_grid used only for curve-fitting) from models_bank.yaml."""
    config = yaml.safe_load(MODELS_BANK_YAML.read_text())
    return config["model_bank"]["lambdas"]


def dataset_dir_name(dataset_name):
    return dataset_name.replace(" ", "_")


def lambda_dir_name(lambda_value):
    return f"lambda_{lambda_value:g}"


def build_grid():
    dataset_names = get_dataset_names()
    lambdas = get_lambdas()
    return [
        (dataset_name, lambda_value)
        for dataset_name in dataset_names
        for lambda_value in lambdas
    ]


def checkpoint_placeholder(dataset_name, lambda_value):
    return f"""# Pending checkpoint

- **dataset**: {dataset_name}
- **lambda**: {lambda_value:g}
- **model**: gpt2-medium, trained with KL-DRO
- **produced by**: `diagnostic_experiment/models_bank.py`
- **config**: `configs/diagnostic/models_bank.yaml`

No checkpoint has been trained/uploaded here yet. Once trained, this folder
will hold the model weights, tokenizer files, and training analysis
artifacts saved locally to
`outputs/model_bank/{dataset_dir_name(dataset_name)}/{lambda_dir_name(lambda_value)}/`.
"""


def top_level_readme(grid):
    dataset_names = sorted({d for d, _ in grid})
    lambdas = sorted({l for _, l in grid})
    rows = "\n".join(
        f"| {d} | {lambda_dir_name(l)} | `{dataset_dir_name(d)}/{lambda_dir_name(l)}/` |"
        for d, l in grid
    )
    return f"""---
license: mit
tags:
  - shiftlab
  - kl-dro
  - robustness
base_model: gpt2-medium
---

# Robustness Model Bank

KL-DRO-trained `gpt2-medium` checkpoints across {len(dataset_names)} source
datasets ({", ".join(dataset_names)}) and {len(lambdas)} robustness
coefficients (lambda in {{{", ".join(f"{l:g}" for l in lambdas)}}}), used by
`diagnostic_experiment/algorithm_2.py` for shift-aware model selection and
interpolation.

**Status: skeleton only.** This repo currently holds placeholder folders and
no trained weights. See `model_bank_metadata.csv` for the full grid and
per-checkpoint status.

## Layout

| Dataset | Lambda | Path |
|---|---|---|
{rows}

Load a specific checkpoint once trained:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "{REPO_ID}",
    subfolder="<dataset>/<lambda_dir>",
)
```

Produced by: `diagnostic_experiment/models_bank.py`, config:
`configs/diagnostic/models_bank.yaml`.
"""


def metadata_csv_stub(grid):
    lines = ["dataset_name,lambda,subfolder,status"]
    for dataset_name, lambda_value in grid:
        subfolder = f"{dataset_dir_name(dataset_name)}/{lambda_dir_name(lambda_value)}"
        lines.append(f"{dataset_name},{lambda_value:g},{subfolder},pending")
    return "\n".join(lines) + "\n"


def build_operations(grid):
    operations = [
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=top_level_readme(grid).encode(),
        ),
        CommitOperationAdd(
            path_in_repo="model_bank_metadata.csv",
            path_or_fileobj=metadata_csv_stub(grid).encode(),
        ),
    ]

    for dataset_name, lambda_value in grid:
        subfolder = f"{dataset_dir_name(dataset_name)}/{lambda_dir_name(lambda_value)}"
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"{subfolder}/README.md",
                path_or_fileobj=checkpoint_placeholder(dataset_name, lambda_value).encode(),
            )
        )

    return operations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    grid = build_grid()
    operations = build_operations(grid)

    print(f"Repo: {args.repo_id}")
    print(f"Grid: {len(grid)} checkpoints ({len(set(d for d, _ in grid))} datasets x {len(set(l for _, l in grid))} lambdas)")
    print(f"Files to create: {len(operations)} (1 README + 1 metadata CSV + {len(grid)} placeholder READMEs)")

    if args.dry_run:
        print("\n--dry-run: no changes made. Paths that would be created:")
        for op in operations:
            print(" -", op.path_in_repo)
        return

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=True, exist_ok=True)

    api.create_commit(
        repo_id=args.repo_id,
        operations=operations,
        commit_message="Add empty model-bank skeleton (datasets x lambda grid)",
    )
    print(f"\nDone. View at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
