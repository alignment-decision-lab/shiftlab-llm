"""One-time restructure of the model-bank HF repo: move all existing
gpt2-medium content under a gpt2Medium/ prefix, and add four empty
placeholder folders (gpt2Tiny, gpt2Small, gpt2Large, gpt2Xlarge) for future
model sizes.

Move, not copy: every existing file is relocated (CommitOperationCopy +
CommitOperationDelete of the old path), so nothing ends up duplicated.
LFS files (the actual checkpoint weights) are copied server-side by the
Hub -- this script never downloads or re-uploads the 32GB of model weights.

Also refreshes the two files that describe the bank, since both were stale
relative to what's actually in the repo:
  - README.md said "gpt2-Tiny" and "no trained weights"; the real
    checkpoints are gpt2-medium and 23/27 are actually trained.
  - model_bank_metadata.csv's status column said "pending" for everything.

Usage:
    python scripts/restructure_hf_model_bank_by_size.py --dry-run
    python scripts/restructure_hf_model_bank_by_size.py
"""
import argparse

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi

REPO_ID = "alignment-decision-lab/robustness-model-bank"
MEDIUM_PREFIX = "gpt2Medium"
NEW_EMPTY_SIZE_FOLDERS = ["gpt2Tiny", "gpt2Small", "gpt2Large", "gpt2Xlarge"]

# Known from the real repo listing: these 4 slots are still placeholders,
# everything else under these 3 sources is an actual trained checkpoint.
PENDING_SLOTS = {
    "FreeLaw/lambda_0.3", "FreeLaw/lambda_0.4",
    "PubMed_Central/lambda_0.3", "PubMed_Central/lambda_0.4",
}

DATASETS = ["FreeLaw", "PubMed_Central", "ArXiv"]
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def lambda_dir_name(lam):
    return f"lambda_{lam:g}"


def build_readme():
    rows = []
    for dataset in DATASETS:
        for lam in LAMBDAS:
            slot = f"{dataset}/{lambda_dir_name(lam)}"
            status = "pending" if slot in PENDING_SLOTS else "trained"
            display_dataset = dataset.replace("_", " ")
            rows.append(f"| {display_dataset} | {lambda_dir_name(lam)} | `{MEDIUM_PREFIX}/{slot}/` | {status} |")

    trained_count = len(DATASETS) * len(LAMBDAS) - len(PENDING_SLOTS)
    total_count = len(DATASETS) * len(LAMBDAS)

    return f"""---
license: mit
tags:
  - shiftlab
  - kl-dro
  - robustness
base_model: gpt2-medium
---

# Robustness Model Bank

KL-DRO-trained checkpoints across model sizes, source datasets, and
robustness coefficients, used by `diagnostic_experiment/algorithm_2.py` and
`diagnostic_experiment/hierarchical_routing.py` for shift-aware model
selection and interpolation.

**Status: {trained_count}/{total_count} gpt2-medium checkpoints trained.** See
`model_bank_metadata.csv` for the full grid and per-checkpoint status.
Other model sizes (`gpt2Tiny/`, `gpt2Small/`, `gpt2Large/`, `gpt2Xlarge/`)
are placeholders -- no checkpoints trained yet.

## Layout

```
{MEDIUM_PREFIX}/          <- gpt2-medium, {trained_count}/{total_count} trained
gpt2Tiny/            <- not started
gpt2Small/           <- not started
gpt2Large/           <- not started
gpt2Xlarge/          <- not started
```

### gpt2-medium

| Dataset | Lambda | Path | Status |
|---|---|---|---|
{chr(10).join(rows)}

Load a specific checkpoint:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "{REPO_ID}",
    subfolder="{MEDIUM_PREFIX}/<dataset>/<lambda_dir>",
)
```

Produced by: `diagnostic_experiment/models_bank.py`, config:
`configs/diagnostic/models_bank.yaml`.
"""


def build_metadata_csv():
    lines = ["dataset_name,lambda,subfolder,status"]
    for dataset in DATASETS:
        for lam in LAMBDAS:
            slot = f"{dataset}/{lambda_dir_name(lam)}"
            status = "pending" if slot in PENDING_SLOTS else "trained"
            display_dataset = dataset.replace("_", " ")
            lines.append(f"{display_dataset},{lam:g},{MEDIUM_PREFIX}/{slot},{status}")
    return "\n".join(lines) + "\n"


def empty_size_placeholder(size_name):
    return f"""# {size_name}

No checkpoints trained yet for this model size. See the repo root README
for the current layout and status across all sizes.
"""


def build_operations(existing_files):
    operations = []

    for f in existing_files:
        if f in (".gitattributes",):
            continue
        if f in ("README.md", "model_bank_metadata.csv"):
            continue  # rewritten below, not moved
        new_path = f"{MEDIUM_PREFIX}/{f}"
        operations.append(CommitOperationCopy(src_path_in_repo=f, path_in_repo=new_path))
        operations.append(CommitOperationDelete(path_in_repo=f))

    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=build_readme().encode()))
    operations.append(CommitOperationAdd(path_in_repo="model_bank_metadata.csv", path_or_fileobj=build_metadata_csv().encode()))

    for size_name in NEW_EMPTY_SIZE_FOLDERS:
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"{size_name}/README.md",
                path_or_fileobj=empty_size_placeholder(size_name).encode(),
            )
        )

    return operations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    existing_files = api.list_repo_files(args.repo_id)
    operations = build_operations(existing_files)

    num_moves = sum(1 for op in operations if isinstance(op, CommitOperationCopy))
    print(f"Repo: {args.repo_id}")
    print(f"Files to move into {MEDIUM_PREFIX}/: {num_moves}")
    print("Rewriting: README.md, model_bank_metadata.csv")
    print(f"New empty placeholder folders: {NEW_EMPTY_SIZE_FOLDERS}")

    if args.dry_run:
        print(f"\n--dry-run: no changes made. {len(operations)} operations would run in one commit:")
        for op in operations:
            if isinstance(op, CommitOperationCopy):
                print(f"  MOVE   {op.src_path_in_repo}  ->  {op.path_in_repo}")
            elif isinstance(op, CommitOperationDelete):
                pass  # paired with the MOVE line above, don't print twice
            elif isinstance(op, CommitOperationAdd):
                print(f"  WRITE  {op.path_in_repo}")
        return

    api.create_commit(
        repo_id=args.repo_id,
        operations=operations,
        commit_message=f"Restructure: move gpt2-medium bank under {MEDIUM_PREFIX}/, add empty size folders",
    )
    print(f"\nDone. View at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
