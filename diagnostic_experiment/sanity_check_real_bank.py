"""Real, CPU-only sanity check: one arriving batch, routed through
Hierarchical, Hard, and Flat routing against real checkpoints downloaded
from alignment-decision-lab/robustness-model-bank.

Unlike test_hierarchical_routing_synthetic.py (tiny random models, checks
mechanics only), this uses real trained gpt2-medium checkpoints and a real
tokenized batch -- it's meant to answer "does this actually work end to
end," not to be a fast repeatable unit test. Kept deliberately small (2
sources x 2 lambdas) to stay CPU-feasible: downloads ~5 real ~1.6GB
checkpoints once (cached after), then a few short forward/backward passes.

Run with: python sanity_check_real_bank.py
"""
import argparse
import subprocess
import sys
import time

import pandas as pd
import torch
from transformers import AutoTokenizer

import hierarchical_routing as hr
import routing_baselines as rb

BANK_REPO_ID = "alignment-decision-lab/robustness-model-bank"
PRETRAINED_MODEL_NAME = "gpt2-medium"

# Kept to 2 rows (one lambda per source) on purpose: this machine has ~9GB
# free RAM, and Flat/Hard routing hold every candidate's full gpt2-medium
# state dict (~1.4GB each in fp32) concurrently during interpolation. 2
# sources is still enough to exercise Hierarchical's top-H source selection
# meaningfully -- a bigger sweep belongs on a GPU box, not this sanity check.
BANK_ROWS = [
    {"dataset_name": "ArXiv", "lambda": 0.0, "subfolder": "gpt2Medium/ArXiv/lambda_0", "status": "trained"},
    {"dataset_name": "FreeLaw", "lambda": 0.0, "subfolder": "gpt2Medium/FreeLaw/lambda_0", "status": "trained"},
]

# Reduced optimization budget -- this is a sanity check, not a real
# experiment; enough iterations to exercise the loop, not to converge well.
HIERARCHICAL_CONFIG = {
    "H": 2,
    "num_iters": 5,
    "lr": 0.1,
    "num_random_starts": 1,
}


def make_real_batch(tokenizer, device):
    texts = [
        "The proof proceeds by induction on the number of variables.",
        "The court held that the defendant's motion was without merit.",
    ]
    tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    encoded["labels"] = encoded["input_ids"].clone()
    return {k: v.to(device) for k, v in encoded.items()}


def make_bank_metadata_path():
    bank_df = pd.DataFrame(BANK_ROWS)
    bank_metadata_path = "/tmp/sanity_check_bank_metadata.csv"
    bank_df.to_csv(bank_metadata_path, index=False)
    return bank_metadata_path


def run_hard(device, batch, bank_metadata_path):
    print("=" * 60)
    print("HARD ROUTING")
    print("=" * 60)
    t0 = time.time()
    theta_B, info = rb.run_hard_routing(bank_metadata_path, batch, PRETRAINED_MODEL_NAME, device, bank_repo_id=BANK_REPO_ID)
    print(f"Selected: {info['selected']}  (loss={info['batch_loss']:.4f})")
    print(f"All losses: {info['all_losses']}")
    print(f"[{time.time()-t0:.1f}s]")


def run_flat(device, batch, bank_metadata_path):
    print("=" * 60)
    print("FLAT ROUTING")
    print("=" * 60)
    t0 = time.time()
    theta_B, info = rb.run_flat_routing(bank_metadata_path, batch, PRETRAINED_MODEL_NAME, device, tau=1.0, bank_repo_id=BANK_REPO_ID)
    print(f"Weights: { {k: round(v, 4) for k, v in info['weights'].items()} }")
    print(f"[{time.time()-t0:.1f}s]")


def run_hierarchical(device, batch, bank_metadata_path):
    print("=" * 60)
    print("HIERARCHICAL ROUTING")
    print("=" * 60)
    t0 = time.time()
    theta_B, info = hr.run_hierarchical_routing(
        bank_metadata_path, batch, PRETRAINED_MODEL_NAME, device,
        config=HIERARCHICAL_CONFIG, bank_repo_id=BANK_REPO_ID,
    )
    print(f"Selected sources: {info['selected_sources']}")
    print(f"Weights: { {k: round(v, 4) for k, v in info['weights'].items()} }")
    print(f"Routed batch loss: {info['batch_loss']:.4f}")
    print(f"Vertex losses: { {k: round(v, 4) for k, v in info['vertex_losses'].items()} }")
    print(f"[{time.time()-t0:.1f}s]")

    # theta_B is a real, usable model -- prove it with one more forward pass.
    theta_B.eval()
    with torch.no_grad():
        out = theta_B(**batch)
    print(f"Composed theta_B forward pass loss: {out.loss.item():.4f} (sanity: should be finite and positive)")


STRATEGIES = {"hard": run_hard, "flat": run_flat, "hierarchical": run_hierarchical}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGIES) + ["all"], default="all")
    args = parser.parse_args()

    if args.strategy == "all":
        # Run each strategy as its own subprocess. On this machine (~9GB
        # free RAM) holding all three routing strategies' loaded
        # gpt2-medium candidates alive in one process risked an OOM kill
        # (exit 137, which took the whole desktop session down with it).
        # A fresh subprocess per strategy guarantees the OS reclaims every
        # byte between them instead of memory accumulating across calls.
        for name in STRATEGIES:
            print(f"\n### launching subprocess: --strategy {name} ###")
            result = subprocess.run([sys.executable, __file__, "--strategy", name])
            if result.returncode != 0:
                print(f"### subprocess --strategy {name} FAILED (exit {result.returncode}) ###")
                sys.exit(result.returncode)
        print("\nAll three routing strategies ran successfully on a real batch against the real bank.")
        return

    device = torch.device("cpu")
    bank_metadata_path = make_bank_metadata_path()
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    batch = make_real_batch(tokenizer, device)
    STRATEGIES[args.strategy](device, batch, bank_metadata_path)


if __name__ == "__main__":
    main()
