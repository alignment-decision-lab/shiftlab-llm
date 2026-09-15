"""Synthetic, CPU-only sanity check for hierarchical_routing.py.

Builds a tiny random GPT-2 config (small enough to run in seconds on CPU,
no download needed) and a fake model bank of a few such checkpoints, then
runs the real run_hierarchical_routing() entry point end to end. This is a
mechanics check -- shapes, device handling, the exponentiated-gradient loop,
functional_call -- not a check of anything about routing *quality*, since
the "bank" here is just random weights with no real structure to select
between.

Run with: python test_hierarchical_routing_synthetic.py
"""
import shutil
import tempfile

import pandas as pd
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import hierarchical_routing as hr
import routing_baselines as rb


TINY_CONFIG = GPT2Config(
    vocab_size=50,
    n_positions=32,
    n_embd=16,
    n_layer=2,
    n_head=2,
)


def make_tiny_checkpoint(save_dir, seed):
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(TINY_CONFIG)
    model.save_pretrained(save_dir)
    return save_dir


def make_synthetic_batch(batch_size=4, seq_len=8, vocab_size=50, seed=0):
    torch.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def test_exponentiated_gradient_step():
    weights = torch.tensor([0.5, 0.5])
    grad = torch.tensor([1.0, -1.0])
    lr = 1.0

    updated = hr.exponentiated_gradient_step(weights, grad, lr)

    assert torch.isclose(updated.sum(), torch.tensor(1.0), atol=1e-6), "weights must stay on the simplex"
    assert updated[1] > updated[0], "the candidate with the more negative gradient should gain weight"
    print("test_exponentiated_gradient_step: OK", updated.tolist())


def test_interpolate_state_dicts():
    candidates = {
        "a": {"w": torch.tensor([1.0, 0.0])},
        "b": {"w": torch.tensor([0.0, 1.0])},
    }
    weights = torch.tensor([0.25, 0.75])

    result = hr.interpolate_state_dicts(candidates, weights, torch.device("cpu"))

    expected = torch.tensor([0.25, 0.75])
    assert torch.allclose(result["w"], expected), f"expected {expected}, got {result['w']}"
    print("test_interpolate_state_dicts: OK", result["w"].tolist())


def make_synthetic_bank(tmp_dir):
    """Shared by all end-to-end tests: one tiny pretrained checkpoint plus a
    2-source x 3-lambda bank, all random weights."""
    pretrained_dir = make_tiny_checkpoint(f"{tmp_dir}/theta_0", seed=0)

    rows = []
    seed = 1
    for source in ["SourceA", "SourceB"]:
        for lam in [0.1, 0.5, 1.0]:
            save_dir = f"{tmp_dir}/{source}_lambda_{lam}"
            make_tiny_checkpoint(save_dir, seed=seed)
            rows.append({"dataset_name": source, "lambda": lam, "save_dir": save_dir})
            seed += 1

    bank_df = pd.DataFrame(rows)
    bank_metadata_path = f"{tmp_dir}/model_bank_metadata.csv"
    bank_df.to_csv(bank_metadata_path, index=False)

    return pretrained_dir, bank_metadata_path


def test_end_to_end_on_synthetic_bank():
    device = torch.device("cpu")
    tmp_dir = tempfile.mkdtemp(prefix="hr_synthetic_")

    try:
        pretrained_dir, bank_metadata_path = make_synthetic_bank(tmp_dir)
        batch = make_synthetic_batch(vocab_size=TINY_CONFIG.vocab_size)
        bank_df = hr.filter_trained_rows(hr.load_model_bank_metadata(bank_metadata_path))
        scored_bank_df = hr.compute_bank_batch_losses(bank_df, batch, device)
        pretrained_loss = rb.compute_pretrained_loss(
            pretrained_model_name=pretrained_dir, batch=batch, device=device,
        )

        # H=3: theta_0 now competes for a slot against the 2 source
        # representatives (select_routing_candidates), rather than always
        # getting a free slot -- with only 3 representatives total, H=3
        # guarantees all of them survive regardless of who wins, keeping
        # this test deterministic despite theta_0/sources being random
        # synthetic checkpoints.
        config = {
            "H": 3,
            "num_iters": 5,          # tiny, just enough to exercise the loop
            "lr": 0.1,
            "num_random_starts": 2,
        }

        theta_B, info = hr.run_hierarchical_routing(
            model_bank_metadata_path=bank_metadata_path,
            batch=batch,
            pretrained_model_name=pretrained_dir,
            device=device,
            config=config,
            bank_df=bank_df,
            scored_bank_df=scored_bank_df,
            pretrained_loss=pretrained_loss,
        )

        weight_sum = sum(info["weights"].values())
        assert abs(weight_sum - 1.0) < 1e-4, f"routing weights should sum to 1, got {weight_sum}"
        assert len(info["selected_sources"]) == 2, "H=3 with 2 sources + theta_0 should keep both sources"
        assert info["batch_loss"] < float("inf")

        # theta_B should be a real usable model.
        theta_B.eval()
        with torch.no_grad():
            out = theta_B(**batch)
        assert out.loss.item() > 0

        print("test_end_to_end_on_synthetic_bank: OK")
        print("  selected sources:", info["selected_sources"])
        print("  weights:", info["weights"])
        print("  routed batch loss:", info["batch_loss"])
        print("  vertex losses:", info["vertex_losses"])

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_hard_routing_on_synthetic_bank():
    device = torch.device("cpu")
    tmp_dir = tempfile.mkdtemp(prefix="hr_hard_synthetic_")

    try:
        pretrained_dir, bank_metadata_path = make_synthetic_bank(tmp_dir)
        batch = make_synthetic_batch(vocab_size=TINY_CONFIG.vocab_size)

        theta_B, info = rb.run_hard_routing(
            model_bank_metadata_path=bank_metadata_path,
            batch=batch,
            pretrained_model_name=pretrained_dir,
            device=device,
        )

        # 2 sources x 3 lambdas + theta_0 = 7 candidates scored.
        assert len(info["all_losses"]) == 7
        assert info["selected"] in info["all_losses"]
        assert info["batch_loss"] == min(info["all_losses"].values()), "hard routing must pick the global minimum"

        theta_B.eval()
        with torch.no_grad():
            out = theta_B(**batch)
        assert out.loss.item() > 0
        assert abs(out.loss.item() - info["batch_loss"]) < 1e-4, "returned model's loss should match the reported one"

        print("test_hard_routing_on_synthetic_bank: OK")
        print("  selected:", info["selected"])
        print("  batch_loss:", info["batch_loss"])

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_flat_routing_on_synthetic_bank():
    device = torch.device("cpu")
    tmp_dir = tempfile.mkdtemp(prefix="hr_flat_synthetic_")

    try:
        pretrained_dir, bank_metadata_path = make_synthetic_bank(tmp_dir)
        batch = make_synthetic_batch(vocab_size=TINY_CONFIG.vocab_size)
        bank_df = hr.filter_trained_rows(hr.load_model_bank_metadata(bank_metadata_path))
        scored_bank_df = hr.compute_bank_batch_losses(bank_df, batch, device)
        pretrained_loss = rb.compute_pretrained_loss(
            pretrained_model_name=pretrained_dir, batch=batch, device=device,
        )

        theta_B, info = rb.run_flat_routing(
            model_bank_metadata_path=bank_metadata_path,
            batch=batch,
            pretrained_model_name=pretrained_dir,
            device=device,
            tau=1.0,
            H=3,  # 2 sources + theta_0 = 3 representatives total; H=3 keeps all of them
            bank_df=bank_df,
            scored_bank_df=scored_bank_df,
            pretrained_loss=pretrained_loss,
        )

        # Flat Routing is now Top-H screened (over source representatives,
        # not the full lambda grid): 2 sources + theta_0 = 3 candidates,
        # not 7 (2 sources x 3 lambdas + theta_0) as in the old full-bank
        # version.
        assert len(info["weights"]) == 3
        weight_sum = sum(info["weights"].values())
        assert abs(weight_sum - 1.0) < 1e-4, f"flat routing weights should sum to 1, got {weight_sum}"
        assert all(w >= 0 for w in info["weights"].values())

        theta_B.eval()
        with torch.no_grad():
            out = theta_B(**batch)
        assert out.loss.item() > 0

        print("test_flat_routing_on_synthetic_bank: OK")
        print("  weights:", {k: round(v, 4) for k, v in info["weights"].items()})
        print("  routed batch loss:", out.loss.item())

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_exponentiated_gradient_step()
    test_interpolate_state_dicts()
    test_end_to_end_on_synthetic_bank()
    test_hard_routing_on_synthetic_bank()
    test_flat_routing_on_synthetic_bank()
    print("\nAll synthetic checks passed.")
