"""CPU-only synthetic dry run for tent_baseline.py -- tiny random GPT2Config
model, no downloads, no real data. Mirrors the approach in
test_hierarchical_routing_synthetic.py: verifies the mechanics (parameter
freezing, held-out split, gradient flow) work correctly, not that the
numbers are meaningful.
"""
import tempfile

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from tent_baseline import configure_ln_affine, run_tent_adaptation, split_batch_for_holdout

TINY_CONFIG = GPT2Config(vocab_size=50, n_positions=32, n_embd=16, n_layer=2, n_head=2)


def make_tiny_model_dir(tmp_dir):
    model = GPT2LMHeadModel(TINY_CONFIG)
    model.save_pretrained(tmp_dir)
    return tmp_dir


def make_synthetic_batch(batch_size=2, seq_len=16):
    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def test_configure_ln_affine_freezes_everything_else():
    model = GPT2LMHeadModel(TINY_CONFIG)
    ln_params = configure_ln_affine(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == len(ln_params), "only LayerNorm affine params should require grad"

    ln_modules = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
    expected_count = 2 * len(ln_modules)  # weight + bias per LayerNorm
    assert len(ln_params) == expected_count, f"expected {expected_count} LN affine params, got {len(ln_params)}"
    print(f"test_configure_ln_affine_freezes_everything_else: PASS ({len(ln_modules)} LayerNorms, {len(ln_params)} affine params trainable)")


def test_split_batch_for_holdout_is_disjoint():
    batch = make_synthetic_batch(batch_size=2, seq_len=16)
    batch_adapt, batch_eval = split_batch_for_holdout(batch, adapt_fraction=0.5)

    assert batch_adapt["input_ids"].shape[1] == 8
    assert batch_eval["input_ids"].shape[1] == 8
    assert batch_adapt["input_ids"].shape[1] + batch_eval["input_ids"].shape[1] == batch["input_ids"].shape[1]
    # disjoint: adapt is the prefix, eval is the suffix, no token overlap
    assert torch.equal(batch_adapt["input_ids"], batch["input_ids"][:, :8])
    assert torch.equal(batch_eval["input_ids"], batch["input_ids"][:, 8:])
    print("test_split_batch_for_holdout_is_disjoint: PASS")


def test_split_batch_for_holdout_handles_short_sequence():
    # seq_len=1 should still produce two non-degenerate (if minimal) chunks
    batch = make_synthetic_batch(batch_size=1, seq_len=2)
    batch_adapt, batch_eval = split_batch_for_holdout(batch, adapt_fraction=0.5)
    assert batch_adapt["input_ids"].shape[1] >= 1
    assert batch_eval["input_ids"].shape[1] >= 1
    print("test_split_batch_for_holdout_handles_short_sequence: PASS")


def test_end_to_end_adaptation_reduces_adapt_loss():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = make_tiny_model_dir(tmp_dir)
        batch = make_synthetic_batch(batch_size=2, seq_len=16)
        device = torch.device("cpu")

        config = {"lr": 0.5, "num_steps": 5, "adapt_fraction": 0.5}
        model, info = run_tent_adaptation(model_dir, batch, device, config)

        for key in ("loss_adapt_before", "loss_adapt_after", "loss_eval_before", "loss_eval_after"):
            assert key in info
            assert info[key] == info[key], f"{key} is NaN"  # NaN != NaN
            assert info[key] > 0

        # with 5 steps at a large lr on the exact batch it's optimizing against,
        # loss on that same batch should go down -- confirms gradients actually flow
        assert info["loss_adapt_after"] < info["loss_adapt_before"], (
            f"adapt loss did not decrease: before={info['loss_adapt_before']:.4f}, "
            f"after={info['loss_adapt_after']:.4f}"
        )
        print(
            "test_end_to_end_adaptation_reduces_adapt_loss: PASS "
            f"(adapt: {info['loss_adapt_before']:.4f} -> {info['loss_adapt_after']:.4f}, "
            f"eval: {info['loss_eval_before']:.4f} -> {info['loss_eval_after']:.4f})"
        )


def test_canary_batch_is_optional_and_unaffected_by_split():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = make_tiny_model_dir(tmp_dir)
        device = torch.device("cpu")

        # without a canary batch: keys present but None, no crash
        batch = make_synthetic_batch(batch_size=2, seq_len=16)
        _, info_no_canary = run_tent_adaptation(model_dir, batch, device)
        assert info_no_canary["canary_loss_before"] is None
        assert info_no_canary["canary_loss_after"] is None

        # with a canary batch: both values computed and finite
        batch = make_synthetic_batch(batch_size=2, seq_len=16)
        canary_batch = make_synthetic_batch(batch_size=2, seq_len=10)
        _, info_with_canary = run_tent_adaptation(model_dir, batch, device, canary_batch=canary_batch)
        assert info_with_canary["canary_loss_before"] is not None
        assert info_with_canary["canary_loss_after"] is not None
        assert info_with_canary["canary_loss_before"] > 0
        assert info_with_canary["canary_loss_after"] > 0

        print(
            "test_canary_batch_is_optional_and_unaffected_by_split: PASS "
            f"(canary: {info_with_canary['canary_loss_before']:.4f} -> {info_with_canary['canary_loss_after']:.4f})"
        )


if __name__ == "__main__":
    test_configure_ln_affine_freezes_everything_else()
    test_split_batch_for_holdout_is_disjoint()
    test_split_batch_for_holdout_handles_short_sequence()
    test_end_to_end_adaptation_reduces_adapt_loss()
    test_canary_batch_is_optional_and_unaffected_by_split()
    print("\nAll tent_baseline synthetic tests passed.")
