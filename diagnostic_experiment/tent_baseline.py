"""CE-based, LayerNorm-affine-only online adaptation baseline.

GPT-2 analog of TENT's mechanism, using the causal-LM cross-entropy
loss instead of prediction entropy.

The incoming batch is split along the token dimension into an
adaptation prefix and a disjoint held-out evaluation suffix.
"""

import torch
from transformers import AutoModelForCausalLM


DEFAULT_CONFIG = {
    "lr": 1e-3,
    "num_steps": 1,
    "adapt_fraction": 0.5,
}


def configure_ln_affine(model):
    """Freeze everything except LayerNorm affine parameters."""
    for param in model.parameters():
        param.requires_grad = False

    ln_params = []
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            if module.weight is not None:
                module.weight.requires_grad = True
                ln_params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad = True
                ln_params.append(module.bias)

    if not ln_params:
        raise ValueError("No LayerNorm affine parameters found.")

    return ln_params


def split_batch_for_holdout(batch, adapt_fraction=0.5):
    """Split each sequence into an adaptation prefix and held-out suffix."""
    if not 0.0 < adapt_fraction < 1.0:
        raise ValueError("adapt_fraction must be strictly between 0 and 1.")

    seq_len = batch["input_ids"].shape[1]
    if seq_len < 2:
        raise ValueError("Sequence length must be at least 2.")

    split_point = max(1, min(seq_len - 1, int(seq_len * adapt_fraction)))
    batch_adapt = {k: v[:, :split_point] for k, v in batch.items()}
    batch_eval = {k: v[:, split_point:] for k, v in batch.items()}

    return batch_adapt, batch_eval


def _compute_loss(model, batch):
    with torch.no_grad():
        return float(model(**batch).loss.item())


def load_base_model(base_model_name_or_path, device, bank_repo_id=None):
    """Load Tent's starting model locally or from a HF bank subfolder."""
    if bank_repo_id is not None:
        model = AutoModelForCausalLM.from_pretrained(
            bank_repo_id,
            subfolder=base_model_name_or_path,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path
        )

    model = model.to(device)
    model.eval()
    return model


def run_tent_adaptation(base_model_name_or_path, batch, device, config=None, canary_batch=None, bank_repo_id=None):
    """Adapt LayerNorm affine parameters and evaluate on held-out tokens."""
    config = {**DEFAULT_CONFIG, **(config or {})}

    batch = {k: v.to(device) for k, v in batch.items()}
    if canary_batch is not None:
        canary_batch = {k: v.to(device) for k, v in canary_batch.items()}

    model = load_base_model(
        base_model_name_or_path=base_model_name_or_path,
        device=device,
        bank_repo_id=bank_repo_id,
    )

    ln_params = configure_ln_affine(model)

    batch_adapt, batch_eval = split_batch_for_holdout(
        batch,
        config["adapt_fraction"],
    )

    loss_adapt_before = _compute_loss(model, batch_adapt)
    loss_eval_before = _compute_loss(model, batch_eval)

    canary_loss_before = (
        _compute_loss(model, canary_batch)
        if canary_batch is not None else None
    )

    optimizer = torch.optim.Adam(
        ln_params,
        lr=config["lr"],
    )

    for _ in range(config["num_steps"]):
        optimizer.zero_grad()
        loss = model(**batch_adapt).loss
        loss.backward()
        optimizer.step()

    loss_adapt_after = _compute_loss(model, batch_adapt)
    loss_eval_after = _compute_loss(model, batch_eval)

    canary_loss_after = (
        _compute_loss(model, canary_batch)
        if canary_batch is not None else None
    )

    info = {
        "loss_adapt_before": loss_adapt_before,
        "loss_adapt_after": loss_adapt_after,
        "loss_eval_before": loss_eval_before,
        "loss_eval_after": loss_eval_after,
        "canary_loss_before": canary_loss_before,
        "canary_loss_after": canary_loss_after,
    }

    return model, info