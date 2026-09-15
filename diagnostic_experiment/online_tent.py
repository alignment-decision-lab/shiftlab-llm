"""CE-based, LayerNorm-affine-only online test-time adaptation for causal LMs.

This is a GPT-2 adaptation of TENT's online mechanism. Instead of prediction
entropy, it minimizes the causal-LM next-token cross-entropy because labels
are directly available from the input sequence.

The starting model is provided by the experimental pipeline rather than
loaded here. This allows the same TENT implementation to be initialized from:
    - Best Single-FT;
    - Hierarchical Routing.

TENT is online: after adapting on batch B_t, the adapted LayerNorm parameters
and optimizer state are retained for B_{t+1}. With num_steps=1, one gradient
update is performed per incoming batch.

Static TENT is handled by the experimental pipeline: TENT is applied on the
first batch and the resulting model is then frozen for subsequent batches.
"""

import torch


DEFAULT_CONFIG = {
    "lr": 1e-3,
    "num_steps": 1,
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


def _move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _compute_loss(model, batch):
    model.eval()
    with torch.no_grad():
        return float(model(**batch).loss.item())


def initialize_online_tent(model, device, config=None):
    """Initialize persistent online TENT from an already constructed model."""
    config = {**DEFAULT_CONFIG, **(config or {})}

    if config["num_steps"] <= 0:
        raise ValueError("num_steps must be > 0.")
    if config["lr"] <= 0:
        raise ValueError("lr must be > 0.")

    model = model.to(device)
    model.eval()

    ln_params = configure_ln_affine(model)
    optimizer = torch.optim.Adam(ln_params, lr=float(config["lr"]))

    return {
        "model": model,
        "optimizer": optimizer,
        "ln_params": ln_params,
        "config": config,
        "num_batches_seen": 0,
    }


def run_online_tent_batch(state, batch, device, canary_batch=None):
    """Adapt persistent TENT state on one full batch and evaluate on that batch."""
    model = state["model"]
    optimizer = state["optimizer"]
    ln_params = state["ln_params"]
    config = state["config"]

    batch = _move_batch_to_device(batch, device)
    if canary_batch is not None:
        canary_batch = _move_batch_to_device(canary_batch, device)

    loss_before = _compute_loss(model, batch)
    canary_loss_before = _compute_loss(model, canary_batch) if canary_batch is not None else None

    model.train()
    for _ in range(config["num_steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()

    loss_after = _compute_loss(model, batch)
    canary_loss_after = _compute_loss(model, canary_batch) if canary_batch is not None else None
    state["num_batches_seen"] += 1

    info = {
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_improvement": loss_before - loss_after,
        "canary_loss_before": canary_loss_before,
        "canary_loss_after": canary_loss_after,
        "num_trainable_params": sum(param.numel() for param in ln_params),
        "num_adaptation_steps": config["num_steps"],
        "adaptation_lr": config["lr"],
        "num_batches_seen": state["num_batches_seen"],
    }

    return state, info