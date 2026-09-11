"""CE-based, LayerNorm-affine-only online test-time adaptation.

GPT-2 analog of TENT's mechanism, using the causal-LM cross-entropy
loss instead of prediction entropy.

The model is initialized once from the Mixed-FT checkpoint. For each
incoming deployment batch, the full batch is used for both adaptation
and evaluation: B_t -> theta_t -> L_{B_t}(theta_t). Unlike episodic
Tent, the adapted LayerNorm parameters and optimizer state are retained
and become the starting point for the next batch.

In the Primary Episodic protocol, a new online state must be created
at the beginning of each deployment dataset. In the Non-stationary
Stream protocol, the same state is kept across domain switches.
"""

import torch
from transformers import AutoModelForCausalLM

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
    return {k: v.to(device) for k, v in batch.items()}


def _compute_loss(model, batch):
    model.eval()
    with torch.no_grad():
        return float(model(**batch).loss.item())


def load_base_model(base_model_name_or_path, device, bank_repo_id=None):
    """Load Tent's starting model locally/publicly or from a HF bank subfolder."""
    if bank_repo_id is not None:
        model = AutoModelForCausalLM.from_pretrained(bank_repo_id, subfolder=base_model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)

    model = model.to(device)
    model.eval()
    return model


def initialize_online_tent(base_model_name_or_path, device, config=None, bank_repo_id=None):
    """Initialize the persistent online Tent model and optimizer."""
    config = {**DEFAULT_CONFIG, **(config or {})}

    if config["num_steps"] <= 0:
        raise ValueError("num_steps must be > 0.")

    model = load_base_model(
        base_model_name_or_path=base_model_name_or_path,
        device=device,
        bank_repo_id=bank_repo_id,
    )
    ln_params = configure_ln_affine(model)
    optimizer = torch.optim.Adam(ln_params, lr=config["lr"])

    return {
        "model": model,
        "optimizer": optimizer,
        "ln_params": ln_params,
        "config": config,
        "num_batches_seen": 0,
    }


def run_online_tent_batch(state, batch, device, canary_batch=None):
    """Adapt the persistent online model on one full batch and evaluate on that same batch."""
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
        optimizer.zero_grad()
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