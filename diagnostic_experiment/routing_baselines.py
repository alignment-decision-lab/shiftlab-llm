"""Hard Routing and Flat Soft Routing deployment-time baselines.

Hard Routing:
    theta_B = argmin_{theta in M} L_hat_B(theta),
where M = {theta_0} U {theta_{j,lambda}}.

Flat Routing:
    1. Select the best lambda for each source on the incoming batch.
    2. Add theta_0 and retain the Top-H representatives.
    3. Apply closed-form softmax weighting:
           w_k(B) = softmax(-L_hat_B(theta_k) / tau)
    4. Compose:
           theta_B = sum_k w_k(B) theta_k.

Both methods can reuse bank losses and the pretrained loss computed once
per incoming batch.
"""

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from algorithm_2 import load_model_bank_metadata
from hierarchical_routing import (
    build_candidate_state_dicts,
    compose_model,
    compute_bank_batch_losses,
    compute_source_relevance,
    filter_trained_rows,
    load_bank_checkpoint,
    select_routing_candidates,
)
from utils import move_batch_to_device


# ============================================================
# PRETRAINED MODEL
# ============================================================

def load_pretrained_model(pretrained_model_name, device, pretrained_subfolder=None, bank_repo_id=None):
    """Load theta_0 from the model bank when available, otherwise from pretrained_model_name."""
    if pretrained_subfolder is not None:
        if bank_repo_id is None:
            raise ValueError("bank_repo_id is required when pretrained_subfolder is provided.")
        model = AutoModelForCausalLM.from_pretrained(bank_repo_id, subfolder=pretrained_subfolder).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name).to(device)

    model.eval()
    return model


# ============================================================
# SHARED SCORING
# ============================================================

def compute_pretrained_loss(pretrained_model_name, batch, device, pretrained_subfolder=None, bank_repo_id=None):
    """Compute L_hat_B(theta_0)."""
    model = load_pretrained_model(pretrained_model_name, device, pretrained_subfolder, bank_repo_id)

    with torch.no_grad():
        loss = model(**batch).loss.item()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return loss


def _bank_candidate_name(row):
    return f"{row['dataset_name']}_lambda_{row['lambda']:g}"


def _all_bank_losses(bank_df, pretrained_model_name, batch, device, bank_repo_id=None, scored_bank_df=None, pretrained_loss=None, pretrained_subfolder=None):
    """Return losses for the complete bank plus theta_0."""
    if scored_bank_df is None:
        scored_bank_df = compute_bank_batch_losses(bank_df, batch, device, bank_repo_id)

    names = [_bank_candidate_name(row) for _, row in scored_bank_df.iterrows()]
    losses = scored_bank_df["batch_loss"].astype(float).tolist()

    if pretrained_loss is None:
        pretrained_loss = compute_pretrained_loss(
            pretrained_model_name=pretrained_model_name,
            batch=batch,
            device=device,
            pretrained_subfolder=pretrained_subfolder,
            bank_repo_id=bank_repo_id,
        )

    names.append("theta_0")
    losses.append(float(pretrained_loss))
    return names, losses


# ============================================================
# HARD ROUTING
# ============================================================

def run_hard_routing(model_bank_metadata_path, batch, pretrained_model_name, device, bank_repo_id=None, bank_df=None, scored_bank_df=None, pretrained_loss=None, pretrained_subfolder=None):
    """Select the single best checkpoint from theta_0 and the complete bank."""
    batch = move_batch_to_device(batch, device)

    if bank_df is None:
        bank_df = filter_trained_rows(load_model_bank_metadata(model_bank_metadata_path))

    names, losses = _all_bank_losses(
        bank_df=bank_df,
        pretrained_model_name=pretrained_model_name,
        batch=batch,
        device=device,
        bank_repo_id=bank_repo_id,
        scored_bank_df=scored_bank_df,
        pretrained_loss=pretrained_loss,
        pretrained_subfolder=pretrained_subfolder,
    )

    best_idx = min(range(len(losses)), key=lambda i: losses[i])
    best_name, best_loss = names[best_idx], losses[best_idx]

    if best_name == "theta_0":
        theta_B = load_pretrained_model(
            pretrained_model_name=pretrained_model_name,
            device=device,
            pretrained_subfolder=pretrained_subfolder,
            bank_repo_id=bank_repo_id,
        )
    else:
        source_df = scored_bank_df if scored_bank_df is not None else bank_df
        theta_B = load_bank_checkpoint(source_df.iloc[best_idx], bank_repo_id).to(device)

    theta_B.eval()

    info = {
        "selected": best_name,
        "batch_loss": float(best_loss),
        "all_losses": dict(zip(names, losses)),
    }

    return theta_B, info


# ============================================================
# FLAT SOFT ROUTING
# ============================================================

def run_flat_routing(model_bank_metadata_path, batch, pretrained_model_name, device, tau=1.0, H=3, bank_repo_id=None, bank_df=None, scored_bank_df=None, pretrained_loss=None, selected_candidates_df=None, pretrained_subfolder=None):
    """Pre-select Top-H representatives, then apply softmax weighting."""
    batch = move_batch_to_device(batch, device)

    if bank_df is None:
        bank_df = filter_trained_rows(load_model_bank_metadata(model_bank_metadata_path))

    if scored_bank_df is None:
        raise ValueError("Flat Routing requires precomputed bank losses through scored_bank_df.")

    if pretrained_loss is None:
        raise ValueError("Flat Routing requires the precomputed theta_0 loss through pretrained_loss.")

    source_relevance_df = compute_source_relevance(scored_bank_df)

    if selected_candidates_df is None:
        representatives_df, selected_candidates_df = select_routing_candidates(source_relevance_df, pretrained_loss, H)
    else:
        selected_candidates_df = selected_candidates_df.copy()
        representatives_df, _ = select_routing_candidates(source_relevance_df, pretrained_loss, H)

    names = selected_candidates_df["candidate_name"].tolist()
    losses = selected_candidates_df["best_loss"].astype(float).tolist()
    weights = F.softmax(-torch.tensor(losses, dtype=torch.float32) / tau, dim=0)

    candidate_state_dicts = build_candidate_state_dicts(
        selected_candidates_df=selected_candidates_df,
        pretrained_model_name=pretrained_model_name,
        device=device,
        bank_repo_id=bank_repo_id,
        pretrained_subfolder=pretrained_subfolder,
    )

    model_template = load_pretrained_model(
        pretrained_model_name=pretrained_model_name,
        device=device,
        pretrained_subfolder=pretrained_subfolder,
        bank_repo_id=bank_repo_id,
    )

    theta_B = compose_model(model_template, candidate_state_dicts, weights.to(device), device)
    theta_B.eval()

    selected_ft_df = selected_candidates_df[selected_candidates_df["is_pretrained"] == False]

    info = {
        "candidate_names": names,
        "weights": dict(zip(names, weights.tolist())),
        "candidate_losses": dict(zip(names, losses)),
        "pretrained_loss": float(pretrained_loss),
        "source_relevance": source_relevance_df.to_dict(orient="records"),
        "representatives": representatives_df.to_dict(orient="records"),
        "selected_sources": selected_ft_df["dataset_name"].tolist(),
        "selected_lambdas": {
            row["dataset_name"]: float(row["lambda_star"])
            for _, row in selected_ft_df.iterrows()
        },
        "selected_candidates": selected_candidates_df.to_dict(orient="records"),
        "tau": tau,
        "H": H,
    }

    del model_template
    del candidate_state_dicts

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return theta_B, info