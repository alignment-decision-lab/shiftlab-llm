"""Hierarchical Routing with source-first ERM screening.

Given an incoming batch B and a model bank
M = {theta_0} U {theta_{j,lambda}}:

1. Score theta_0 and the standard ERM checkpoint theta_{j,0} of every source.
2. Keep the H candidates with the smallest incoming-batch loss among
   {theta_0} U {theta_{j,0}}.
3. For each selected fine-tuned source j, score all its lambda checkpoints:
       lambda*_j(B) = argmin_lambda L_hat_B(theta_{j,lambda}).
   If theta_0 was selected at Step 2, it remains unchanged.
4. Replace each selected source ERM representative by theta_{j,lambda*_j}.
5. Run the exact same simplex EG optimization as hierarchical_routing.py.
6. Compose theta_B = sum_k w*_k theta_k.

Relative to hierarchical_routing.py, only the candidate-selection order changes.
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM

import hierarchical_routing as hr
from algorithm_2 import load_model_bank_metadata
from utils import move_batch_to_device


DEFAULT_CONFIG = {
    "H": 3,
    "num_iters": 100,
    "lr": 0.1,
    "num_random_starts": 5,
    "dirichlet_concentration": 1.0,
    "flat_tau": 1.0,
    "seed": 42,
}


# ============================================================
# MODEL LOADING
# ============================================================

def load_bank_checkpoint(row, bank_repo_id=None):
    """Load one bank checkpoint from Hugging Face or local disk."""
    if bank_repo_id is not None:
        return AutoModelForCausalLM.from_pretrained(bank_repo_id, subfolder=row["subfolder"])
    return AutoModelForCausalLM.from_pretrained(row["save_dir"])


def load_pretrained_checkpoint(pretrained_model_name, device=None, pretrained_subfolder=None, bank_repo_id=None):
    """Load theta_0 from the model bank when configured, otherwise from Hugging Face."""
    if pretrained_subfolder is not None:
        if bank_repo_id is None:
            raise ValueError("bank_repo_id is required when pretrained_subfolder is provided.")
        model = AutoModelForCausalLM.from_pretrained(bank_repo_id, subfolder=pretrained_subfolder)
    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)

    if device is not None:
        model = model.to(device)

    model.eval()
    return model


def filter_trained_rows(bank_df):
    """Drop rows explicitly marked as pending."""
    if "status" not in bank_df.columns:
        return bank_df
    return bank_df[bank_df["status"] != "pending"].reset_index(drop=True)


# ============================================================
# STEP 1-2: ERM SCREENING
# ============================================================

def select_erm_representatives(bank_df):
    """Select the lambda=0.0 ERM checkpoint of every source."""
    rows = []

    for dataset_name, group in bank_df.groupby("dataset_name"):
        lambdas = pd.to_numeric(group["lambda"], errors="coerce")
        matches = group[lambdas.abs() < 1e-12]

        if matches.empty:
            available = sorted(lambdas.dropna().unique().tolist())
            raise ValueError(
                f"Source {dataset_name!r} has no lambda=0.0 ERM checkpoint. "
                f"Available lambdas: {available}"
            )

        row = dict(matches.iloc[0])
        row["dataset_name"] = dataset_name
        rows.append(row)

    return pd.DataFrame(rows)


def compute_checkpoint_batch_losses(bank_df, batch, device, bank_repo_id=None):
    """Compute incoming-batch loss for exactly the checkpoints in bank_df."""
    losses = []

    for _, row in bank_df.iterrows():
        model = load_bank_checkpoint(row, bank_repo_id).to(device)
        model.eval()

        with torch.no_grad():
            loss = model(**batch).loss.item()

        losses.append(float(loss))
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    scored_df = bank_df.copy()
    scored_df["batch_loss"] = losses
    return scored_df


def select_top_h_erm_candidates(scored_erm_df, pretrained_loss, H):
    """Keep Top-H among theta_0 and the lambda=0.0 ERM model of every source."""
    rows = [{
        "candidate_name": "theta_0",
        "dataset_name": "theta_0",
        "lambda": None,
        "batch_loss": float(pretrained_loss),
        "is_pretrained": True,
    }]

    for _, source_row in scored_erm_df.iterrows():
        row = dict(source_row)
        row.update({"candidate_name": source_row["dataset_name"], "is_pretrained": False})
        rows.append(row)

    screening_candidates_df = pd.DataFrame(rows)
    num_candidates = min(H, len(screening_candidates_df))
    selected_screening_df = (
        screening_candidates_df.sort_values("batch_loss")
        .head(num_candidates)
        .reset_index(drop=True)
    )
    return screening_candidates_df, selected_screening_df


# ============================================================
# STEP 3: SELECTED-SOURCE LAMBDA SCORING
# ============================================================

def score_selected_source_banks(bank_df, selected_sources, scored_erm_df, batch, device, bank_repo_id=None):
    """Score all lambdas only for selected sources, reusing their lambda=0 losses."""
    selected_bank_df = bank_df[bank_df["dataset_name"].isin(selected_sources)].copy()
    erm_losses = {
        row["dataset_name"]: float(row["batch_loss"])
        for _, row in scored_erm_df.iterrows()
    }
    losses = []

    for _, row in selected_bank_df.iterrows():
        dataset_name, lambda_value = row["dataset_name"], float(row["lambda"])

        if abs(lambda_value) < 1e-12:
            losses.append(erm_losses[dataset_name])
            continue

        model = load_bank_checkpoint(row, bank_repo_id).to(device)
        model.eval()

        with torch.no_grad():
            loss = model(**batch).loss.item()

        losses.append(float(loss))
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected_bank_df["batch_loss"] = losses
    return selected_bank_df.reset_index(drop=True)


def compute_source_relevance(scored_bank_df):
    """Find lambda*_j(B), best loss and alpha_j(B) for each selected source."""
    rows = []

    for dataset_name, group in scored_bank_df.groupby("dataset_name"):
        best_idx = group["batch_loss"].idxmin()
        best_row = group.loc[best_idx]

        row = dict(best_row)
        row.update({
            "dataset_name": dataset_name,
            "lambda_star": float(best_row["lambda"]),
            "best_loss": float(best_row["batch_loss"]),
            "alpha": -float(best_row["batch_loss"]),
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# STEP 4: FINAL CANDIDATES
# ============================================================

def build_final_candidates(selected_screening_df, source_relevance_df, pretrained_loss):
    """Replace each selected ERM representative by its source-optimal lambda*_j."""
    rows = []

    for _, selected_row in selected_screening_df.iterrows():
        if bool(selected_row["is_pretrained"]):
            rows.append({
                "candidate_name": "theta_0",
                "dataset_name": "theta_0",
                "lambda_star": None,
                "best_loss": float(pretrained_loss),
                "alpha": -float(pretrained_loss),
                "is_pretrained": True,
            })
            continue

        dataset_name = selected_row["dataset_name"]
        source_row = source_relevance_df[
            source_relevance_df["dataset_name"] == dataset_name
        ]

        if source_row.empty:
            raise ValueError(
                f"No lambda scoring result found for selected source {dataset_name!r}."
            )

        row = dict(source_row.iloc[0])
        row.update({"candidate_name": dataset_name, "is_pretrained": False})
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


# ============================================================
# FINAL CANDIDATE STATE DICTS
# ============================================================

def build_candidate_state_dicts(selected_candidates_df, pretrained_model_name, device,
                                bank_repo_id=None, pretrained_subfolder=None):
    """Load only the final H candidates as CPU state dictionaries."""
    candidates = {}

    for _, row in selected_candidates_df.iterrows():
        if bool(row["is_pretrained"]):
            model = load_pretrained_checkpoint(
                pretrained_model_name=pretrained_model_name,
                pretrained_subfolder=pretrained_subfolder,
                bank_repo_id=bank_repo_id,
            )
            candidate_name = "theta_0"
        else:
            model = load_bank_checkpoint(row, bank_repo_id)
            candidate_name = row["dataset_name"]

        candidates[candidate_name] = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        del model

    return candidates


# ============================================================
# HIERARCHICAL ROUTING — NEW ORDER
# ============================================================

def run_hierarchical_routing_new_order(model_bank_metadata_path, batch,
                                       pretrained_model_name, device, config=None,
                                       bank_repo_id=None, bank_df=None,
                                       pretrained_loss=None, pretrained_subfolder=None):
    """Run ERM-screened Hierarchical Routing using the original shared EG engine."""
    config = {**DEFAULT_CONFIG, **(config or {})}
    batch = move_batch_to_device(batch, device)

    if bank_df is None:
        bank_df = load_model_bank_metadata(model_bank_metadata_path)
        bank_df = filter_trained_rows(bank_df)

    if pretrained_loss is None:
        raise ValueError(
            "New-order Hierarchical Routing requires the precomputed "
            "theta_0 batch loss through pretrained_loss."
        )

    # Step 1: score lambda=0.0 for every source; theta_0 is already scored.
    erm_df = select_erm_representatives(bank_df)
    scored_erm_df = compute_checkpoint_batch_losses(
        erm_df, batch, device, bank_repo_id
    )

    # Step 2: Top-H among theta_0 and all source ERM representatives.
    screening_candidates_df, selected_screening_df = select_top_h_erm_candidates(
        scored_erm_df, pretrained_loss, config["H"]
    )
    selected_sources = selected_screening_df.loc[
        selected_screening_df["is_pretrained"] == False, "dataset_name"
    ].tolist()

    # Step 3: score all lambdas only for fine-tuned sources surviving Top-H.
    if selected_sources:
        scored_selected_bank_df = score_selected_source_banks(
            bank_df, selected_sources, scored_erm_df, batch, device, bank_repo_id
        )
        source_relevance_df = compute_source_relevance(scored_selected_bank_df)
    else:
        scored_selected_bank_df = pd.DataFrame()
        source_relevance_df = pd.DataFrame()

    # Step 4: replace each selected ERM representative by theta_{j,lambda*_j}.
    selected_candidates_df = build_final_candidates(
        selected_screening_df, source_relevance_df, pretrained_loss
    )
    candidate_state_dicts = build_candidate_state_dicts(
        selected_candidates_df=selected_candidates_df,
        pretrained_model_name=pretrained_model_name,
        device=device,
        bank_repo_id=bank_repo_id,
        pretrained_subfolder=pretrained_subfolder,
    )

    model_template = load_pretrained_checkpoint(
        pretrained_model_name=pretrained_model_name,
        device=device,
        pretrained_subfolder=pretrained_subfolder,
        bank_repo_id=bank_repo_id,
    )

    # Step 5: exact same simplex EG engine as current-order Hierarchical Routing.
    optimization_info = hr.multi_start_optimize(
        model_template, candidate_state_dicts, batch, device, config
    )

    # Step 6: compose the routed model with the same implementation as Current.
    best_weights = optimization_info["best_weights"]
    theta_B = hr.compose_model(
        model_template, candidate_state_dicts, best_weights, device
    )

    names = list(candidate_state_dicts.keys())
    selected_ft_df = selected_candidates_df[
        selected_candidates_df["is_pretrained"] == False
    ]

    info = {
        "candidate_names": names,
        "weights": dict(zip(names, best_weights.detach().cpu().tolist())),
        "batch_loss": float(optimization_info["best_loss"]),
        "vertex_losses": optimization_info["vertex_losses"],
        "pretrained_loss": float(pretrained_loss),
        "screening_lambda": 0.0,
        "screening_candidates": screening_candidates_df.to_dict(orient="records"),
        "screening_losses": {
            row["candidate_name"]: float(row["batch_loss"])
            for _, row in screening_candidates_df.iterrows()
        },
        "screening_top_h": selected_screening_df["candidate_name"].tolist(),
        "selected_sources": selected_ft_df["dataset_name"].tolist(),
        "selected_lambdas": {
            row["dataset_name"]: float(row["lambda_star"])
            for _, row in selected_ft_df.iterrows()
        },
        "source_relevance": source_relevance_df.to_dict(orient="records"),
        "selected_candidates": selected_candidates_df.to_dict(orient="records"),
        "best_start_id": optimization_info["best_start_id"],
        "best_start_name": optimization_info["best_start_name"],
        "best_iteration": optimization_info["best_iteration"],
        "best_is_vertex": optimization_info["best_is_vertex"],
        "best_vertex_name": optimization_info["best_vertex_name"],
        "optimization_trajectories": optimization_info["trajectories"],
        "config": config,
    }

    del model_template, candidate_state_dicts
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return theta_B, info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-bank-metadata", required=True)
    parser.add_argument("--pretrained-model-name", default="gpt2")
    parser.add_argument("--bank-repo-id", default=None)
    parser.add_argument("--pretrained-subfolder", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "This module exposes run_hierarchical_routing_new_order("
        "model_bank_metadata_path, batch, pretrained_model_name, device, config, ...)."
    )