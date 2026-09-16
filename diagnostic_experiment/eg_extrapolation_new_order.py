"""Tangent-cone routing with source-first ERM screening.

This module combines the New Order candidate selection from
hierarchical_routing_new_order.py with the tangent-cone optimization from
eg_extrapolation.py.

Given an incoming unlabeled batch B and a model bank
M = {theta_0} U {theta_{j,lambda}}:

1. Score theta_0 and the standard ERM checkpoint theta_{j,0} of every source.
2. Keep the H candidates with the smallest incoming-batch loss among:
   {theta_0} U {theta_{j,0}}.
3. For each selected fine-tuned source j, score all its lambda checkpoints:
   lambda*_j(B) = argmin_lambda L_hat_B(theta_{j,lambda}).
4. Build C_B from theta_0 when selected and the lambda*_j(B) checkpoint of
   every selected source.
5. Optimize over the tangent cone using the same multi-start Cone procedure,
   with plain Hierarchical Routing kept as its mandatory simplex fallback.
6. Compose theta_B from the selected weights.

Relative to eg_extrapolation.py, only the candidate-selection order changes.
The Cone update, anchor selection, adaptive line search, multi-start search
and simplex fallback are reused unchanged.
"""

import pandas as pd
import torch

import hierarchical_routing_new_order as hr_new
import eg_extrapolation as eg


DEFAULT_CONFIG = {
    **eg.DEFAULT_CONFIG,
    "H": 3,
}


# ============================================================
# CONE ROUTING — NEW ORDER
# ============================================================

def run_cone_routing_new_order(model_bank_metadata_path, batch, pretrained_model_name, device, config=None,
                               bank_repo_id=None, bank_df=None, pretrained_loss=None, pretrained_subfolder=None):
    """Run ERM-screened tangent-cone routing and return theta_B plus routing information."""
    config = {**DEFAULT_CONFIG, **(config or {})}
    batch = hr_new.move_batch_to_device(batch, device)

    if bank_df is None:
        bank_df = hr_new.load_model_bank_metadata(model_bank_metadata_path)
        bank_df = hr_new.filter_trained_rows(bank_df)

    if pretrained_loss is None:
        raise ValueError(
            "New-order Cone Routing requires the precomputed theta_0 batch loss "
            "through pretrained_loss."
        )

    # Step 1: score theta_{j,0} for every source; theta_0 is already scored.
    erm_df = hr_new.select_erm_representatives(bank_df)
    scored_erm_df = hr_new.compute_checkpoint_batch_losses(
        erm_df, batch, device, bank_repo_id
    )

    # Step 2: Top-H among theta_0 and all source ERM representatives.
    screening_candidates_df, selected_screening_df = hr_new.select_top_h_erm_candidates(
        scored_erm_df, pretrained_loss, config["H"]
    )

    selected_sources = selected_screening_df.loc[
        selected_screening_df["is_pretrained"] == False, "dataset_name"
    ].tolist()

    # Step 3: score every lambda only for fine-tuned sources surviving Top-H.
    if selected_sources:
        scored_selected_bank_df = hr_new.score_selected_source_banks(
            bank_df, selected_sources, scored_erm_df, batch, device, bank_repo_id
        )
        source_relevance_df = hr_new.compute_source_relevance(
            scored_selected_bank_df
        )
    else:
        scored_selected_bank_df = pd.DataFrame()
        source_relevance_df = pd.DataFrame()

    # Step 4: replace selected ERM representatives by their lambda*_j checkpoints.
    selected_candidates_df = hr_new.build_final_candidates(
        selected_screening_df,
        source_relevance_df,
        pretrained_loss,
    )

    candidate_state_dicts = hr_new.build_candidate_state_dicts(
        selected_candidates_df=selected_candidates_df,
        pretrained_model_name=pretrained_model_name,
        device=device,
        bank_repo_id=bank_repo_id,
        pretrained_subfolder=pretrained_subfolder,
    )

    model_template = hr_new.load_pretrained_checkpoint(
        pretrained_model_name=pretrained_model_name,
        device=device,
        pretrained_subfolder=pretrained_subfolder,
        bank_repo_id=bank_repo_id,
    )
    model_template.config.use_cache = False

    # Step 5: identical tangent-cone optimization to eg_extrapolation.py.
    # The Cone optimizer and its simplex fallback both use micro-batched
    # gradient accumulation through eg_extrapolation.py / hierarchical_routing.py.
    optimization_info = eg.multi_start_optimize_cone(
        model_template,
        candidate_state_dicts,
        batch,
        device,
        config,
    )

    best_weights = optimization_info["best_weights"]

    theta_B = hr_new.compose_model(
        model_template,
        candidate_state_dicts,
        best_weights,
        device,
    )

    names = list(candidate_state_dicts.keys())
    selected_ft_df = selected_candidates_df[
        selected_candidates_df["is_pretrained"] == False
    ]

    info = {
        "candidate_names": names,
        "weights": dict(
            zip(names, best_weights.detach().cpu().tolist())
        ),
        "batch_loss": float(optimization_info["best_loss"]),
        "used_cone": optimization_info["used_cone"],
        "anchor_name": optimization_info["anchor_name"],
        "anchor_mode": optimization_info["anchor_mode"],
        "per_anchor_best_loss": optimization_info["per_anchor_best_loss"],
        "cone_lr": optimization_info["cone_lr"],
        "max_log_step": optimization_info["max_log_step"],
        "cone_loss": optimization_info["cone_loss"],
        "baseline_loss": optimization_info["baseline_loss"],
        "baseline_best_is_vertex": optimization_info["baseline_best_is_vertex"],
        "baseline_best_vertex_name": optimization_info["baseline_best_vertex_name"],
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
        "cone_best_start_id": optimization_info["cone_best_start_id"],
        "cone_best_start_name": optimization_info["cone_best_start_name"],
        "cone_best_iteration": optimization_info["cone_best_iteration"],
        "cone_trajectories": optimization_info["cone_trajectories"],
        "baseline_trajectories": optimization_info["baseline_trajectories"],
        "config": config,
    }

    del model_template
    del candidate_state_dicts

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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        "This module exposes run_cone_routing_new_order("
        "model_bank_metadata_path, batch, pretrained_model_name, device, config, ...)."
    )