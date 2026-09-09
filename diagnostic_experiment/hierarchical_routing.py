"""Hierarchical Routing: the paper's proposed deployment-time routing
strategy (Algorithm 2, "Hierarchical (ours)" branch), as opposed to the
closest-source + fixed-midpoint strategy in algorithm_2.py.

Given an incoming unlabeled batch B and a model bank
M = {theta_0} U {theta_{j,lambda} : j in sources, lambda in Lambda_j}:

1. Score every bank checkpoint by its own loss on B (no separate distance
   measure -- same s(theta;B) used everywhere in the paper).
2. For each source j, alpha_j(B) = -min_lambda L_hat_B(theta_{j,lambda}) and
   lambda*_j(B) = argmin_lambda L_hat_B(theta_{j,lambda}) come from the same
   pass over that source's robustness grid.
3. Keep the H sources with the largest alpha_j(B) (TopH).
4. Build a small candidate set C_B = {theta_0} U {theta_{j,lambda*_j} : j in J_B}.
5. Optimize interpolation weights w over the simplex Delta(C_B) via
   exponentiated-gradient descent, from several starting points, comparing
   against every simplex vertex.
6. Compose theta_B = sum_k w*_k theta_k.

Unlike algorithm_2.py, this never touches token-KL distributional distance
at all -- every decision (source relevance, robustness level, interpolation
weights) is driven by the same batch-loss measurement.

Several hyperparameters used below (H, EG iteration count/step size, number
of random restarts, Dirichlet concentration) are not given concrete values
anywhere in the paper draft -- the defaults here are placeholders for you to
tune, not validated experimental settings.
"""

import copy

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from algorithm_2 import load_model_bank_metadata
from utils import move_batch_to_device

try:
    from torch.func import functional_call
except ImportError:  # torch < 2.0
    from torch.nn.utils.stateless import functional_call


DEFAULT_CONFIG = {
    "H": 2,
    "num_iters": 100,
    "lr": 0.1,
    "num_random_starts": 5,
    "dirichlet_concentration": 1.0,
    "flat_tau": 1.0,
    "seed": 42,
}


# ------------------------------------------------------------------
# Loading bank checkpoints
# ------------------------------------------------------------------

def load_bank_checkpoint(row, bank_repo_id=None):
    """Load one bank checkpoint, either from the Hub or from local disk."""
    if bank_repo_id is not None:
        return AutoModelForCausalLM.from_pretrained(bank_repo_id, subfolder=row["subfolder"])
    return AutoModelForCausalLM.from_pretrained(row["save_dir"])


def filter_trained_rows(bank_df):
    """Drop rows whose 'status' column marks them as not yet trained."""
    if "status" not in bank_df.columns:
        return bank_df
    return bank_df[bank_df["status"] != "pending"].reset_index(drop=True)


# ------------------------------------------------------------------
# Steps 1-3: score every bank checkpoint, derive alpha_j(B) and lambda*_j(B)
# ------------------------------------------------------------------

def compute_bank_batch_losses(bank_df, batch, device, bank_repo_id=None):
    """Compute L_hat_B(theta_{j,lambda}) for every checkpoint in the bank."""
    losses = []

    for _, row in bank_df.iterrows():
        model = load_bank_checkpoint(row, bank_repo_id).to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(**batch)

        losses.append(outputs.loss.item())

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scored = bank_df.copy()
    scored["batch_loss"] = losses
    return scored


def compute_source_relevance(scored_bank_df):
    """Find the best-fitting robustness level and relevance score for each source."""
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


def select_top_h_sources(source_relevance_df, H):
    """Keep the H sources with the largest alpha_j(B)."""
    H = min(H, len(source_relevance_df))
    return source_relevance_df.sort_values("alpha", ascending=False).head(H).reset_index(drop=True)


# ------------------------------------------------------------------
# Step 4: candidate set C_B
# ------------------------------------------------------------------

def build_candidate_state_dicts(top_sources_df, pretrained_model_name, device, bank_repo_id=None):
    """Load theta_0 and selected per-source checkpoints as CPU state dicts."""
    candidates = {}

    theta_0 = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
    candidates["theta_0"] = {k: v.detach().clone() for k, v in theta_0.state_dict().items()}
    del theta_0

    for _, row in top_sources_df.iterrows():
        model = load_bank_checkpoint(row, bank_repo_id)
        candidates[row["dataset_name"]] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        del model

    return candidates


# ------------------------------------------------------------------
# Step 5: differentiable interpolation + EG optimization
# ------------------------------------------------------------------

def interpolate_state_dicts(candidate_state_dicts, weights, device):
    """theta(w) = sum_k w_k theta_k, differentiable with respect to weights."""
    names = list(candidate_state_dicts.keys())
    param_keys = candidate_state_dicts[names[0]].keys()

    interpolated = {}

    for key in param_keys:
        stacked = torch.stack([candidate_state_dicts[name][key].to(device) for name in names], dim=0)
        w = weights.view(-1, *([1] * (stacked.dim() - 1)))
        interpolated[key] = (w * stacked).sum(dim=0)

    return interpolated


def batch_loss_at_weights(model_template, candidate_state_dicts, weights, batch, device):
    """Forward pass of theta(w) on batch B."""
    params = interpolate_state_dicts(candidate_state_dicts, weights, device)
    outputs = functional_call(model_template, params, args=(), kwargs=batch, tie_weights=False)
    return outputs.loss


def exponentiated_gradient_step(weights, grad, lr):
    """w_k <- w_k exp(-lr g_k) / sum_r w_r exp(-lr g_r)."""
    with torch.no_grad():
        scaled = weights * torch.exp(-lr * grad)
        return scaled / scaled.sum()


def optimize_weights(model_template, candidate_state_dicts, batch, device, init_weights, num_iters, lr):
    """Run EG from one starting point and record the full optimization trajectory."""
    weights = init_weights.clone().to(device)
    best_weights = None
    best_loss = float("inf")
    best_iteration = None
    trajectory = []

    for iteration in range(num_iters):
        weights = weights.detach().requires_grad_(True)
        loss = batch_loss_at_weights(model_template, candidate_state_dicts, weights, batch, device)
        current_loss = loss.item()

        trajectory.append({
            "iteration": iteration,
            "loss": float(current_loss),
            "weights": weights.detach().cpu().tolist(),
        })

        if current_loss < best_loss:
            best_loss = current_loss
            best_weights = weights.detach().clone()
            best_iteration = iteration

        loss.backward()
        grad = weights.grad.detach()
        weights = exponentiated_gradient_step(weights.detach(), grad, lr)

    # Evaluate the last point produced by the final EG update.
    with torch.no_grad():
        final_loss = batch_loss_at_weights(model_template, candidate_state_dicts, weights, batch, device).item()

    trajectory.append({
        "iteration": num_iters,
        "loss": float(final_loss),
        "weights": weights.detach().cpu().tolist(),
    })

    if final_loss < best_loss:
        best_loss = final_loss
        best_weights = weights.detach().clone()
        best_iteration = num_iters

    return best_weights, best_loss, best_iteration, trajectory


def multi_start_optimize(model_template, candidate_state_dicts, batch, device, config):
    """Run EG from multiple starts, compare with vertices and retain all trajectories."""
    names = list(candidate_state_dicts.keys())
    K = len(names)

    with torch.no_grad():
        vertex_losses = []
        for i in range(K):
            v = torch.zeros(K, device=device)
            v[i] = 1.0
            loss = batch_loss_at_weights(model_template, candidate_state_dicts, v, batch, device).item()
            vertex_losses.append(loss)

    starts = []
    start_names = []

    starts.append(torch.full((K,), 1.0 / K))
    start_names.append("uniform")

    flat_tau = config.get("flat_tau", DEFAULT_CONFIG["flat_tau"])
    starts.append(F.softmax(-torch.tensor(vertex_losses) / flat_tau, dim=0))
    start_names.append("flat_soft")

    R = config.get("num_random_starts", DEFAULT_CONFIG["num_random_starts"])
    concentration = config.get("dirichlet_concentration", DEFAULT_CONFIG["dirichlet_concentration"])
    seed = config.get("seed", DEFAULT_CONFIG["seed"])
    dirichlet = torch.distributions.Dirichlet(torch.full((K,), concentration))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for random_id in range(R):
            starts.append(dirichlet.sample())
            start_names.append(f"random_{random_id + 1}")

    num_iters = config.get("num_iters", DEFAULT_CONFIG["num_iters"])
    lr = config.get("lr", DEFAULT_CONFIG["lr"])

    best_weights = None
    best_loss = float("inf")
    best_start_id = None
    best_start_name = None
    best_iteration = None

    all_trajectories = []

    for start_id, (start_name, start) in enumerate(zip(start_names, starts)):
        weights, loss, best_iter, trajectory = optimize_weights(
            model_template=model_template,
            candidate_state_dicts=candidate_state_dicts,
            batch=batch,
            device=device,
            init_weights=start,
            num_iters=num_iters,
            lr=lr,
        )

        for point in trajectory:
            point["start_id"] = start_id
            point["start_name"] = start_name

        all_trajectories.extend(trajectory)

        if loss < best_loss:
            best_weights = weights
            best_loss = loss
            best_start_id = start_id
            best_start_name = start_name
            best_iteration = best_iter

    best_is_vertex = False
    best_vertex_name = None

    for i, vertex_loss in enumerate(vertex_losses):
        if vertex_loss < best_loss:
            v = torch.zeros(K, device=device)
            v[i] = 1.0

            best_weights = v
            best_loss = vertex_loss
            best_start_id = None
            best_start_name = None
            best_iteration = None
            best_is_vertex = True
            best_vertex_name = names[i]

    return {
        "best_weights": best_weights,
        "best_loss": best_loss,
        "best_start_id": best_start_id,
        "best_start_name": best_start_name,
        "best_iteration": best_iteration,
        "best_is_vertex": best_is_vertex,
        "best_vertex_name": best_vertex_name,
        "vertex_losses": dict(zip(names, vertex_losses)),
        "trajectories": all_trajectories,
    }


# ------------------------------------------------------------------
# Step 6: compose theta_B
# ------------------------------------------------------------------

def compose_model(model_template, candidate_state_dicts, weights, device):
    """Materialize theta_B = sum_k w*_k theta_k as an actual model."""
    with torch.no_grad():
        params = interpolate_state_dicts(candidate_state_dicts, weights, device)

    model = copy.deepcopy(model_template).to(device)
    model.load_state_dict(params)
    return model


# ------------------------------------------------------------------
# Top-level entry point
# ------------------------------------------------------------------

def run_hierarchical_routing(model_bank_metadata_path, batch, pretrained_model_name, device, config=None, bank_repo_id=None, bank_df=None, scored_bank_df=None):
    """Run Hierarchical Routing and return theta_B plus routing information."""
    config = {**DEFAULT_CONFIG, **(config or {})}
    batch = move_batch_to_device(batch, device)

    if bank_df is None:
        bank_df = load_model_bank_metadata(model_bank_metadata_path)
        bank_df = filter_trained_rows(bank_df)

    if scored_bank_df is None:
        scored_bank_df = compute_bank_batch_losses(bank_df, batch, device, bank_repo_id)

    source_relevance_df = compute_source_relevance(scored_bank_df)
    top_sources_df = select_top_h_sources(source_relevance_df, config["H"])

    candidate_state_dicts = build_candidate_state_dicts(top_sources_df, pretrained_model_name, device, bank_repo_id)

    model_template = AutoModelForCausalLM.from_pretrained(pretrained_model_name).to(device)
    model_template.eval()

    optimization_info = multi_start_optimize(model_template, candidate_state_dicts, batch, device, config)

    best_weights = optimization_info["best_weights"]
    theta_B = compose_model(model_template, candidate_state_dicts, best_weights, device)

    names = list(candidate_state_dicts.keys())

    info = {
        "candidate_names": names,
        "weights": dict(zip(names, best_weights.detach().cpu().tolist())),
        "batch_loss": optimization_info["best_loss"],
        "vertex_losses": optimization_info["vertex_losses"],
        "source_relevance": source_relevance_df.to_dict(orient="records"),
        "selected_sources": top_sources_df["dataset_name"].tolist(),
        "selected_lambdas": {row["dataset_name"]: float(row["lambda_star"]) for _, row in top_sources_df.iterrows()},
        "selected_candidates": top_sources_df.to_dict(orient="records"),
        "best_start_id": optimization_info["best_start_id"],
        "best_start_name": optimization_info["best_start_name"],
        "best_iteration": optimization_info["best_iteration"],
        "best_is_vertex": optimization_info["best_is_vertex"],
        "best_vertex_name": optimization_info["best_vertex_name"],
        "optimization_trajectories": optimization_info["trajectories"],
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
    parser.add_argument("--pretrained-model-name", default="gpt2-medium")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("This module exposes run_hierarchical_routing(model_bank_metadata_path, batch, pretrained_model_name, device, config).")
