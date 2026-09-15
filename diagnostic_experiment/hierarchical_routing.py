"""Hierarchical Routing: the paper's deployment-time routing strategy.

Given an incoming unlabeled batch B and a model bank
M = {theta_0} U {theta_{j,lambda}}:

1. Use the precomputed batch losses of every bank checkpoint and theta_0.
2. For each source j:
       lambda*_j(B) = argmin_lambda L_hat_B(theta_{j,lambda})
       alpha_j(B) = -min_lambda L_hat_B(theta_{j,lambda})
3. Consider theta_0 together with the best checkpoint of every source and
   keep the H representatives with the smallest batch loss.
4. Optimize interpolation weights w over the simplex Delta(C_B) via
   exponentiated-gradient descent, from several starting points, and
   compare against every simplex vertex.
5. Compose:
       theta_B = sum_k w*_k theta_k.

All routing decisions are driven by the incoming-batch language-model loss.
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
except ImportError:
    from torch.nn.utils.stateless import functional_call


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
# BANK SCORING
# ============================================================

def compute_bank_batch_losses(bank_df, batch, device, bank_repo_id=None):
    """Compute L_hat_B(theta_{j,lambda}) for every bank checkpoint."""
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


# ============================================================
# SOURCE RELEVANCE AND CANDIDATE SELECTION
# ============================================================

def compute_source_relevance(scored_bank_df):
    """Find lambda*_j(B), best loss and alpha_j(B) for each source."""
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
    """Keep the H source families with the largest alpha_j(B)."""
    H = min(H, len(source_relevance_df))
    return source_relevance_df.sort_values("alpha", ascending=False).head(H).reset_index(drop=True)


def select_routing_candidates(source_relevance_df, pretrained_loss, H):
    """Compete theta_0 against each source representative and keep H candidates total."""
    rows = [{
        "candidate_name": "theta_0",
        "dataset_name": "theta_0",
        "lambda_star": None,
        "best_loss": float(pretrained_loss),
        "alpha": -float(pretrained_loss),
        "is_pretrained": True,
    }]

    for _, source_row in source_relevance_df.iterrows():
        row = dict(source_row)
        row.update({
            "candidate_name": source_row["dataset_name"],
            "is_pretrained": False,
        })
        rows.append(row)

    representatives_df = pd.DataFrame(rows)
    num_candidates = min(H, len(representatives_df))

    selected_candidates_df = (
        representatives_df
        .sort_values("best_loss", ascending=True)
        .head(num_candidates)
        .reset_index(drop=True)
    )

    return representatives_df, selected_candidates_df


# ============================================================
# CANDIDATE STATE DICTS
# ============================================================

def build_candidate_state_dicts(selected_candidates_df, pretrained_model_name, device, bank_repo_id=None, pretrained_subfolder=None):
    """Load only selected candidates as CPU state dictionaries."""
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
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        del model

    return candidates


# ============================================================
# DIFFERENTIABLE INTERPOLATION
# ============================================================

def interpolate_state_dicts(candidate_state_dicts, weights, device):
    """theta(w) = sum_k w_k theta_k, differentiable with respect to weights."""
    names = list(candidate_state_dicts.keys())
    param_keys = candidate_state_dicts[names[0]].keys()
    interpolated = {}

    for key in param_keys:
        stacked = torch.stack(
            [candidate_state_dicts[name][key].to(device) for name in names],
            dim=0,
        )

        w = weights.view(-1, *([1] * (stacked.dim() - 1)))
        interpolated[key] = (w * stacked).sum(dim=0)

    return interpolated


def batch_loss_at_weights(model_template, candidate_state_dicts, weights, batch, device):
    """Forward pass of theta(w) on incoming batch B."""
    params = interpolate_state_dicts(candidate_state_dicts, weights, device)
    outputs = functional_call(model_template, params, args=(), kwargs=batch, tie_weights=False)
    return outputs.loss


def exponentiated_gradient_step(weights, grad, lr):
    """w_k <- w_k exp(-lr g_k) / sum_r w_r exp(-lr g_r)."""
    with torch.no_grad():
        scaled = weights * torch.exp(-lr * grad)
        return scaled / scaled.sum()


# ============================================================
# WEIGHT OPTIMIZATION
# ============================================================

def optimize_weights(model_template, candidate_state_dicts, batch, device, init_weights, num_iters, lr):
    """Run exponentiated gradient from one initialization."""
    weights = init_weights.clone().to(device)

    best_weights = None
    best_loss = float("inf")
    best_iteration = None
    trajectory = []

    for iteration in range(num_iters):
        weights = weights.detach().requires_grad_(True)
        loss = batch_loss_at_weights(model_template, candidate_state_dicts, weights, batch, device)
        current_loss = float(loss.item())

        trajectory.append({
            "iteration": iteration,
            "loss": current_loss,
            "weights": weights.detach().cpu().tolist(),
        })

        if current_loss < best_loss:
            best_loss = current_loss
            best_weights = weights.detach().clone()
            best_iteration = iteration

        loss.backward()
        grad = weights.grad.detach()
        weights = exponentiated_gradient_step(weights.detach(), grad, lr)

    with torch.no_grad():
        final_loss = float(
            batch_loss_at_weights(
                model_template,
                candidate_state_dicts,
                weights,
                batch,
                device,
            ).item()
        )

    trajectory.append({
        "iteration": num_iters,
        "loss": final_loss,
        "weights": weights.detach().cpu().tolist(),
    })

    if final_loss < best_loss:
        best_loss = final_loss
        best_weights = weights.detach().clone()
        best_iteration = num_iters

    return best_weights, best_loss, best_iteration, trajectory


def multi_start_optimize(model_template, candidate_state_dicts, batch, device, config):
    """Run EG from multiple starts and compare against every simplex vertex."""
    names = list(candidate_state_dicts.keys())
    K = len(names)

    with torch.no_grad():
        vertex_losses = []

        for i in range(K):
            vertex = torch.zeros(K, device=device)
            vertex[i] = 1.0

            loss = batch_loss_at_weights(
                model_template,
                candidate_state_dicts,
                vertex,
                batch,
                device,
            ).item()

            vertex_losses.append(float(loss))

    starts = [torch.full((K,), 1.0 / K)]
    start_names = ["uniform"]

    flat_tau = config.get("flat_tau", DEFAULT_CONFIG["flat_tau"])
    starts.append(
        F.softmax(
            -torch.tensor(vertex_losses, dtype=torch.float32) / flat_tau,
            dim=0,
        )
    )
    start_names.append("flat_soft")

    R = config.get("num_random_starts", DEFAULT_CONFIG["num_random_starts"])
    concentration = config.get(
        "dirichlet_concentration",
        DEFAULT_CONFIG["dirichlet_concentration"],
    )
    seed = config.get("seed", DEFAULT_CONFIG["seed"])

    dirichlet = torch.distributions.Dirichlet(
        torch.full((K,), concentration)
    )

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
            vertex = torch.zeros(K, device=device)
            vertex[i] = 1.0

            best_weights = vertex
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


# ============================================================
# MODEL COMPOSITION
# ============================================================

def compose_model(model_template, candidate_state_dicts, weights, device):
    """Materialize theta_B = sum_k w*_k theta_k."""
    with torch.no_grad():
        params = interpolate_state_dicts(
            candidate_state_dicts,
            weights,
            device,
        )

    model = copy.deepcopy(model_template).to(device)
    model.load_state_dict(params)
    model.eval()

    return model


# ============================================================
# HIERARCHICAL ROUTING
# ============================================================

def run_hierarchical_routing(model_bank_metadata_path, batch, pretrained_model_name, device, config=None, bank_repo_id=None, bank_df=None, scored_bank_df=None, pretrained_loss=None, pretrained_subfolder=None):
    """Run Hierarchical Routing and return theta_B plus routing information."""
    config = {**DEFAULT_CONFIG, **(config or {})}
    batch = move_batch_to_device(batch, device)

    if bank_df is None:
        bank_df = load_model_bank_metadata(model_bank_metadata_path)
        bank_df = filter_trained_rows(bank_df)

    if scored_bank_df is None:
        raise ValueError(
            "Hierarchical Routing requires the precomputed losses of all "
            "bank checkpoints through scored_bank_df."
        )

    if pretrained_loss is None:
        raise ValueError(
            "Hierarchical Routing requires the precomputed theta_0 batch "
            "loss through pretrained_loss."
        )

    source_relevance_df = compute_source_relevance(scored_bank_df)

    representatives_df, selected_candidates_df = select_routing_candidates(
        source_relevance_df,
        pretrained_loss,
        config["H"],
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

    optimization_info = multi_start_optimize(
        model_template,
        candidate_state_dicts,
        batch,
        device,
        config,
    )

    best_weights = optimization_info["best_weights"]

    theta_B = compose_model(
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
            zip(
                names,
                best_weights.detach().cpu().tolist(),
            )
        ),
        "batch_loss": float(optimization_info["best_loss"]),
        "vertex_losses": optimization_info["vertex_losses"],
        "pretrained_loss": float(pretrained_loss),
        "source_relevance": source_relevance_df.to_dict(orient="records"),
        "representatives": representatives_df.to_dict(orient="records"),
        "selected_sources": selected_ft_df["dataset_name"].tolist(),
        "selected_lambdas": {
            row["dataset_name"]: float(row["lambda_star"])
            for _, row in selected_ft_df.iterrows()
        },
        "selected_candidates": selected_candidates_df.to_dict(orient="records"),
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
    parser.add_argument("--bank-repo-id", default=None)
    parser.add_argument("--pretrained-subfolder", default=None)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        "This module exposes run_hierarchical_routing("
        "model_bank_metadata_path, batch, pretrained_model_name, "
        "device, config, ...)."
    )