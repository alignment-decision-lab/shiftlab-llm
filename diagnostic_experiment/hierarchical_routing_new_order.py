"""Hierarchical Routing with source-first ERM screening.

Given an incoming unlabeled batch B and a model bank
M = {theta_0} U {theta_{j,lambda}}:

1. Score theta_0 and the standard ERM checkpoint theta_{j,0} of every source.
2. Keep the H candidates with the smallest incoming-batch loss among:
       {theta_0} U {theta_{j,0}}.
3. For each selected fine-tuned source j, score all its lambda checkpoints:
       lambda*_j(B) = argmin_lambda L_hat_B(theta_{j,lambda}).
   If theta_0 was selected at Step 2, it remains unchanged.
4. Build C_B from theta_0 when selected and the lambda*_j(B) checkpoint of
   every selected source.
5. Optimize interpolation weights w over the simplex Delta(C_B) via
   exponentiated-gradient descent, from several starting points, and compare
   against every simplex vertex.
6. Compose:
       theta_B = sum_k w*_k theta_k.

Relative to hierarchical_routing.py, only the candidate-selection order changes.
The routing batch is unchanged; EG gradients are accumulated over micro-batches
only to reduce peak GPU memory.
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
    "microbatch_size": 4,
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
# ERM SCREENING
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
            loss = model(**batch, use_cache=False).loss.item()

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
    selected_screening_df = screening_candidates_df.sort_values(
        "batch_loss"
    ).head(num_candidates).reset_index(drop=True)

    return screening_candidates_df, selected_screening_df


# ============================================================
# SELECTED-SOURCE LAMBDA SCORING
# ============================================================

def score_selected_source_banks(bank_df, selected_sources, scored_erm_df, batch, device, bank_repo_id=None):
    """Score all lambdas only for selected sources, reusing their ERM losses."""
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
            loss = model(**batch, use_cache=False).loss.item()

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
        source_row = source_relevance_df[source_relevance_df["dataset_name"] == dataset_name]

        if source_row.empty:
            raise ValueError(f"No lambda scoring result found for selected source {dataset_name!r}.")

        row = dict(source_row.iloc[0])
        row.update({"candidate_name": dataset_name, "is_pretrained": False})
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


# ============================================================
# CANDIDATE STATE DICTS
# ============================================================

def build_candidate_state_dicts(selected_candidates_df, pretrained_model_name, device, bank_repo_id=None, pretrained_subfolder=None):
    """Load only final selected candidates as CPU state dictionaries."""
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
# DIFFERENTIABLE INTERPOLATION
# ============================================================

def interpolate_state_dicts(candidate_state_dicts, weights, device):
    """theta(w) = sum_k w_k theta_k, without stacking candidates on GPU."""
    names = list(candidate_state_dicts.keys())
    param_keys = candidate_state_dicts[names[0]].keys()
    interpolated = {}

    for key in param_keys:
        value = candidate_state_dicts[names[0]][key].to(device) * weights[0]
        for i, name in enumerate(names[1:], start=1):
            value = value + candidate_state_dicts[name][key].to(device) * weights[i]
        interpolated[key] = value

    return interpolated


def batch_loss_at_weights(model_template, candidate_state_dicts, weights, batch, device):
    """Forward pass of theta(w) on incoming batch B."""
    params = interpolate_state_dicts(candidate_state_dicts, weights, device)
    kwargs = dict(batch)
    kwargs["use_cache"] = False
    outputs = functional_call(model_template, params, args=(), kwargs=kwargs, tie_weights=False)
    return outputs.loss


def get_batch_size(batch):
    """Infer the number of sequences in a tokenized batch."""
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return value.shape[0]
    raise ValueError("Could not infer batch size.")


def slice_batch(batch, start, end):
    """Slice tensor-valued batch entries along the sequence dimension."""
    return {
        key: value[start:end] if torch.is_tensor(value) and value.ndim > 0 else value
        for key, value in batch.items()
    }


def microbatch_weight(microbatch, full_batch_size):
    """Weight a micro-batch so accumulated losses reproduce the full-batch mean."""
    return get_batch_size(microbatch) / full_batch_size


def batch_loss_and_grad_at_weights(model_template, candidate_state_dicts, weights, batch, device, microbatch_size):
    """Compute full-batch loss and dL/dw while freeing each micro-batch graph immediately."""
    batch_size = get_batch_size(batch)
    microbatch_size = min(max(1, int(microbatch_size)), batch_size)
    total_loss = 0.0
    total_grad = torch.zeros_like(weights)

    for start in range(0, batch_size, microbatch_size):
        microbatch = slice_batch(batch, start, min(start + microbatch_size, batch_size))
        weight = microbatch_weight(microbatch, batch_size)
        loss = batch_loss_at_weights(model_template, candidate_state_dicts, weights, microbatch, device)
        grad = torch.autograd.grad(loss, weights, retain_graph=False, create_graph=False)[0]
        total_loss += weight * float(loss.detach().item())
        total_grad.add_(grad.detach(), alpha=weight)
        del loss, grad

    return total_loss, total_grad


def exponentiated_gradient_step(weights, grad, lr):
    """w_k <- w_k exp(-lr g_k) / sum_r w_r exp(-lr g_r)."""
    with torch.no_grad():
        scaled = weights * torch.exp(-lr * grad)
        return scaled / scaled.sum()


# ============================================================
# WEIGHT OPTIMIZATION
# ============================================================

def optimize_weights(model_template, candidate_state_dicts, batch, device, init_weights, num_iters, lr, microbatch_size=4):
    """Run exponentiated gradient from one initialization using micro-batch accumulation."""
    weights = init_weights.clone().to(device)
    best_weights, best_loss, best_iteration = None, float("inf"), None
    trajectory = []

    for iteration in range(num_iters):
        weights = weights.detach().requires_grad_(True)
        current_loss, grad = batch_loss_and_grad_at_weights(
            model_template, candidate_state_dicts, weights, batch, device, microbatch_size
        )

        trajectory.append({
            "iteration": iteration,
            "loss": current_loss,
            "weights": weights.detach().cpu().tolist(),
        })

        if current_loss < best_loss:
            best_loss, best_weights, best_iteration = current_loss, weights.detach().clone(), iteration

        weights = exponentiated_gradient_step(weights.detach(), grad, lr)

    with torch.no_grad():
        final_loss = float(batch_loss_at_weights(
            model_template, candidate_state_dicts, weights, batch, device
        ).item())

    trajectory.append({
        "iteration": num_iters,
        "loss": final_loss,
        "weights": weights.detach().cpu().tolist(),
    })

    if final_loss < best_loss:
        best_loss, best_weights, best_iteration = final_loss, weights.detach().clone(), num_iters

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
                model_template, candidate_state_dicts, vertex, batch, device
            ).item()
            vertex_losses.append(float(loss))

    starts = [torch.full((K,), 1.0 / K)]
    start_names = ["uniform"]

    flat_tau = config.get("flat_tau", DEFAULT_CONFIG["flat_tau"])
    starts.append(F.softmax(
        -torch.tensor(vertex_losses, dtype=torch.float32) / flat_tau, dim=0
    ))
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
    microbatch_size = config.get("microbatch_size", DEFAULT_CONFIG["microbatch_size"])

    best_weights, best_loss = None, float("inf")
    best_start_id, best_start_name, best_iteration = None, None, None
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
            microbatch_size=microbatch_size,
        )

        for point in trajectory:
            point["start_id"] = start_id
            point["start_name"] = start_name

        all_trajectories.extend(trajectory)

        if loss < best_loss:
            best_weights, best_loss = weights, loss
            best_start_id, best_start_name, best_iteration = start_id, start_name, best_iter

    best_is_vertex, best_vertex_name = False, None

    for i, vertex_loss in enumerate(vertex_losses):
        if vertex_loss < best_loss:
            vertex = torch.zeros(K, device=device)
            vertex[i] = 1.0
            best_weights, best_loss = vertex, vertex_loss
            best_start_id, best_start_name, best_iteration = None, None, None
            best_is_vertex, best_vertex_name = True, names[i]

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
        params = interpolate_state_dicts(candidate_state_dicts, weights, device)

    model = copy.deepcopy(model_template).to(device)
    model.load_state_dict(params)
    model.eval()
    return model


# ============================================================
# HIERARCHICAL ROUTING — NEW ORDER
# ============================================================

def run_hierarchical_routing_new_order(model_bank_metadata_path, batch, pretrained_model_name, device, config=None, bank_repo_id=None, bank_df=None, pretrained_loss=None, pretrained_subfolder=None):
    """Run ERM-screened Hierarchical Routing and return theta_B plus routing information."""
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

    # Step 1: score theta_{j,0} for every source; theta_0 is already scored.
    erm_df = select_erm_representatives(bank_df)
    scored_erm_df = compute_checkpoint_batch_losses(erm_df, batch, device, bank_repo_id)

    # Step 2: Top-H among theta_0 and all source ERM representatives.
    screening_candidates_df, selected_screening_df = select_top_h_erm_candidates(
        scored_erm_df, pretrained_loss, config["H"]
    )

    selected_sources = selected_screening_df.loc[
        selected_screening_df["is_pretrained"] == False, "dataset_name"
    ].tolist()

    # Step 3: score every lambda only for fine-tuned sources surviving Top-H.
    if selected_sources:
        scored_selected_bank_df = score_selected_source_banks(
            bank_df, selected_sources, scored_erm_df, batch, device, bank_repo_id
        )
        source_relevance_df = compute_source_relevance(scored_selected_bank_df)
    else:
        scored_selected_bank_df = pd.DataFrame()
        source_relevance_df = pd.DataFrame()

    # Step 4: replace selected ERM representatives by their lambda*_j checkpoints.
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
    model_template.config.use_cache = False

    # Step 5: same simplex optimization as current-order Hierarchical Routing.
    optimization_info = multi_start_optimize(
        model_template, candidate_state_dicts, batch, device, config
    )

    best_weights = optimization_info["best_weights"]
    theta_B = compose_model(model_template, candidate_state_dicts, best_weights, device)

    names = list(candidate_state_dicts.keys())
    selected_ft_df = selected_candidates_df[selected_candidates_df["is_pretrained"] == False]

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "This module exposes run_hierarchical_routing_new_order("
        "model_bank_metadata_path, batch, pretrained_model_name, device, config, ...)."
    )