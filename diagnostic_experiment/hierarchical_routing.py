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
    "H": 2,                        # number of source families to keep (TopH)
    "num_iters": 100,              # exponentiated-gradient iterations per start
    "lr": 0.1,                     # exponentiated-gradient step size (fixed, no schedule)
    "num_random_starts": 5,        # R random Dirichlet-sampled starting points
    "dirichlet_concentration": 1.0,
    "flat_tau": 1.0,                # temperature for the flat soft-routing start
}


# ------------------------------------------------------------------
# Loading bank checkpoints -- either from local disk (models_bank.py's own
# output, `save_dir` column) or from a Hub repo's subfolder (`subfolder`
# column, e.g. the alignment-decision-lab/robustness-model-bank layout).
# ------------------------------------------------------------------

def load_bank_checkpoint(row, bank_repo_id=None):
    """Load one bank checkpoint, either from the Hub or from local disk.

    If `bank_repo_id` is given, `row['subfolder']` is loaded from that repo
    (e.g. "gpt2Medium/FreeLaw/lambda_0"); transformers downloads once and
    caches locally (~/.cache/huggingface/hub), so repeated calls across many
    batches don't re-download. Otherwise falls back to `row['save_dir']`,
    the local-path convention models_bank.py already produces -- this keeps
    a locally-trained bank working unchanged.
    """
    if bank_repo_id is not None:
        return AutoModelForCausalLM.from_pretrained(bank_repo_id, subfolder=row["subfolder"])
    return AutoModelForCausalLM.from_pretrained(row["save_dir"])


def filter_trained_rows(bank_df):
    """Drop rows whose 'status' column marks them as not yet trained.

    The Hub skeleton's metadata CSV pre-populates every (source, lambda)
    slot with status="pending" before training, then flips it to "trained"
    once a real checkpoint is pushed -- without this filter, a routing run
    against that CSV would try to load checkpoints that don't exist yet.
    Locally-trained metadata (models_bank.py's own output) has no 'status'
    column at all -- every row it contains is already a real checkpoint --
    so this is a no-op for that case.
    """
    if "status" not in bank_df.columns:
        return bank_df
    return bank_df[bank_df["status"] != "pending"].reset_index(drop=True)


# ------------------------------------------------------------------
# Steps 1-3: score every bank checkpoint, derive alpha_j(B) and lambda*_j(B)
# ------------------------------------------------------------------

def compute_bank_batch_losses(bank_df, batch, device, bank_repo_id=None):
    """Compute L_hat_B(theta_{j,lambda}) for every checkpoint in the bank.

    This is the expensive step of Hierarchical Routing: it requires loading
    and forward-passing every single bank member on B, since lambda*_j(B) is
    selected by direct empirical minimization rather than a precomputed
    lambda-rho curve. Returns bank_df with an added 'batch_loss' column.
    """
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
    """For each source, find its best-fitting robustness level and the
    resulting relevance score.

    alpha_j(B) = -min_{lambda in Lambda_j} L_hat_B(theta_{j,lambda})
    lambda*_j(B) = argmin_{lambda in Lambda_j} L_hat_B(theta_{j,lambda})

    Both come from the same groupby-and-argmin pass, matching the paper's
    "both quantities are obtained from a single pass over Lambda_j."
    """
    rows = []

    for dataset_name, group in scored_bank_df.groupby("dataset_name"):
        best_idx = group["batch_loss"].idxmin()
        best_row = group.loc[best_idx]

        # Keep every original column (save_dir for a local bank, subfolder
        # for a Hub-hosted one) so downstream loading works either way,
        # rather than hardcoding one location scheme here.
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
    """J_B <- TopH_j{alpha_j(B)}: keep the H sources with the largest
    alpha_j(B), i.e. the smallest best achievable loss."""
    H = min(H, len(source_relevance_df))
    return (
        source_relevance_df
        .sort_values("alpha", ascending=False)
        .head(H)
        .reset_index(drop=True)
    )


# ------------------------------------------------------------------
# Step 4: candidate set C_B = {theta_0} U {theta_{j,lambda*_j} : j in J_B}
# ------------------------------------------------------------------

def build_candidate_state_dicts(top_sources_df, pretrained_model_name, device, bank_repo_id=None):
    """Load theta_0 and the selected per-source checkpoints as plain state
    dicts. Kept as CPU tensors; only moved to `device` at interpolation time,
    since holding H+1 full gpt2-medium checkpoints on GPU simultaneously is
    unnecessary until they're actually combined.

    theta_0 always comes from `pretrained_model_name` directly (e.g. plain
    "gpt2-medium"), not from the bank repo's own theta_0/ subfolder -- those
    are still empty placeholders as of this writing, no checkpoint pushed.
    """
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
# Step 5: differentiable interpolation + exponentiated-gradient optimization
# ------------------------------------------------------------------

def interpolate_state_dicts(candidate_state_dicts, weights, device):
    """theta(w) = sum_k w_k * theta_k, differentiable w.r.t. `weights`.

    Building theta(w) this way -- rather than materializing a static average
    -- is what makes the exponentiated-gradient step below work without a
    separate dot-product computation: since theta(w) is linear in w,
    d(loss)/d(w_k) is *exactly* <grad_theta L_hat_B(theta(w)), theta_k> by
    the chain rule, so plain autograd on `weights` already gives the g_k
    values the paper's update rule needs.
    """
    names = list(candidate_state_dicts.keys())
    param_keys = candidate_state_dicts[names[0]].keys()

    interpolated = {}
    for key in param_keys:
        stacked = torch.stack([candidate_state_dicts[name][key].to(device) for name in names], dim=0)
        w = weights.view(-1, *([1] * (stacked.dim() - 1)))
        interpolated[key] = (w * stacked).sum(dim=0)

    return interpolated


def batch_loss_at_weights(model_template, candidate_state_dicts, weights, batch, device):
    """Forward pass of theta(w) on batch B. Calling .backward() on the
    result populates weights.grad with exactly g_k for every candidate k.

    tie_weights=False: GPT-2 ties lm_head.weight to transformer.wte.weight
    (same underlying tensor), so state_dict() lists both names holding
    identical values for every candidate. Interpolating them independently
    still produces identical results (same linear combination applied to
    identical inputs) -- this just stops functional_call from rejecting the
    (consistent) duplicate values.
    """
    params = interpolate_state_dicts(candidate_state_dicts, weights, device)
    outputs = functional_call(model_template, params, args=(), kwargs=batch, tie_weights=False)
    return outputs.loss


def exponentiated_gradient_step(weights, grad, lr):
    """w_k <- w_k exp(-lr g_k) / sum_r w_r exp(-lr g_r).

    Preserves the simplex constraint without explicit projection. Note:
    a weight that starts at exactly 0 stays at 0 under this update (0 times
    anything is 0) -- this is why simplex vertices are only used below as
    static comparison points, never as optimization starting points.
    """
    with torch.no_grad():
        scaled = weights * torch.exp(-lr * grad)
        return scaled / scaled.sum()


def optimize_weights(model_template, candidate_state_dicts, batch, device, init_weights, num_iters, lr):
    """Run exponentiated-gradient descent over the simplex from one starting
    point. Returns the lowest-loss iterate seen, not necessarily the final
    one, since the objective is nonconvex.
    """
    weights = init_weights.clone().to(device)

    with torch.no_grad():
        best_loss = batch_loss_at_weights(model_template, candidate_state_dicts, weights, batch, device).item()
    best_weights = weights.clone()

    for _ in range(num_iters):
        weights = weights.detach().requires_grad_(True)
        loss = batch_loss_at_weights(model_template, candidate_state_dicts, weights, batch, device)
        loss.backward()
        grad = weights.grad.detach()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = weights.detach().clone()

        weights = exponentiated_gradient_step(weights.detach(), grad, lr)

    return best_weights, best_loss


def multi_start_optimize(model_template, candidate_state_dicts, batch, device, config):
    """Run optimize_weights from the uniform mixture, the flat soft-routing
    solution, and R random Dirichlet-sampled interior points; separately
    evaluate (without optimizing) every simplex vertex; return the
    lowest-loss result overall. This guarantees the returned solution never
    performs worse on B than the best individual candidate, matching the
    paper's stated guarantee.
    """
    names = list(candidate_state_dicts.keys())
    K = len(names)

    with torch.no_grad():
        vertex_losses = []
        for i in range(K):
            v = torch.zeros(K, device=device)
            v[i] = 1.0
            vertex_losses.append(batch_loss_at_weights(model_template, candidate_state_dicts, v, batch, device).item())

    starts = [torch.full((K,), 1.0 / K)]

    flat_tau = config.get("flat_tau", DEFAULT_CONFIG["flat_tau"])
    starts.append(F.softmax(-torch.tensor(vertex_losses) / flat_tau, dim=0))

    R = config.get("num_random_starts", DEFAULT_CONFIG["num_random_starts"])
    concentration = config.get("dirichlet_concentration", DEFAULT_CONFIG["dirichlet_concentration"])
    dirichlet = torch.distributions.Dirichlet(torch.full((K,), concentration))
    for _ in range(R):
        starts.append(dirichlet.sample())

    num_iters = config.get("num_iters", DEFAULT_CONFIG["num_iters"])
    lr = config.get("lr", DEFAULT_CONFIG["lr"])

    best_weights, best_loss = None, float("inf")

    for start in starts:
        w, loss = optimize_weights(model_template, candidate_state_dicts, batch, device, start, num_iters, lr)
        if loss < best_loss:
            best_weights, best_loss = w, loss

    for i, vloss in enumerate(vertex_losses):
        if vloss < best_loss:
            v = torch.zeros(K, device=device)
            v[i] = 1.0
            best_weights, best_loss = v, vloss

    return best_weights, best_loss, dict(zip(names, vertex_losses))


# ------------------------------------------------------------------
# Step 6: compose theta_B
# ------------------------------------------------------------------

def compose_model(model_template, candidate_state_dicts, weights, device):
    """Materialize theta_B = sum_k w*_k theta_k as an actual loaded model."""
    with torch.no_grad():
        params = interpolate_state_dicts(candidate_state_dicts, weights, device)

    model = copy.deepcopy(model_template).to(device)
    model.load_state_dict(params)
    return model


# ------------------------------------------------------------------
# Top-level entry point
# ------------------------------------------------------------------

def run_hierarchical_routing(model_bank_metadata_path, batch, pretrained_model_name, device, config=None, bank_repo_id=None):
    """Run Hierarchical Routing and return the deployed model theta_B plus a
    trail of the routing decision.

    `batch` is moved to `device` once here; every helper below assumes it
    already lives there.

    bank_repo_id: if given (e.g. "alignment-decision-lab/robustness-model-bank"),
    bank checkpoints are pulled from that Hub repo's subfolders instead of
    from local `save_dir` paths -- see load_bank_checkpoint(). The metadata
    CSV itself can still come from anywhere, local or downloaded.
    """
    config = {**DEFAULT_CONFIG, **(config or {})}
    batch = move_batch_to_device(batch, device)

    bank_df = load_model_bank_metadata(model_bank_metadata_path)
    bank_df = filter_trained_rows(bank_df)
    scored_bank_df = compute_bank_batch_losses(bank_df, batch, device, bank_repo_id)
    source_relevance_df = compute_source_relevance(scored_bank_df)
    top_sources_df = select_top_h_sources(source_relevance_df, config["H"])

    candidate_state_dicts = build_candidate_state_dicts(top_sources_df, pretrained_model_name, device, bank_repo_id)

    model_template = AutoModelForCausalLM.from_pretrained(pretrained_model_name).to(device)
    model_template.eval()

    best_weights, best_loss, vertex_losses = multi_start_optimize(
        model_template, candidate_state_dicts, batch, device, config
    )

    theta_B = compose_model(model_template, candidate_state_dicts, best_weights, device)

    names = list(candidate_state_dicts.keys())
    info = {
        "candidate_names": names,
        "weights": {name: w for name, w in zip(names, best_weights.detach().cpu().tolist())},
        "batch_loss": best_loss,
        "vertex_losses": vertex_losses,
        "source_relevance": source_relevance_df.to_dict(orient="records"),
        "selected_sources": top_sources_df["dataset_name"].tolist(),
        "config": config,
    }

    return theta_B, info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-bank-metadata", required=True)
    parser.add_argument("--pretrained-model-name", default="gpt2-medium")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "This module exposes run_hierarchical_routing(model_bank_metadata_path, "
        "batch, pretrained_model_name, device, config) -- pass in a real "
        "tokenized deployment batch (input_ids/attention_mask/labels) from "
        "your calling script; there's no standalone batch source here."
    )
