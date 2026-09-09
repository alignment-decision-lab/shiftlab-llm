"""Hard Routing and Flat Soft Routing: the two simpler baseline strategies
from the paper's Section 4/5.1, evaluated over the *entire* model bank
M = {theta_0} U {theta_{j,lambda}} -- unlike Hierarchical Routing
(hierarchical_routing.py), neither needs a top-H source reduction or the
exponentiated-gradient weight optimizer.

Hard routing:  theta_B = argmin_{theta in M} L_hat_B(theta)
Flat routing:  w_k(B) = softmax(-s_k(B)/tau) over every k in M;
               theta_B = sum_k w_k(B) theta_k  (closed form, no optimization)

Both reuse the scoring/interpolation machinery already built in
hierarchical_routing.py rather than duplicating it. Note the "flat"
starting point used inside hierarchical_routing.multi_start_optimize is a
different thing wearing a similar name -- it's a softmax-weighted init for
the Hierarchical optimizer's reduced H+1 candidate set, not this standalone
whole-bank strategy.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from algorithm_2 import load_model_bank_metadata
from hierarchical_routing import compute_bank_batch_losses, compose_model, filter_trained_rows, load_bank_checkpoint
from utils import move_batch_to_device


def compute_pretrained_loss(pretrained_model_name, batch, device):
    """L_hat_B(theta_0): the pretrained checkpoint's own loss on the batch,
    needed since Hard/Flat routing treat theta_0 as a bank member too."""
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(**batch)
    loss = outputs.loss.item()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return loss


def _bank_candidate_name(row):
    return f"{row['dataset_name']}_lambda_{row['lambda']:g}"


def _all_bank_losses(bank_df, pretrained_model_name, batch, device, bank_repo_id=None, scored_bank_df=None, pretrained_loss=None):
    if scored_bank_df is None:
        scored_bank_df = compute_bank_batch_losses(bank_df, batch, device, bank_repo_id)

    names = [_bank_candidate_name(row) for _, row in scored_bank_df.iterrows()]
    losses = scored_bank_df["batch_loss"].tolist()

    if pretrained_loss is None:
        pretrained_loss = compute_pretrained_loss(pretrained_model_name, batch, device)

    names.append("theta_0")
    losses.append(pretrained_loss)

    return names, losses


def compose_full_bank_streaming(bank_df, pretrained_model_name, names, weights, device, bank_repo_id=None):
    """Compose the Flat routing model while loading only one bank checkpoint at a time."""
    weight_dict = dict(zip(names, weights.tolist()))

    theta_0 = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
    mixed_state = {}

    with torch.no_grad():
        for name, tensor in theta_0.state_dict().items():
            if torch.is_floating_point(tensor):
                mixed_state[name] = tensor.detach().cpu().float() * weight_dict["theta_0"]
            else:
                mixed_state[name] = tensor.detach().cpu().clone()

    del theta_0

    for _, row in bank_df.iterrows():
        candidate_name = _bank_candidate_name(row)
        weight = weight_dict[candidate_name]
        model = load_bank_checkpoint(row, bank_repo_id)

        with torch.no_grad():
            for name, tensor in model.state_dict().items():
                if torch.is_floating_point(tensor):
                    mixed_state[name].add_(tensor.detach().cpu().float(), alpha=weight)

        del model

    model_template = AutoModelForCausalLM.from_pretrained(pretrained_model_name).to(device)
    model_template.load_state_dict(mixed_state)
    model_template.eval()

    del mixed_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return model_template


# ------------------------------------------------------------------
# Hard routing
# ------------------------------------------------------------------

def run_hard_routing(model_bank_metadata_path, batch, pretrained_model_name, device, bank_repo_id=None, bank_df=None, scored_bank_df=None, pretrained_loss=None):
    """theta_B = argmin_{theta in M} L_hat_B(theta), M = {theta_0} U bank.

    No interpolation -- the single best-scoring checkpoint is returned as is.

    bank_repo_id: if given, bank checkpoints are pulled from that Hub repo's
    subfolders instead of local `save_dir` paths -- see
    hierarchical_routing.load_bank_checkpoint().
    """
    batch = move_batch_to_device(batch, device)

    if bank_df is None:
        bank_df = load_model_bank_metadata(model_bank_metadata_path)
        bank_df = filter_trained_rows(bank_df)

    names, losses = _all_bank_losses(bank_df, pretrained_model_name, batch, device, bank_repo_id, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss)
    best_idx = min(range(len(losses)), key=lambda i: losses[i])
    best_name, best_loss = names[best_idx], losses[best_idx]

    if best_name == "theta_0":
        theta_B = AutoModelForCausalLM.from_pretrained(pretrained_model_name).to(device)
    else:
        theta_B = load_bank_checkpoint(bank_df.iloc[best_idx], bank_repo_id).to(device)

    info = {
        "selected": best_name,
        "batch_loss": best_loss,
        "all_losses": dict(zip(names, losses)),
    }
    return theta_B, info


# ------------------------------------------------------------------
# Flat soft routing
# ------------------------------------------------------------------

def run_flat_routing(model_bank_metadata_path, batch, pretrained_model_name, device, tau=1.0, bank_repo_id=None, bank_df=None, scored_bank_df=None, pretrained_loss=None):
    """w_k(B) = softmax(-s_k(B)/tau) over every k in M = {theta_0} U bank;
    theta_B = sum_k w_k(B) theta_k. Closed-form, no iterative optimization.

    bank_repo_id: if given, bank checkpoints are pulled from that Hub repo's
    subfolders instead of local `save_dir` paths -- see
    hierarchical_routing.load_bank_checkpoint().
    """
    batch = move_batch_to_device(batch, device)

    if bank_df is None:
        bank_df = load_model_bank_metadata(model_bank_metadata_path)
        bank_df = filter_trained_rows(bank_df)

    names, losses = _all_bank_losses(bank_df, pretrained_model_name, batch, device, bank_repo_id, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss)

    weights = F.softmax(-torch.tensor(losses) / tau, dim=0)

    theta_B = compose_full_bank_streaming(
        bank_df=bank_df,
        pretrained_model_name=pretrained_model_name,
        names=names,
        weights=weights,
        device=device,
        bank_repo_id=bank_repo_id,
    )

    info = {
        "weights": dict(zip(names, weights.tolist())),
        "all_losses": dict(zip(names, losses)),
        "tau": tau,
    }
    return theta_B, info
