"""Proposed extension: tangent-cone routing (paper Appendix C.1).

Standalone module, deliberately kept separate from hierarchical_routing.py:
this is a proposed, not-yet-validated extension with no improvement
guarantee (see the paper's Appendix C "Scope" paragraph), and
hierarchical_routing.py is the file that produced every reported result --
it should not be touched by speculative code. Everything stable (candidate
selection, bank loading, state-dict interpolation, model composition) is
imported and reused as-is from hierarchical_routing.py; only the weight
UPDATE rule is new.

Given the model bank M = {theta_0} U {theta_{j,lambda}} and the usual
Top-H candidate set C_B (selected exactly as in Algorithm 1), Section 4
optimizes over the simplex Delta(C_B). This module instead optimizes over
the *tangent cone* of Delta(C_B) at a distinguished vertex theta_bar: the
facet opposite theta_bar is relaxed, so theta_bar's implicit weight may go
negative (extrapolate past it), while every other candidate stays strictly
non-negative (never negated). See Appendix C.1 for the full derivation.

Algorithmically this needs no new mechanism, only *removing* one: ordinary
EG's positivity is a property of the multiplicative update itself; it is
only the final renormalization that confines the iterate to the simplex.
Dropping that renormalization for the non-anchor coordinates, and deriving
the anchor's coordinate as 1 - sum(others), reaches the cone instead.

Because the cone is unbounded, there is no finite vertex set to fall back
on the way Algorithm 1 falls back on the H+1 simplex vertices. The only
guarantee available -- "no worse than plain Hierarchical Routing" -- is
recovered here explicitly, by running Algorithm 1's own multi-start search
(hierarchical_routing.multi_start_optimize) as a mandatory fallback
candidate and keeping whichever of the two is better.

The routing batch itself is unchanged. During optimization, gradients and
forward-only line-search losses are accumulated over micro-batches only to
reduce peak GPU memory.
"""

import torch
import torch.nn.functional as F

import hierarchical_routing as hr


DEFAULT_CONFIG = {
    "H": 3,
    "num_iters": 100,
    "lr": 0.1,
    "cone_lr": None,  # None => 10 * lr
    "max_log_step": None,  # None => MAX_LOG_STEP
    "num_random_starts": 5,
    "dirichlet_concentration": 1.0,
    "flat_tau": 1.0,
    "microbatch_size": 4,
    "seed": 42,
    "anchor_mode": "pretrained",  # or "worst", or "all"
}

# Caps |lr * grad_i| before exponentiating in cone_gradient_step, so a
# single step can never multiply a weight by more than exp(MAX_LOG_STEP).
# Without this, a poorly-scaled (lr, gradient) combination can drive a
# non-anchor weight to exactly floating-point 0.0 in one step -- an
# absorbing state a multiplicative update can never escape -- silently
# trapping the search at the wrong vertex with no error raised.
MAX_LOG_STEP = 10.0

# Plain EG's renormalization injects a shared, gradient-dependent rescaling.
# The cone update drops that renormalization so the anchor can go negative;
# a larger nominal step compensates for the corresponding slower motion.
CONE_LR_MULTIPLIER = 10.0


# ============================================================
# ANCHOR SELECTION
# ============================================================

def select_anchor(names, vertex_losses, mode="pretrained"):
    """Choose the distinguished vertex theta_bar for the cone relaxation."""
    if mode == "pretrained":
        if "theta_0" not in names:
            raise ValueError(
                "anchor_mode='pretrained' requires theta_0 among the "
                "routing candidates."
            )
        return names.index("theta_0")

    if mode == "worst":
        return max(range(len(names)), key=lambda i: vertex_losses[names[i]])

    raise ValueError(
        f"Unknown anchor_mode: {mode!r} "
        "(expected 'pretrained' or 'worst')."
    )


# ============================================================
# TANGENT-CONE UPDATE
# ============================================================

def cone_gradient_step(weights, grad, lr, anchor_idx, max_log_step=MAX_LOG_STEP):
    """One EG update restricted to the tangent cone at anchor_idx."""
    with torch.no_grad():
        K = weights.shape[0]
        mask = torch.ones(K, dtype=torch.bool, device=weights.device)
        mask[anchor_idx] = False

        log_step = torch.clamp(
            -lr * grad[mask],
            min=-max_log_step,
            max=max_log_step,
        )

        updated = weights.clone()
        updated[mask] = weights[mask] * torch.exp(log_step)
        updated[anchor_idx] = 1.0 - updated[mask].sum()

        return updated


# ============================================================
# MEMORY-SAFE FULL-BATCH LOSS
# ============================================================

def batch_loss_microbatched(model_template, candidate_state_dicts, weights, batch, device, microbatch_size):
    """Evaluate the same full-batch mean loss through sequential micro-batches.

    This is forward-only. It is used by the adaptive Cone line search and
    final-loss evaluation so those evaluations do not recreate a large
    full-batch forward peak.
    """
    batch_size = hr.get_batch_size(batch)
    microbatch_size = min(max(1, int(microbatch_size)), batch_size)
    total_loss = 0.0

    with torch.no_grad():
        for start in range(0, batch_size, microbatch_size):
            microbatch = hr.slice_batch(
                batch,
                start,
                min(start + microbatch_size, batch_size),
            )
            weight = hr.microbatch_weight(microbatch, batch_size)
            loss = hr.batch_loss_at_weights(
                model_template,
                candidate_state_dicts,
                weights,
                microbatch,
                device,
            )
            total_loss += weight * float(loss.item())
            del loss

    return total_loss


# ============================================================
# WEIGHT OPTIMIZATION OVER THE CONE
# ============================================================

def adaptive_cone_step(base_weights, grad, current_loss, current_lr,
                       anchor_idx, loss_fn, max_log_step=MAX_LOG_STEP,
                       shrink_factor=0.5, grow_factor=1.2,
                       max_backtracks=10, min_lr=None, armijo_c=0.1):
    """Backtracking line search around cone_gradient_step.

    A candidate step is accepted only if it satisfies the Armijo sufficient
    decrease condition. Otherwise the trial learning rate is reduced before
    another candidate is evaluated.
    """
    if min_lr is None:
        min_lr = current_lr * 1e-4

    trial_lr = current_lr

    for _ in range(max_backtracks):
        candidate = cone_gradient_step(
            base_weights,
            grad,
            trial_lr,
            anchor_idx,
            max_log_step,
        )

        candidate_loss = loss_fn(candidate)
        expected_decrease = -torch.dot(
            grad,
            candidate - base_weights,
        ).item()
        actual_decrease = current_loss - candidate_loss

        if actual_decrease >= armijo_c * max(expected_decrease, 0.0):
            return (
                candidate,
                min(trial_lr * grow_factor, current_lr),
                True,
            )

        trial_lr *= shrink_factor

        if trial_lr < min_lr:
            break

    # No acceptable step found: remain at the current point.
    return base_weights, max(trial_lr, min_lr), False


def optimize_weights_cone(model_template, candidate_state_dicts, batch,
                          device, init_weights, num_iters, lr, anchor_idx,
                          max_log_step=MAX_LOG_STEP, adaptive=True,
                          shrink_factor=0.5, grow_factor=1.2,
                          max_backtracks=10, min_lr_ratio=1e-4,
                          microbatch_size=4):
    """Optimize weights over the tangent cone with micro-batched gradients.

    The mathematical batch B is unchanged. At each iteration the loss and
    dL/dw of that same batch are accumulated sequentially over micro-batches,
    allowing each autograd graph to be freed before the next one is built.
    """
    weights = init_weights.clone().to(device)
    current_lr = lr
    min_lr = lr * min_lr_ratio

    def loss_fn(w):
        return batch_loss_microbatched(
            model_template=model_template,
            candidate_state_dicts=candidate_state_dicts,
            weights=w,
            batch=batch,
            device=device,
            microbatch_size=microbatch_size,
        )

    best_weights = None
    best_loss = float("inf")
    best_iteration = None
    trajectory = []

    for iteration in range(num_iters):
        weights = weights.detach().requires_grad_(True)

        current_loss, grad = hr.batch_loss_and_grad_at_weights(
            model_template=model_template,
            candidate_state_dicts=candidate_state_dicts,
            weights=weights,
            batch=batch,
            device=device,
            microbatch_size=microbatch_size,
        )

        trajectory.append({
            "iteration": iteration,
            "loss": current_loss,
            "weights": weights.detach().cpu().tolist(),
            "lr": current_lr,
        })

        if current_loss < best_loss:
            best_loss = current_loss
            best_weights = weights.detach().clone()
            best_iteration = iteration

        base_weights = weights.detach()

        if not adaptive:
            weights = cone_gradient_step(
                base_weights,
                grad,
                current_lr,
                anchor_idx,
                max_log_step,
            )
            continue

        weights, current_lr, _ = adaptive_cone_step(
            base_weights,
            grad,
            current_loss,
            current_lr,
            anchor_idx,
            loss_fn,
            max_log_step,
            shrink_factor,
            grow_factor,
            max_backtracks,
            min_lr,
        )

    final_loss = loss_fn(weights)

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


# ============================================================
# MULTI-START
# ============================================================

def _generate_starts(vertex_losses_list, K, config):
    """Same starts as hierarchical_routing.multi_start_optimize."""
    starts = [torch.full((K,), 1.0 / K)]
    start_names = ["uniform"]

    flat_tau = config.get("flat_tau", DEFAULT_CONFIG["flat_tau"])
    starts.append(
        F.softmax(
            -torch.tensor(vertex_losses_list, dtype=torch.float32) / flat_tau,
            dim=0,
        )
    )
    start_names.append("flat_soft")

    R = config.get(
        "num_random_starts",
        DEFAULT_CONFIG["num_random_starts"],
    )
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

    return starts, start_names


def multi_start_optimize_cone(model_template, candidate_state_dicts,
                              batch, device, config):
    """Cone search with plain Hierarchical Routing as mandatory fallback."""
    # Mandatory simplex fallback. hr.multi_start_optimize now uses the same
    # micro-batched gradient accumulation.
    baseline = hr.multi_start_optimize(
        model_template,
        candidate_state_dicts,
        batch,
        device,
        config,
    )

    names = list(candidate_state_dicts.keys())
    K = len(names)
    anchor_mode = config.get(
        "anchor_mode",
        DEFAULT_CONFIG["anchor_mode"],
    )

    vertex_losses_list = [
        baseline["vertex_losses"][name]
        for name in names
    ]

    starts, start_names = _generate_starts(
        vertex_losses_list,
        K,
        config,
    )

    num_iters = config.get(
        "num_iters",
        DEFAULT_CONFIG["num_iters"],
    )
    lr = config.get(
        "lr",
        DEFAULT_CONFIG["lr"],
    )

    cone_lr = config.get(
        "cone_lr",
        DEFAULT_CONFIG["cone_lr"],
    )
    if cone_lr is None:
        cone_lr = CONE_LR_MULTIPLIER * lr

    max_log_step = config.get(
        "max_log_step",
        DEFAULT_CONFIG["max_log_step"],
    )
    if max_log_step is None:
        max_log_step = MAX_LOG_STEP

    microbatch_size = config.get(
        "microbatch_size",
        DEFAULT_CONFIG["microbatch_size"],
    )

    # anchor_mode="all" tries every candidate as theta_bar.
    anchor_indices = (
        range(K)
        if anchor_mode == "all"
        else [
            select_anchor(
                names,
                baseline["vertex_losses"],
                mode=anchor_mode,
            )
        ]
    )

    best_weights = None
    best_loss = float("inf")
    best_anchor_idx = None
    best_start_id = None
    best_start_name = None
    best_iteration = None
    all_trajectories = []
    per_anchor_best_loss = {}

    for anchor_idx in anchor_indices:
        anchor_best_loss = float("inf")

        for start_id, (start_name, start) in enumerate(
            zip(start_names, starts)
        ):
            weights, loss, best_iter, trajectory = optimize_weights_cone(
                model_template=model_template,
                candidate_state_dicts=candidate_state_dicts,
                batch=batch,
                device=device,
                init_weights=start,
                num_iters=num_iters,
                lr=cone_lr,
                anchor_idx=anchor_idx,
                max_log_step=max_log_step,
                microbatch_size=microbatch_size,
            )

            for point in trajectory:
                point["start_id"] = start_id
                point["start_name"] = start_name
                point["anchor_idx"] = anchor_idx
                point["anchor_name"] = names[anchor_idx]

            all_trajectories.extend(trajectory)

            if loss < anchor_best_loss:
                anchor_best_loss = loss

            if loss < best_loss:
                best_weights = weights
                best_loss = loss
                best_anchor_idx = anchor_idx
                best_start_id = start_id
                best_start_name = start_name
                best_iteration = best_iter

        per_anchor_best_loss[names[anchor_idx]] = anchor_best_loss

    cone_loss = best_loss
    cone_weights = best_weights
    used_cone = cone_loss < baseline["best_loss"]

    if used_cone:
        final_weights = cone_weights
        final_loss = cone_loss
    else:
        final_weights = baseline["best_weights"]
        final_loss = baseline["best_loss"]

    return {
        "best_weights": final_weights,
        "best_loss": final_loss,
        "used_cone": used_cone,
        "anchor_name": names[best_anchor_idx],
        "anchor_mode": anchor_mode,
        "per_anchor_best_loss": per_anchor_best_loss,
        "cone_lr": cone_lr,
        "max_log_step": max_log_step,
        "cone_loss": cone_loss,
        "cone_best_start_id": best_start_id,
        "cone_best_start_name": best_start_name,
        "cone_best_iteration": best_iteration,
        "baseline_loss": baseline["best_loss"],
        "baseline_best_is_vertex": baseline["best_is_vertex"],
        "baseline_best_vertex_name": baseline["best_vertex_name"],
        "vertex_losses": baseline["vertex_losses"],
        "cone_trajectories": all_trajectories,
        "baseline_trajectories": baseline["trajectories"],
    }


# ============================================================
# TOP-LEVEL ENTRY POINT
# ============================================================

def run_cone_routing(model_bank_metadata_path, batch,
                     pretrained_model_name, device, config=None,
                     bank_repo_id=None, bank_df=None,
                     scored_bank_df=None, pretrained_loss=None,
                     pretrained_subfolder=None):
    """Run tangent-cone routing and return theta_B plus routing information."""
    config = {**DEFAULT_CONFIG, **(config or {})}
    batch = hr.move_batch_to_device(batch, device)

    if bank_df is None:
        bank_df = hr.load_model_bank_metadata(
            model_bank_metadata_path
        )
        bank_df = hr.filter_trained_rows(bank_df)

    if scored_bank_df is None:
        raise ValueError(
            "Cone Routing requires the precomputed losses of all bank "
            "checkpoints through scored_bank_df."
        )

    if pretrained_loss is None:
        raise ValueError(
            "Cone Routing requires the precomputed theta_0 batch loss "
            "through pretrained_loss."
        )

    source_relevance_df = hr.compute_source_relevance(
        scored_bank_df
    )

    representatives_df, selected_candidates_df = (
        hr.select_routing_candidates(
            source_relevance_df,
            pretrained_loss,
            config["H"],
        )
    )

    candidate_state_dicts = hr.build_candidate_state_dicts(
        selected_candidates_df=selected_candidates_df,
        pretrained_model_name=pretrained_model_name,
        device=device,
        bank_repo_id=bank_repo_id,
        pretrained_subfolder=pretrained_subfolder,
    )

    model_template = hr.load_pretrained_checkpoint(
        pretrained_model_name=pretrained_model_name,
        device=device,
        pretrained_subfolder=pretrained_subfolder,
        bank_repo_id=bank_repo_id,
    )
    model_template.config.use_cache = False

    optimization_info = multi_start_optimize_cone(
        model_template,
        candidate_state_dicts,
        batch,
        device,
        config,
    )

    best_weights = optimization_info["best_weights"]

    theta_B = hr.compose_model(
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
        "batch_loss": float(
            optimization_info["best_loss"]
        ),
        "used_cone": optimization_info["used_cone"],
        "anchor_name": optimization_info["anchor_name"],
        "anchor_mode": optimization_info["anchor_mode"],
        "per_anchor_best_loss": optimization_info[
            "per_anchor_best_loss"
        ],
        "cone_lr": optimization_info["cone_lr"],
        "max_log_step": optimization_info["max_log_step"],
        "cone_loss": optimization_info["cone_loss"],
        "baseline_loss": optimization_info["baseline_loss"],
        "baseline_best_is_vertex": optimization_info[
            "baseline_best_is_vertex"
        ],
        "baseline_best_vertex_name": optimization_info[
            "baseline_best_vertex_name"
        ],
        "vertex_losses": optimization_info["vertex_losses"],
        "pretrained_loss": float(pretrained_loss),
        "source_relevance": source_relevance_df.to_dict(
            orient="records"
        ),
        "representatives": representatives_df.to_dict(
            orient="records"
        ),
        "selected_sources": selected_ft_df[
            "dataset_name"
        ].tolist(),
        "selected_lambdas": {
            row["dataset_name"]: float(row["lambda_star"])
            for _, row in selected_ft_df.iterrows()
        },
        "selected_candidates": selected_candidates_df.to_dict(
            orient="records"
        ),
        "cone_best_start_id": optimization_info[
            "cone_best_start_id"
        ],
        "cone_best_start_name": optimization_info[
            "cone_best_start_name"
        ],
        "cone_best_iteration": optimization_info[
            "cone_best_iteration"
        ],
        "cone_trajectories": optimization_info[
            "cone_trajectories"
        ],
        "baseline_trajectories": optimization_info[
            "baseline_trajectories"
        ],
        "config": config,
    }

    del model_template
    del candidate_state_dicts

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return theta_B, info


if __name__ == "__main__":
    print(
        "This module exposes run_cone_routing(model_bank_metadata_path, "
        "batch, pretrained_model_name, device, config, ...) -- the "
        "tangent-cone extension of hierarchical_routing.run_hierarchical_"
        "routing (paper Appendix C.1). See test_eg_extrapolation_synthetic.py "
        "for runnable examples."
    )