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
"""

import torch
import torch.nn.functional as F

import hierarchical_routing as hr


DEFAULT_CONFIG = {
    "H": 3,
    "num_iters": 100,
    "lr": 0.1,
    "cone_lr": None,  # None => 10 * lr (see CONE_LR_MULTIPLIER below)
    "max_log_step": None,  # None => MAX_LOG_STEP below
    "num_random_starts": 5,
    "dirichlet_concentration": 1.0,
    "flat_tau": 1.0,
    "seed": 42,
    "anchor_mode": "pretrained",  # or "worst", or "all" (try every candidate, K times the cost)
}

# Caps |lr * grad_i| before exponentiating in cone_gradient_step, so a
# single step can never multiply a weight by more than exp(MAX_LOG_STEP).
# Without this, a poorly-scaled (lr, gradient) combination can drive a
# non-anchor weight to exactly floating-point 0.0 in one step -- an
# absorbing state a multiplicative update can never escape -- silently
# trapping the search at the wrong vertex with no error raised. See
# test_cone_discovers_far_facet_without_diverging, which reproduces this
# failure at MAX_LOG_STEP=None (unclipped) and shows clipping fixes it.
MAX_LOG_STEP = 10.0

# Plain EG's renormalization (w / sum(w)) is not just a simplex-projection
# step: it also injects a shared, gradient-dependent rescaling
# (Z ~ 1 - lr * sum_k w_k g_k) that acts as a free, data-dependent
# convergence boost. The cone update must drop that renormalization to let
# the anchor coordinate go negative, so it loses that boost too -- the same
# nominal `lr` therefore moves genuinely slower without it, not just
# differently. Empirically (see test_cone_finds_advantage_on_real_model),
# a 10x larger learning rate at the *same* iteration budget recovers the
# improvement that more iterations at the borrowed `lr` would otherwise
# have required -- i.e. this compensates in step size, not compute.
CONE_LR_MULTIPLIER = 10.0


# ============================================================
# ANCHOR SELECTION (Appendix C.1, "Choosing theta_bar")
# ============================================================

def select_anchor(names, vertex_losses, mode="pretrained"):
    """Choose the distinguished vertex theta_bar for the cone relaxation.

    Both heuristics use quantities the shared scoring pass already
    computes, so neither costs anything extra to evaluate:
      - "pretrained": theta_bar = theta_0, matching the
        extrapolate-past-the-fine-tune intuition of Wortsman et al. (2022b).
      - "worst": theta_bar = the candidate with the largest batch loss,
        letting the optimizer actively cancel the least relevant direction
        rather than merely floor its weight at zero.

    Returns the integer index of theta_bar within `names`.
    """
    if mode == "pretrained":
        if "theta_0" not in names:
            raise ValueError(
                "anchor_mode='pretrained' requires theta_0 among the "
                "routing candidates."
            )
        return names.index("theta_0")

    if mode == "worst":
        return max(range(len(names)), key=lambda i: vertex_losses[names[i]])

    raise ValueError(f"Unknown anchor_mode: {mode!r} (expected 'pretrained' or 'worst').")


# ============================================================
# TANGENT-CONE UPDATE (Appendix C.1, eq. 24)
# ============================================================

def cone_gradient_step(weights, grad, lr, anchor_idx, max_log_step=MAX_LOG_STEP):
    """One EG update restricted to the tangent cone at index `anchor_idx`.

    The K-1 non-anchor coordinates follow the ordinary multiplicative EG
    rule *without* renormalizing, so they stay strictly positive but their
    sum is free to drift away from 1. The anchor coordinate is then set to
    1 minus that sum, so weights.sum() == 1 is preserved exactly as in the
    simplex case -- only the anchor's sign is no longer constrained.

    Plain EG's renormalization keeps every step bounded automatically (it
    always re-projects onto the simplex); this update has no such
    correction, so an unlucky combination of lr and gradient scale can
    blow a weight up by a factor of e^16 or more in a *single* step. If
    that drives a non-anchor weight to exactly floating-point 0.0, the
    update can never recover from it (0 * anything = 0 is an absorbing
    state for a multiplicative rule) -- the search then gets silently and
    permanently stuck at the wrong vertex with no error raised. `max_log_step`
    clips |lr * grad_i| before exponentiating, bounding the maximum
    per-step multiplicative factor to exp(max_log_step) regardless of how
    large lr or the gradient happen to be, so no single step can reach that
    absorbing state. See test_cone_discovers_far_facet_without_diverging.
    """
    with torch.no_grad():
        K = weights.shape[0]
        mask = torch.ones(K, dtype=torch.bool, device=weights.device)
        mask[anchor_idx] = False

        log_step = torch.clamp(-lr * grad[mask], min=-max_log_step, max=max_log_step)

        updated = weights.clone()
        updated[mask] = weights[mask] * torch.exp(log_step)
        updated[anchor_idx] = 1.0 - updated[mask].sum()

        return updated


# ============================================================
# WEIGHT OPTIMIZATION OVER THE CONE
# ============================================================

def adaptive_cone_step(base_weights, grad, current_loss, current_lr, anchor_idx, loss_fn, max_log_step=MAX_LOG_STEP, shrink_factor=0.5, grow_factor=1.2, max_backtracks=10, min_lr=None, armijo_c=0.1):
    """Backtracking line search around cone_gradient_step: try the step at
    `current_lr`; if it doesn't decrease `loss_fn` by a *sufficient*
    amount, shrink the trial step by `shrink_factor` and retry (up to
    `max_backtracks` times, down to `min_lr`) before ever committing to a
    move. An accepted step grows the returned lr back up by `grow_factor`
    (capped at `current_lr`'s own starting value) so a temporarily
    cautious phase doesn't permanently slow down later calls.

    "Sufficient decrease" is an Armijo condition, not a plain
    `candidate_loss <= current_loss` check: the latter is provably not
    enough to kill oscillation, because a slowly-damping zigzag can
    satisfy "loss went down since the immediately preceding point" at
    *every single step* while making only tiny net progress for hundreds
    of iterations (see the K=4, distance=2.0 case in
    test_cone_convergence_study.py -- a bare non-increase check accepts
    every step there and never once shrinks, because 0.174 -> 0.171 is a
    real, if tiny, decrease). The Armijo condition instead compares the
    achieved decrease against `expected_decrease`, the first-order
    prediction for the actual step taken (`-<grad, candidate -
    base_weights>` -- valid for whatever direction cone_gradient_step
    produces, not just a pure -grad step): a step is only accepted if
    `current_loss - candidate_loss >= armijo_c * expected_decrease`. An
    overshooting step's real decrease falls far short of its linear
    prediction, which is exactly what makes it get rejected here instead
    of merely "not yet worse".

    `loss_fn(weights) -> float` is injected rather than hard-coded so this
    same, real code path can be driven by an analytic loss with a known
    ground truth in tests (test_cone_convergence_study.py) as well as by
    the real batch_loss_at_weights used in optimize_weights_cone below --
    the fix gets validated directly, not re-implemented in the test.

    Returns (new_weights, new_lr, accepted: bool).
    """
    if min_lr is None:
        min_lr = current_lr * 1e-4

    trial_lr = current_lr
    for _ in range(max_backtracks):
        candidate = cone_gradient_step(base_weights, grad, trial_lr, anchor_idx, max_log_step)
        candidate_loss = loss_fn(candidate)
        expected_decrease = -torch.dot(grad, candidate - base_weights).item()
        actual_decrease = current_loss - candidate_loss
        if actual_decrease >= armijo_c * max(expected_decrease, 0.0):
            return candidate, min(trial_lr * grow_factor, current_lr), True
        trial_lr *= shrink_factor
        if trial_lr < min_lr:
            break

    # No improving step found even at the floor: stay put rather than
    # take a step already known to make things worse.
    return base_weights, max(trial_lr, min_lr), False


def optimize_weights_cone(model_template, candidate_state_dicts, batch, device, init_weights, num_iters, lr, anchor_idx, max_log_step=MAX_LOG_STEP, adaptive=True, shrink_factor=0.5, grow_factor=1.2, max_backtracks=10, min_lr_ratio=1e-4):
    """Same loop as hierarchical_routing.optimize_weights, with the
    simplex-projecting update swapped for cone_gradient_step.

    Fixing CONE_LR_MULTIPLIER at one constant cannot be safe in general:
    the convergence study (see test_cone_convergence_study.py) shows the
    *same* nominal cone_lr converges in 2 iterations for a nearby
    off-simplex optimum but oscillates for hundreds of iterations without
    converging for a farther one, because the right step size depends on
    the local gradient magnitude, not on lr or K alone. Backtracking line
    search (adaptive=True, the default, via adaptive_cone_step above)
    fixes this without per-scenario tuning. Each backtracking trial costs
    one extra forward-only pass (no backward needed just to check a
    candidate's loss), so the added cost is bounded and only paid when
    the current step size actually turns out to be too aggressive.
    Passing adaptive=False recovers the old fixed-step behavior exactly
    (used by test_cone_gradient_step_* to test cone_gradient_step in
    isolation without this wrapper).
    """
    weights = init_weights.clone().to(device)
    current_lr = lr
    min_lr = lr * min_lr_ratio

    def loss_fn(w):
        with torch.no_grad():
            return float(hr.batch_loss_at_weights(model_template, candidate_state_dicts, w, batch, device).item())

    best_weights = None
    best_loss = float("inf")
    best_iteration = None
    trajectory = []

    for iteration in range(num_iters):
        weights = weights.detach().requires_grad_(True)
        loss = hr.batch_loss_at_weights(model_template, candidate_state_dicts, weights, batch, device)
        current_loss = float(loss.item())

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

        loss.backward()
        grad = weights.grad.detach()
        base_weights = weights.detach()

        if not adaptive:
            weights = cone_gradient_step(base_weights, grad, current_lr, anchor_idx, max_log_step)
            continue

        weights, current_lr, _ = adaptive_cone_step(
            base_weights, grad, current_loss, current_lr, anchor_idx, loss_fn,
            max_log_step, shrink_factor, grow_factor, max_backtracks, min_lr,
        )

    with torch.no_grad():
        final_loss = float(
            hr.batch_loss_at_weights(model_template, candidate_state_dicts, weights, batch, device).item()
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


def _generate_starts(vertex_losses_list, K, config):
    """Same start set as hierarchical_routing.multi_start_optimize
    (uniform, flat-softmax, R random Dirichlet draws), duplicated here
    (rather than imported) so this module never has to modify
    hierarchical_routing.py to share internal helpers."""
    starts = [torch.full((K,), 1.0 / K)]
    start_names = ["uniform"]

    flat_tau = config.get("flat_tau", DEFAULT_CONFIG["flat_tau"])
    starts.append(F.softmax(-torch.tensor(vertex_losses_list, dtype=torch.float32) / flat_tau, dim=0))
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

    return starts, start_names


def multi_start_optimize_cone(model_template, candidate_state_dicts, batch, device, config):
    """Cone-extrapolated search, with Algorithm 1's own plain-simplex
    result (hierarchical_routing.multi_start_optimize) as a mandatory
    fallback candidate -- this is what makes "no worse than Hierarchical
    Routing" an enforced comparison rather than an assumption; the cone
    search itself carries no such guarantee on its own (see Appendix C,
    "Scope"), since the cone is unbounded and has no finite vertex set to
    fall back on the way the simplex does.
    """
    baseline = hr.multi_start_optimize(model_template, candidate_state_dicts, batch, device, config)

    names = list(candidate_state_dicts.keys())
    K = len(names)
    anchor_mode = config.get("anchor_mode", DEFAULT_CONFIG["anchor_mode"])

    vertex_losses_list = [baseline["vertex_losses"][name] for name in names]
    starts, start_names = _generate_starts(vertex_losses_list, K, config)

    num_iters = config.get("num_iters", DEFAULT_CONFIG["num_iters"])
    lr = config.get("lr", DEFAULT_CONFIG["lr"])
    cone_lr = config.get("cone_lr", DEFAULT_CONFIG["cone_lr"])
    if cone_lr is None:
        cone_lr = CONE_LR_MULTIPLIER * lr
    max_log_step = config.get("max_log_step", DEFAULT_CONFIG["max_log_step"])
    if max_log_step is None:
        max_log_step = MAX_LOG_STEP

    # anchor_mode="all": try every candidate as theta_bar (Appendix C.1's
    # "the choice of theta_bar is not unique") and let the batch loss
    # decide, instead of committing to one heuristic. Costs K times a
    # single-anchor search.
    anchor_indices = range(K) if anchor_mode == "all" else [select_anchor(names, baseline["vertex_losses"], mode=anchor_mode)]

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

        for start_id, (start_name, start) in enumerate(zip(start_names, starts)):
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

def run_cone_routing(model_bank_metadata_path, batch, pretrained_model_name, device, config=None, bank_repo_id=None, bank_df=None, scored_bank_df=None, pretrained_loss=None, pretrained_subfolder=None):
    """Run tangent-cone routing and return theta_B plus routing
    information. Same required inputs and (theta_B, info) return shape as
    hierarchical_routing.run_hierarchical_routing, so this is a drop-in
    alternative for the same call sites."""
    config = {**DEFAULT_CONFIG, **(config or {})}
    batch = hr.move_batch_to_device(batch, device)

    if bank_df is None:
        bank_df = hr.load_model_bank_metadata(model_bank_metadata_path)
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

    source_relevance_df = hr.compute_source_relevance(scored_bank_df)

    representatives_df, selected_candidates_df = hr.select_routing_candidates(
        source_relevance_df,
        pretrained_loss,
        config["H"],
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

    selected_ft_df = selected_candidates_df[selected_candidates_df["is_pretrained"] == False]

    info = {
        "candidate_names": names,
        "weights": dict(zip(names, best_weights.detach().cpu().tolist())),
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
        "source_relevance": source_relevance_df.to_dict(orient="records"),
        "representatives": representatives_df.to_dict(orient="records"),
        "selected_sources": selected_ft_df["dataset_name"].tolist(),
        "selected_lambdas": {
            row["dataset_name"]: float(row["lambda_star"])
            for _, row in selected_ft_df.iterrows()
        },
        "selected_candidates": selected_candidates_df.to_dict(orient="records"),
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
    print(
        "This module exposes run_cone_routing(model_bank_metadata_path, "
        "batch, pretrained_model_name, device, config, ...) -- the "
        "tangent-cone extension of hierarchical_routing.run_hierarchical_"
        "routing (paper Appendix C.1). See test_eg_extrapolation_synthetic.py "
        "for runnable examples."
    )
