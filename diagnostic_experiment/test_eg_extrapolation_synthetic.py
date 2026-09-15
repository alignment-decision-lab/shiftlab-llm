"""Synthetic, CPU-only sanity check for eg_extrapolation.py (paper
Appendix C.1, tangent-cone routing).

Same philosophy as test_hierarchical_routing_synthetic.py: tiny random
GPT-2 checkpoints, no download needed, seconds on CPU. This is a mechanics
and *guarantee* check -- does the cone update behave like the math says,
and does run_cone_routing ever do worse than plain Hierarchical Routing --
not a check of whether the cone actually helps on real data (it may not;
see the paper's Appendix C "Scope" paragraph).

Run with: python test_eg_extrapolation_synthetic.py
"""
import shutil
import tempfile

import torch

import eg_extrapolation as ce
import hierarchical_routing as hr
import routing_baselines as rb
from test_hierarchical_routing_synthetic import (
    TINY_CONFIG,
    make_synthetic_bank,
    make_synthetic_batch,
)


def test_cone_gradient_step_keeps_sum_to_one_and_non_anchor_positive():
    weights = torch.tensor([0.5, 0.3, 0.2])
    grad = torch.tensor([1.0, -1.0, 0.3])
    lr = 1.0
    anchor_idx = 0

    updated = ce.cone_gradient_step(weights, grad, lr, anchor_idx)

    assert torch.isclose(updated.sum(), torch.tensor(1.0), atol=1e-6), \
        "cone update must still sum to 1 (it's an affine combination, not a free linear one)"
    assert updated[1] > weights[1], "negative gradient should still increase weight (same EG direction as usual)"
    assert updated[2] < weights[2], "positive gradient should still decrease weight"
    assert updated[1] > 0 and updated[2] > 0, "non-anchor coordinates must stay strictly positive"
    print("test_cone_gradient_step_keeps_sum_to_one_and_non_anchor_positive: OK", updated.tolist())


def test_cone_gradient_step_anchor_can_go_negative():
    # Large negative gradient on the non-anchor coordinates blows their
    # weight up past 1, which must push the anchor's derived weight
    # negative -- this is the entire point of the relaxation (Appendix
    # C.1: "the optimizer may push past theta_bar").
    weights = torch.tensor([0.34, 0.33, 0.33])
    grad = torch.tensor([0.0, -5.0, -5.0])
    lr = 1.0
    anchor_idx = 0

    updated = ce.cone_gradient_step(weights, grad, lr, anchor_idx)

    assert torch.isclose(updated.sum(), torch.tensor(1.0), atol=1e-6)
    assert updated[0] < 0, f"anchor weight should go negative under a strong push, got {updated[0].item()}"
    assert updated[1] > 0 and updated[2] > 0, "non-anchor coordinates must stay positive even when anchor goes negative"
    print("test_cone_gradient_step_anchor_can_go_negative: OK", updated.tolist())


def test_cone_gradient_step_matches_plain_eg_when_anchor_untouched():
    # If the non-anchor coordinates happen to sum back to exactly 1 (no
    # push past the far facet), the cone update's anchor coordinate should
    # come out identical to what it started at only in the degenerate
    # zero-gradient case; more usefully, check that with zero gradient
    # nothing moves at all, matching plain EG's fixed-point behavior.
    weights = torch.tensor([0.5, 0.5])
    grad = torch.tensor([0.0, 0.0])
    lr = 1.0
    anchor_idx = 0

    updated = ce.cone_gradient_step(weights, grad, lr, anchor_idx)
    assert torch.allclose(updated, weights, atol=1e-6), "zero gradient should be a fixed point"
    print("test_cone_gradient_step_matches_plain_eg_when_anchor_untouched: OK", updated.tolist())


def test_select_anchor_pretrained_mode():
    names = ["SourceA", "theta_0", "SourceB"]
    vertex_losses = {"SourceA": 3.0, "theta_0": 5.0, "SourceB": 1.0}

    anchor_idx = ce.select_anchor(names, vertex_losses, mode="pretrained")
    assert names[anchor_idx] == "theta_0"
    print("test_select_anchor_pretrained_mode: OK")


def test_select_anchor_worst_mode():
    names = ["SourceA", "theta_0", "SourceB"]
    vertex_losses = {"SourceA": 3.0, "theta_0": 5.0, "SourceB": 1.0}

    anchor_idx = ce.select_anchor(names, vertex_losses, mode="worst")
    assert names[anchor_idx] == "theta_0", "theta_0 has the largest loss here, so it should be picked as the worst"
    print("test_select_anchor_worst_mode: OK")


def test_select_anchor_rejects_unknown_mode():
    try:
        ce.select_anchor(["theta_0"], {"theta_0": 1.0}, mode="nonsense")
        raise AssertionError("expected a ValueError for an unknown anchor mode")
    except ValueError:
        print("test_select_anchor_rejects_unknown_mode: OK")


def test_cone_finds_optimum_that_plain_eg_cannot_reach():
    """A quadratic loss with a KNOWN minimum outside the simplex but inside
    the cone at anchor 0. This isolates "can the mechanism find an
    improvement when one genuinely exists" from
    test_cone_routing_never_worse_than_hierarchical_routing's random
    synthetic bank, which has no real structure to extrapolate toward and
    so cannot tell the two apart.

    w_star = (-0.5, 0.75, 0.75): sums to 1 (a valid affine point), but
    w_star[0] < 0, so it is unreachable by plain EG (confined to the
    simplex) yet reachable by the cone at anchor_idx=0. L(w) = ||w -
    w_star||^2 has its unconstrained minimum exactly at w_star, so the
    cone should be able to drive the loss to ~0 while plain EG gets stuck
    at a positive-loss boundary point.
    """
    w_star = torch.tensor([-0.5, 0.75, 0.75])
    anchor_idx = 0

    def loss_fn(w):
        return ((w - w_star) ** 2).sum()

    def run_plain_eg(num_iters, lr):
        w = torch.full((3,), 1.0 / 3)
        for _ in range(num_iters):
            w = w.detach().requires_grad_(True)
            loss = loss_fn(w)
            loss.backward()
            grad = w.grad.detach()
            w = hr.exponentiated_gradient_step(w.detach(), grad, lr)
        return w.detach(), float(loss_fn(w).item())

    def run_cone(num_iters, lr):
        w = torch.full((3,), 1.0 / 3)
        for _ in range(num_iters):
            w = w.detach().requires_grad_(True)
            loss = loss_fn(w)
            loss.backward()
            grad = w.grad.detach()
            w = ce.cone_gradient_step(w.detach(), grad, lr, anchor_idx)
        return w.detach(), float(loss_fn(w).item())

    num_iters, lr = 200, 0.3
    plain_w, plain_loss = run_plain_eg(num_iters, lr)
    cone_w, cone_loss = run_cone(num_iters, lr)

    assert torch.isclose(plain_w.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(cone_w.sum(), torch.tensor(1.0), atol=1e-5)
    assert (plain_w >= 0).all(), "plain EG must stay on the simplex (non-negative)"

    assert plain_loss > 0.05, (
        f"plain EG should be stuck away from w_star (loss={plain_loss:.6f}); "
        "if this fails the test scenario itself needs revisiting"
    )
    assert cone_loss < 1e-3, f"cone search should reach ~w_star (loss={cone_loss:.6f})"
    assert cone_loss < plain_loss, (
        f"cone loss ({cone_loss:.6f}) should be far below plain EG's loss ({plain_loss:.6f}) "
        "when the true optimum lies outside the simplex"
    )
    assert cone_w[0] < 0, f"cone should have pushed the anchor negative, got {cone_w[0].item()}"
    assert torch.allclose(cone_w, w_star, atol=1e-2), f"cone should converge near w_star, got {cone_w.tolist()}"

    print("test_cone_finds_optimum_that_plain_eg_cannot_reach: OK")
    print(f"  plain EG:  w={plain_w.tolist()}  loss={plain_loss:.6f}")
    print(f"  cone:      w={cone_w.tolist()}  loss={cone_loss:.6f}  (w_star={w_star.tolist()})")


def test_cone_discovers_far_facet_without_diverging():
    """3 candidates, a global optimum placed FAR outside the simplex (past
    one specific facet, by a wide margin -- not just a small nudge), and
    no anchor told in advance: the search must both DISCOVER which of the
    3 facets to relax (by trying each) and CONVERGE all the way to the
    distant optimum. Also locks in the fix for a real bug this exact
    scenario found: with a poorly-scaled cone_lr, a single step can
    overshoot so badly that it drives a non-anchor weight to *exactly*
    floating-point 0.0 -- an absorbing state a multiplicative update can
    never escape (0 * anything = 0), silently and permanently trapping the
    search at the wrong vertex with no error raised. cone_gradient_step's
    max_log_step clipping must prevent that exact failure mode.
    """
    w_star = torch.tensor([-5.0, 3.0, 3.0])  # sums to 1; w_star[0]=-5 is far past the w_0=0 facet
    K = 3

    def loss_fn(w):
        return ((w - w_star) ** 2).sum()

    def run(num_iters, lr, anchor_idx, max_log_step=ce.MAX_LOG_STEP):
        w = torch.full((K,), 1.0 / K)
        evals = 0
        for _ in range(num_iters):
            w = w.detach().requires_grad_(True)
            loss = loss_fn(w)
            loss.backward()
            grad = w.grad.detach()
            evals += 1
            w = ce.cone_gradient_step(w.detach(), grad, lr, anchor_idx, max_log_step)
        return w.detach(), float(loss_fn(w).item()), evals

    # --- Part 1: facet discovery + convergence, with a properly-scaled lr ---
    baseline_iters, working_cone_lr = 300, 0.3
    per_anchor = {}
    for anchor_idx in range(K):
        w, loss, evals = run(baseline_iters, working_cone_lr, anchor_idx)
        per_anchor[anchor_idx] = (w, loss, evals)

    best_anchor = min(per_anchor, key=lambda i: per_anchor[i][1])
    best_w, best_loss, _ = per_anchor[best_anchor]
    total_evals = sum(v[2] for v in per_anchor.values())

    assert best_anchor == 0, f"should discover anchor 0 as the true facet, got {best_anchor}"
    assert torch.allclose(best_w, w_star, atol=1e-2), f"should converge to w_star, got {best_w.tolist()}"
    assert per_anchor[1][1] > 10 and per_anchor[2][1] > 10, "wrong-facet anchors should NOT reach anywhere near w_star"

    slowdown = total_evals / baseline_iters  # facet discovery costs exactly K x a single search
    assert abs(slowdown - K) < 1e-9, f"trying all {K} facets should cost exactly {K}x, got {slowdown}x"

    # --- Part 2: the bug this scenario found -- a poorly-scaled cone_lr
    # (the module's own unsafe-for-this-scale 10x-multiplier default) must
    # not silently lock onto exactly 0.0 anymore, now that clipping exists.
    unsafe_cone_lr = ce.CONE_LR_MULTIPLIER * working_cone_lr  # = 3.0, empirically catastrophic pre-fix
    w_unsafe, loss_unsafe, _ = run(50, unsafe_cone_lr, anchor_idx=0, max_log_step=ce.MAX_LOG_STEP)
    non_anchor = w_unsafe[[1, 2]]
    assert not torch.any(non_anchor == 0.0), (
        f"non-anchor weights must never hit exactly 0.0 (the unrecoverable absorbing state), got {w_unsafe.tolist()}"
    )

    print("test_cone_discovers_far_facet_without_diverging: OK")
    print(f"  per-anchor losses: {[round(per_anchor[i][1], 4) for i in range(K)]}  (correct anchor: 0)")
    print(f"  facet-discovery cost: {total_evals} evals = {slowdown:.2f}x a single baseline search ({baseline_iters} evals)")
    print(f"  unsafe cone_lr={unsafe_cone_lr}: no longer hits exact 0.0, w={w_unsafe.tolist()}")


def test_cone_lr_defaults_to_ten_times_lr_and_is_overridable():
    device = torch.device("cpu")
    tmp_dir = tempfile.mkdtemp(prefix="eg_cone_lr_default_")

    try:
        pretrained_dir, bank_metadata_path = make_synthetic_bank(tmp_dir)
        batch = make_synthetic_batch(vocab_size=TINY_CONFIG.vocab_size)
        bank_df = hr.filter_trained_rows(hr.load_model_bank_metadata(bank_metadata_path))
        scored_bank_df = hr.compute_bank_batch_losses(bank_df, batch, device)
        pretrained_loss = rb.compute_pretrained_loss(
            pretrained_model_name=pretrained_dir, batch=batch, device=device,
        )
        common_kwargs = dict(
            model_bank_metadata_path=bank_metadata_path,
            batch=batch,
            pretrained_model_name=pretrained_dir,
            device=device,
            bank_df=bank_df,
            scored_bank_df=scored_bank_df,
            pretrained_loss=pretrained_loss,
        )

        _, info_default = ce.run_cone_routing(
            config={"H": 3, "num_iters": 3, "num_random_starts": 1, "lr": 0.2},
            **common_kwargs,
        )
        assert info_default["cone_lr"] == ce.CONE_LR_MULTIPLIER * 0.2, (
            f"cone_lr should default to {ce.CONE_LR_MULTIPLIER}x lr, got {info_default['cone_lr']}"
        )

        _, info_override = ce.run_cone_routing(
            config={"H": 3, "num_iters": 3, "num_random_starts": 1, "lr": 0.2, "cone_lr": 0.7},
            **common_kwargs,
        )
        assert info_override["cone_lr"] == 0.7, "an explicit cone_lr should override the derived default"

        print("test_cone_lr_defaults_to_ten_times_lr_and_is_overridable: OK")
        print(f"  default cone_lr: {info_default['cone_lr']}  override cone_lr: {info_override['cone_lr']}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_cone_finds_advantage_on_real_model():
    """Locks in the fix from the lr investigation: plain EG's renormalization
    is an implicit convergence boost that the cone update must drop to let
    the anchor go negative, so the SAME nominal lr converges slower without
    it. At lr=0.1 (this scenario's default) the cone alone does not beat
    Hierarchical Routing within 5 iterations; at the default cone_lr
    (10x lr) it does. If this regresses, either CONE_LR_MULTIPLIER or the
    update rule itself broke.
    """
    device = torch.device("cpu")
    tmp_dir = tempfile.mkdtemp(prefix="eg_cone_advantage_")

    try:
        pretrained_dir, bank_metadata_path = make_synthetic_bank(tmp_dir)
        batch = make_synthetic_batch(vocab_size=TINY_CONFIG.vocab_size)
        bank_df = hr.filter_trained_rows(hr.load_model_bank_metadata(bank_metadata_path))
        scored_bank_df = hr.compute_bank_batch_losses(bank_df, batch, device)
        pretrained_loss = rb.compute_pretrained_loss(
            pretrained_model_name=pretrained_dir, batch=batch, device=device,
        )

        config = {"H": 3, "num_iters": 5, "num_random_starts": 2, "seed": 42, "anchor_mode": "worst"}

        _, plain_info = hr.run_hierarchical_routing(
            model_bank_metadata_path=bank_metadata_path, batch=batch,
            pretrained_model_name=pretrained_dir, device=device, config=config,
            bank_df=bank_df, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss,
        )

        _, cone_info = ce.run_cone_routing(
            model_bank_metadata_path=bank_metadata_path, batch=batch,
            pretrained_model_name=pretrained_dir, device=device, config=config,
            bank_df=bank_df, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss,
        )

        assert cone_info["used_cone"] is True, (
            "with the default cone_lr, the cone search should actually win on this "
            "scenario within the same 5-iteration budget, not just fall back"
        )
        assert cone_info["batch_loss"] < plain_info["batch_loss"], (
            f"cone ({cone_info['batch_loss']}) should beat plain Hierarchical Routing "
            f"({plain_info['batch_loss']}) with the properly-scaled cone_lr"
        )

        print("test_cone_finds_advantage_on_real_model: OK")
        print(f"  cone_lr={cone_info['cone_lr']}  cone_loss={cone_info['batch_loss']:.6f}"
              f"  plain_hierarchical_loss={plain_info['batch_loss']:.6f}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_cone_routing_never_worse_than_hierarchical_routing():
    """The one guarantee this extension actually offers: run_cone_routing's
    reported loss must never exceed plain run_hierarchical_routing's loss
    on the same bank, batch, and config, because the cone search's
    fallback IS hierarchical_routing.multi_start_optimize itself."""
    device = torch.device("cpu")
    tmp_dir = tempfile.mkdtemp(prefix="eg_cone_synthetic_")

    try:
        pretrained_dir, bank_metadata_path = make_synthetic_bank(tmp_dir)
        batch = make_synthetic_batch(vocab_size=TINY_CONFIG.vocab_size)
        bank_df = hr.filter_trained_rows(hr.load_model_bank_metadata(bank_metadata_path))
        scored_bank_df = hr.compute_bank_batch_losses(bank_df, batch, device)
        pretrained_loss = rb.compute_pretrained_loss(
            pretrained_model_name=pretrained_dir, batch=batch, device=device,
        )

        config = {
            "H": 3,
            "num_iters": 5,
            "lr": 0.1,
            "num_random_starts": 2,
            "seed": 42,
        }

        common_kwargs = dict(
            model_bank_metadata_path=bank_metadata_path,
            batch=batch,
            pretrained_model_name=pretrained_dir,
            device=device,
            bank_df=bank_df,
            scored_bank_df=scored_bank_df,
            pretrained_loss=pretrained_loss,
        )

        _, plain_info = hr.run_hierarchical_routing(config=config, **common_kwargs)

        for anchor_mode in ("pretrained", "worst", "all"):
            cone_theta_B, cone_info = ce.run_cone_routing(
                config={**config, "anchor_mode": anchor_mode},
                **common_kwargs,
            )

            weight_sum = sum(cone_info["weights"].values())
            assert abs(weight_sum - 1.0) < 1e-4, f"cone weights should still sum to 1, got {weight_sum}"

            assert cone_info["batch_loss"] <= plain_info["batch_loss"] + 1e-5, (
                f"[anchor_mode={anchor_mode}] cone routing ({cone_info['batch_loss']}) must never be "
                f"worse than plain Hierarchical Routing ({plain_info['batch_loss']})"
            )
            assert cone_info["baseline_loss"] == plain_info["batch_loss"] or abs(
                cone_info["baseline_loss"] - plain_info["batch_loss"]
            ) < 1e-5, "the internal fallback baseline should reproduce plain Hierarchical Routing exactly"

            cone_theta_B.eval()
            with torch.no_grad():
                out = cone_theta_B(**batch)
            assert out.loss.item() > 0
            assert abs(out.loss.item() - cone_info["batch_loss"]) < 1e-4, \
                "returned model's actual loss should match the reported batch_loss"

            if anchor_mode == "all":
                # "all" must actually have tried every candidate as anchor,
                # through the real run_cone_routing entry point -- not just
                # multi_start_optimize_cone in isolation, which is all the
                # earlier ad-hoc investigation ever exercised.
                K = len(cone_info["candidate_names"])
                assert len(cone_info["per_anchor_best_loss"]) == K, (
                    f"anchor_mode='all' should have tried all {K} candidates as anchor, "
                    f"got {len(cone_info['per_anchor_best_loss'])}: {cone_info['per_anchor_best_loss']}"
                )
                assert set(cone_info["per_anchor_best_loss"].keys()) == set(cone_info["candidate_names"])
                num_starts = 2 + config["num_random_starts"]  # uniform + flat_soft + R random
                assert len(cone_info["cone_trajectories"]) == K * num_starts * (config["num_iters"] + 1), (
                    "trajectory length should reflect K anchors x num_starts x (num_iters+1) points logged, "
                    f"got {len(cone_info['cone_trajectories'])}"
                )

            print(f"test_cone_routing_never_worse_than_hierarchical_routing[{anchor_mode}]: OK")
            print(f"  anchor: {cone_info['anchor_name']}  used_cone: {cone_info['used_cone']}")
            print(f"  cone_loss: {cone_info['cone_loss']:.6f}  baseline_loss: {cone_info['baseline_loss']:.6f}"
                  f"  plain_hierarchical_loss: {plain_info['batch_loss']:.6f}")
            if anchor_mode == "all":
                print(f"  per_anchor_best_loss: { {k: round(v,6) for k,v in cone_info['per_anchor_best_loss'].items()} }")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_cone_gradient_step_keeps_sum_to_one_and_non_anchor_positive()
    test_cone_gradient_step_anchor_can_go_negative()
    test_cone_gradient_step_matches_plain_eg_when_anchor_untouched()
    test_select_anchor_pretrained_mode()
    test_select_anchor_worst_mode()
    test_select_anchor_rejects_unknown_mode()
    test_cone_finds_optimum_that_plain_eg_cannot_reach()
    test_cone_discovers_far_facet_without_diverging()
    test_cone_lr_defaults_to_ten_times_lr_and_is_overridable()
    test_cone_finds_advantage_on_real_model()
    test_cone_routing_never_worse_than_hierarchical_routing()
    print("\nAll synthetic checks passed.")
