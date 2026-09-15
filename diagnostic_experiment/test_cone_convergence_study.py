"""Systematic convergence study for eg_extrapolation's adaptive_cone_step,
locking in the fix for the reliability problem found by hand in the
original convergence investigation:

A fixed cone_lr (the old CONE_LR_MULTIPLIER * lr, still the *starting*
value here) converges in 2 iterations for an off-simplex optimum close to
the simplex, but oscillates for 500+ iterations without converging for a
farther one, or for the same distance at a smaller K -- the right step
size depends on the local gradient magnitude, not on lr, distance, or K
individually. A first version of backtracking (accept any step that
doesn't increase the loss) did NOT fix this: a slowly-damping oscillation
still satisfies "loss went down since the last point" at every single
step (e.g. 0.174 -> 0.171), so naive backtracking never once shrinks.
The actual fix is an Armijo *sufficient decrease* condition: reject a step
whose achieved decrease falls far short of what the linearized gradient
predicted for that step, which is exactly what an overshooting,
oscillating step does (achieving ~0.4% of its linear prediction in the
motivating case below) and a good step does not.

This file drives the real, production adaptive_cone_step against an
analytic quadratic loss with a known ground-truth optimum -- fast, exact,
and no real model needed -- so the fix is validated directly rather than
re-implemented here. See test_cone_finds_advantage_on_real_model and
test_cone_routing_never_worse_than_hierarchical_routing in
test_eg_extrapolation_synthetic.py for the same machinery exercised
through the real neural-network path.

Run with: python test_cone_convergence_study.py
"""
import torch

import eg_extrapolation as ce


def make_w_star(K, anchor_idx, distance):
    """A ground-truth optimum that sums to 1 (a valid affine point) with
    w_star[anchor_idx] = -distance, and the remaining mass split evenly
    across the other K-1 coordinates -- i.e. `distance` outside the
    simplex, in the one direction the cone at `anchor_idx` can reach."""
    w = torch.full((K,), (1.0 + distance) / (K - 1))
    w[anchor_idx] = -distance
    return w


def run_to_convergence(K, w_star, anchor_idx, lr, num_iters=100, armijo_c=None):
    """Drives eg_extrapolation.adaptive_cone_step (the real production
    function) against L(w) = ||w - w_star||^2 from a uniform start."""
    def loss_fn(w):
        return float(((w - w_star) ** 2).sum().item())

    kwargs = {}
    if armijo_c is not None:
        kwargs["armijo_c"] = armijo_c

    w = torch.full((K,), 1.0 / K)
    current_lr = lr
    min_lr = lr * 1e-4
    for iteration in range(num_iters):
        w = w.detach().requires_grad_(True)
        loss = ((w - w_star) ** 2).sum()
        current_loss = float(loss.item())
        if current_loss < 1e-6:
            return w.detach(), current_loss, iteration
        loss.backward()
        grad = w.grad.detach()
        w, current_lr, _ = ce.adaptive_cone_step(
            w.detach(), grad, current_loss, current_lr, anchor_idx, loss_fn,
            ce.MAX_LOG_STEP, 0.5, 1.2, 10, min_lr, **kwargs,
        )
    return w.detach(), loss_fn(w), num_iters


# The exact starting cone_lr the module derives by default (Appendix C.1's
# CONE_LR_MULTIPLIER applied to a representative baseline lr) -- used
# unchanged across every scenario below. The point of this test is that
# NOTHING is tuned per scenario.
BASELINE_LR = 0.1
DEFAULT_CONE_LR = ce.CONE_LR_MULTIPLIER * BASELINE_LR
CONVERGED_TOL = 1e-3


def test_naive_backtracking_is_not_enough_armijo_is():
    """The motivating counter-example: at K=4, distance=2.0, a bare
    non-increase check accepts every oscillating step and never shrinks,
    because each step is a real (if tiny) improvement over the last. The
    Armijo condition correctly identifies these as insufficient and forces
    a shrink, because the achieved decrease is a tiny fraction of what the
    gradient predicted for that step size."""
    K = 4
    w_star = make_w_star(K, anchor_idx=0, distance=2.0)

    _, naive_loss, naive_iters = run_to_convergence(K, w_star, 0, DEFAULT_CONE_LR, num_iters=100, armijo_c=0.0)
    _, armijo_loss, armijo_iters = run_to_convergence(K, w_star, 0, DEFAULT_CONE_LR, num_iters=100, armijo_c=0.1)

    assert naive_loss > CONVERGED_TOL, (
        f"the counter-example should still fail to converge in 100 iters with armijo_c=0 "
        f"(bare non-increase), got loss={naive_loss}"
    )
    assert armijo_loss < CONVERGED_TOL, f"Armijo (c=0.1) should converge here, got loss={armijo_loss}"
    assert armijo_iters < naive_iters

    print("test_naive_backtracking_is_not_enough_armijo_is: OK")
    print(f"  armijo_c=0.0 (naive): loss={naive_loss:.6f} after {naive_iters} iters (did not converge)")
    print(f"  armijo_c=0.1 (fixed): loss={armijo_loss:.6f} after {armijo_iters} iters")


def test_convergence_across_distances_same_starting_lr():
    """K=3 fixed, sweeping how far w_star sits outside the simplex, using
    the SAME starting cone_lr (DEFAULT_CONE_LR) for every distance -- no
    per-scenario tuning. Before the Armijo fix, distances >= 1.0 failed to
    converge within 500 iterations at this same starting lr."""
    K = 3
    results = {}
    for distance in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        w_star = make_w_star(K, anchor_idx=0, distance=distance)
        w, loss, iters = run_to_convergence(K, w_star, 0, DEFAULT_CONE_LR)
        results[distance] = (loss, iters)
        assert loss < CONVERGED_TOL, f"distance={distance} should converge, got loss={loss} after {iters} iters"
        assert iters <= 20, f"distance={distance} took {iters} iters -- expected fast convergence (<=20)"

    print("test_convergence_across_distances_same_starting_lr: OK")
    for distance, (loss, iters) in results.items():
        print(f"  distance={distance:>5}: converged in {iters} iters, loss={loss:.2e}")


def test_convergence_across_K_same_starting_lr():
    """Fixed distance=2.0, sweeping K (number of candidates), same starting
    cone_lr throughout. Before the fix, K=2/3/4 failed to converge within
    500 iterations even though K=6/8/12 (at the same distance) were fine --
    confirming it's gradient magnitude, not K itself, that matters, and
    that the fix must not depend on knowing which regime you're in."""
    distance = 2.0
    results = {}
    for K in [2, 3, 4, 6, 8, 12]:
        w_star = make_w_star(K, anchor_idx=0, distance=distance)
        w, loss, iters = run_to_convergence(K, w_star, 0, DEFAULT_CONE_LR)
        results[K] = (loss, iters)
        assert loss < CONVERGED_TOL, f"K={K} should converge, got loss={loss} after {iters} iters"
        assert iters <= 20, f"K={K} took {iters} iters -- expected fast convergence (<=20)"

    print("test_convergence_across_K_same_starting_lr: OK")
    for K, (loss, iters) in results.items():
        print(f"  K={K:>3}: converged in {iters} iters, loss={loss:.2e}")


def test_accepted_step_always_lands_in_the_cone():
    """Sanity check independent of convergence: every accepted step across
    a full run stays a valid point of the cone (sums to 1, non-anchor
    coordinates non-negative) -- the line search only changes which step
    size is taken, never the feasible region cone_gradient_step defines."""
    K = 4
    w_star = make_w_star(K, anchor_idx=0, distance=2.0)

    def loss_fn(w):
        return float(((w - w_star) ** 2).sum().item())

    w = torch.full((K,), 1.0 / K)
    current_lr = DEFAULT_CONE_LR
    min_lr = current_lr * 1e-4
    for _ in range(30):
        w = w.detach().requires_grad_(True)
        loss = ((w - w_star) ** 2).sum()
        current_loss = float(loss.item())
        loss.backward()
        grad = w.grad.detach()
        w, current_lr, _ = ce.adaptive_cone_step(
            w.detach(), grad, current_loss, current_lr, 0, loss_fn, ce.MAX_LOG_STEP, 0.5, 1.2, 10, min_lr,
        )
        assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5), f"weights left the affine hull: {w.tolist()}"
        non_anchor = w[[i for i in range(K) if i != 0]]
        assert (non_anchor >= -1e-6).all(), f"a non-anchor coordinate went negative: {w.tolist()}"

    print("test_accepted_step_always_lands_in_the_cone: OK")


if __name__ == "__main__":
    test_naive_backtracking_is_not_enough_armijo_is()
    test_convergence_across_distances_same_starting_lr()
    test_convergence_across_K_same_starting_lr()
    test_accepted_step_always_lands_in_the_cone()
    print("\nAll convergence-study checks passed.")
