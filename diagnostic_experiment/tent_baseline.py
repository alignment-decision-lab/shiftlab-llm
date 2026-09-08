"""CE-based, LayerNorm-affine-only online adaptation baseline -- a GPT-2
analog of TENT's mechanism (Wang et al., ICLR 2021), but optimizing the
real per-batch cross-entropy loss instead of entropy. TENT needs entropy
because image labels are genuinely unavailable at test time; that
constraint doesn't hold for causal LM batches, whose next-token labels
are already part of the text itself (see diagnostic_experiment/README.md
discussion), so we adapt directly against the true loss instead.

Includes a held-out split from the start: adapt on a prefix of the
arriving text, score on a disjoint suffix never touched by the gradient
step. An optimizer handed a number and told to minimize it will trivially
look good on that same number regardless of whether anything
generalizable was learned -- see prior discussion. Splitting along the
token dimension (not across batch examples) keeps this applicable even
to a single incoming sequence.
"""
import torch
from transformers import AutoModelForCausalLM

DEFAULT_CONFIG = {
    "lr": 1e-3,
    "num_steps": 1,
    "adapt_fraction": 0.5,
}


def configure_ln_affine(model):
    """Freeze everything except LayerNorm weight/bias -- the GPT-2 analog
    of TENT's configure_model(), which does the same for BatchNorm2d.
    Unlike BatchNorm, LayerNorm has no running statistics to swap out (it
    always computes fresh per-token stats regardless of train/eval mode),
    so only the affine-parameter optimization carries over from TENT.
    """
    for param in model.parameters():
        param.requires_grad = False

    ln_params = []
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.weight.requires_grad = True
            module.bias.requires_grad = True
            ln_params += [module.weight, module.bias]

    return ln_params


def split_batch_for_holdout(batch, adapt_fraction=0.5):
    """Split each sequence along the token dimension into an adapt prefix
    and a held-out eval suffix. Works for any batch size, including 1."""
    seq_len = batch["input_ids"].shape[1]
    split_point = max(1, min(seq_len - 1, int(seq_len * adapt_fraction)))

    batch_adapt = {k: v[:, :split_point] for k, v in batch.items()}
    batch_eval = {k: v[:, split_point:] for k, v in batch.items()}
    return batch_adapt, batch_eval


def _compute_loss(model, batch):
    with torch.no_grad():
        return model(**batch).loss.item()


def run_tent_adaptation(base_model_name_or_path, batch, device, config=None, canary_batch=None):
    """Adapt LayerNorm affine params to batch_adapt via real CE loss, then
    report loss on batch_adapt (in-sample, the "chased" number) and
    batch_eval (held-out, the fair generalization number) both before and
    after adaptation, so the actual delta from adapting is visible for
    both -- not just a single post-hoc number.

    canary_batch: optional batch from a domain unrelated to `batch`,
    never touched by the gradient step. If given, its loss is measured
    before and after adaptation to detect collateral damage/forgetting --
    i.e. did adapting to handle this batch's shift make the model worse
    at things that have nothing to do with it. Meaningful even in
    episodic (single-call) use: it shows the per-call footprint, which is
    the quantity that would accumulate across calls under online mode.
    """
    config = {**DEFAULT_CONFIG, **(config or {})}
    batch = {k: v.to(device) for k, v in batch.items()}
    if canary_batch is not None:
        canary_batch = {k: v.to(device) for k, v in canary_batch.items()}

    model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path).to(device)
    ln_params = configure_ln_affine(model)

    batch_adapt, batch_eval = split_batch_for_holdout(batch, config["adapt_fraction"])

    loss_adapt_before = _compute_loss(model, batch_adapt)
    loss_eval_before = _compute_loss(model, batch_eval)
    canary_loss_before = _compute_loss(model, canary_batch) if canary_batch is not None else None

    optimizer = torch.optim.Adam(ln_params, lr=config["lr"])
    for _ in range(config["num_steps"]):
        optimizer.zero_grad()
        loss = model(**batch_adapt).loss
        loss.backward()
        optimizer.step()

    loss_adapt_after = _compute_loss(model, batch_adapt)
    loss_eval_after = _compute_loss(model, batch_eval)
    canary_loss_after = _compute_loss(model, canary_batch) if canary_batch is not None else None

    info = {
        "loss_adapt_before": loss_adapt_before,
        "loss_adapt_after": loss_adapt_after,
        "loss_eval_before": loss_eval_before,
        "loss_eval_after": loss_eval_after,
        "canary_loss_before": canary_loss_before,
        "canary_loss_after": canary_loss_after,
    }
    return model, info
