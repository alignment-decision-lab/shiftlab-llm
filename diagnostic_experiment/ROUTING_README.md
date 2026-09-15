# Deployment-Time Routing: How to Use It

This is a practical how-to for **`hierarchical_routing.py`** and
**`routing_baselines.py`** — the three deployment-time routing strategies
compared in the paper (Section 4/5.1): **Hierarchical (ours)**, **Hard**,
and **Flat soft** — plus **`eg_extrapolation.py`**'s proposed tangent-cone
extension (paper Appendix C.1, branch `eg-cone-extrapolation`). For the
underlying math and where these fit in the overall Thread B pipeline, see
the main [README.md](README.md) — this document only covers *how to call
the code*.

All four strategies answer the same question — given a pretrained
checkpoint `theta_0`, a bank of `(source, lambda)` fine-tuned checkpoints,
and an unlabeled deployment batch `B` — which single model or interpolated
combination of models should be deployed on `B`? They differ in how much of
the bank they consider and how they combine it:

| Strategy | Function | Considers | Combination |
|---|---|---|---|
| Hierarchical (ours) | `hierarchical_routing.run_hierarchical_routing` | Top-H best-fitting **sources** only, each at its best λ (competing against `theta_0` for a slot — see `select_routing_candidates`) | Optimized interpolation weights (exponentiated gradient) |
| Hard | `routing_baselines.run_hard_routing` | Every `(source, λ)` checkpoint + `theta_0` | Single best checkpoint, no interpolation |
| Flat soft | `routing_baselines.run_flat_routing` | Top-H, same screening as Hierarchical (**not** the full bank — changed from the original design) | Closed-form softmax weights, no optimization |
| Cone extrapolation *(proposed, unvalidated)* | `eg_extrapolation.run_cone_routing` | Same Top-H candidates as Hierarchical | Same EG loop, but optimized over the tangent cone at one candidate instead of the simplex — see README.md item 10 |

All four share the same call shape: pass a bank metadata CSV path and a
tokenized batch in, get back `(theta_B, info)` — a ready-to-use
`transformers` model plus a dict recording the routing decision.

## 1. Prerequisites

**Environment.** These modules use flat imports (`from algorithm_2 import
...`, `from utils import ...`), so you need to run from inside
`diagnostic_experiment/`, or otherwise put it on `sys.path`. They also
transitively need the top-level `shiftlab` package (via
`algorithm_2.py` → `shift_measurement.py`), so `src/` must be on
`PYTHONPATH` too:

```bash
cd diagnostic_experiment
export PYTHONPATH=/path/to/shiftlab-llm/src
```

Required packages: `torch`, `transformers`, `pandas`, `datasets`,
`matplotlib`, `pyyaml`. A GPU is not required by the code, but is
effectively necessary for anything beyond a toy batch — pass whichever
`torch.device` you want (see §3); to pin everything to one GPU, set
`CUDA_VISIBLE_DEVICES` before launching Python rather than relying on the
default device.

**Hugging Face access.** If your bank checkpoints live on the Hub (e.g.
`alignment-decision-lab/robustness-model-bank`), you need to be logged in
(`hf auth login`) *before* calling any of these functions — they load
checkpoints via `AutoModelForCausalLM.from_pretrained(bank_repo_id,
subfolder=...)`, which needs read access to that repo.

> **Gotcha we hit in practice:** a *fine-grained* access token is scoped
> per-repo at creation time. Being a member of the `alignment-decision-lab`
> org is **not** enough — if the token wasn't explicitly granted access to
> `alignment-decision-lab/robustness-model-bank` specifically, every call
> against it will fail with a plain `404 Repository Not Found`, which looks
> identical to the repo not existing at all. Either scope a fine-grained
> token to that exact repo, or use a classic (non-fine-grained) token, which
> grants access to everything your account can already reach.

## 2. The model bank metadata CSV

Every routing function takes `model_bank_metadata_path`, a CSV with one row
per trained `(source, lambda)` checkpoint. Two shapes exist depending on
where the bank lives:

- **Hub-hosted** (`alignment-decision-lab/robustness-model-bank`'s own
  `model_bank_metadata.csv`): columns `dataset_name, lambda, subfolder,
  status`. Pass `bank_repo_id="alignment-decision-lab/robustness-model-bank"`
  so checkpoints load from that repo's `subfolder` column. Rows with
  `status == "pending"` (not yet trained/pushed) are automatically dropped
  by `filter_trained_rows` — you don't need to filter them yourself.
- **Locally-trained** (`models_bank.py`'s own output): columns include
  `save_dir` instead of `subfolder`, and there's no `status` column (every
  row is already a real checkpoint). Omit `bank_repo_id` and checkpoints
  load from `save_dir` on local disk.

Download the Hub CSV once with:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="alignment-decision-lab/robustness-model-bank",
    filename="model_bank_metadata.csv",
)
```

**For a quick/cheap test run**, filter this CSV down to a handful of rows
before passing it in — every routing function scores *every row it's
given*, so trimming the CSV (e.g. to 2 sources × 3 λ values) is the direct
lever for controlling how many checkpoints get downloaded and forward-passed:

```python
import pandas as pd
df = pd.read_csv(path)
small = df[df["dataset_name"].isin(["FreeLaw", "ArXiv"])
           & df["lambda"].isin([0.0, 0.5, 1.0])
           & (df["status"] == "trained")]
small.to_csv("small_bank_metadata.csv", index=False)
```

## 3. Building a deployment batch

`batch` must be a dict of tensors with `input_ids`, `attention_mask`, and
`labels` (labels are required — that's what makes
`model(**batch).loss` populated). It can live on CPU; every routing
function moves it to `device` internally.

```python
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

texts = [...]  # your incoming unlabeled deployment texts
enc = tokenizer(texts, padding="max_length", truncation=True,
                 max_length=128, return_tensors="pt")
labels = enc["input_ids"].clone()
labels[enc["attention_mask"] == 0] = -100  # ignore padded positions in the loss

batch = {"input_ids": enc["input_ids"],
         "attention_mask": enc["attention_mask"],
         "labels": labels}
```

## 4. Running each strategy

Hierarchical, Flat, and the cone extension all *require* the bank's shared
scores (`scored_bank_df`) and `theta_0`'s batch loss (`pretrained_loss`)
precomputed and passed in — they raise `ValueError` if you omit them. Hard
routing can compute these itself if you don't pass them, but the pattern
below computes them once and reuses them everywhere, matching
`routing_test.py`'s actual pipeline and charging the shared-scoring cost
only once instead of once per strategy:

```python
import torch
import hierarchical_routing as hr
import routing_baselines as rb
import eg_extrapolation as ce  # proposed extension, branch eg-cone-extrapolation
from utils import move_batch_to_device

device = torch.device("cuda:0")
metadata = "small_bank_metadata.csv"
bank_repo_id = "alignment-decision-lab/robustness-model-bank"
pretrained = "gpt2-medium"

# --- Shared scoring, computed once ---
bank_df = hr.filter_trained_rows(hr.load_model_bank_metadata(metadata))
batch_device = move_batch_to_device(batch, device)
scored_bank_df = hr.compute_bank_batch_losses(bank_df, batch_device, device, bank_repo_id)
pretrained_loss = rb.compute_pretrained_loss(
    pretrained_model_name=pretrained, batch=batch_device, device=device, bank_repo_id=bank_repo_id,
)
shared = dict(bank_df=bank_df, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss)

# --- Hierarchical (ours) ---
theta_B, info = hr.run_hierarchical_routing(
    model_bank_metadata_path=metadata,
    batch=batch,
    pretrained_model_name=pretrained,
    device=device,
    bank_repo_id=bank_repo_id,
    config={"H": 2, "num_iters": 100, "lr": 0.1, "num_random_starts": 5},
    **shared,
)

# --- Hard routing (single best checkpoint) ---
theta_B, info = rb.run_hard_routing(
    model_bank_metadata_path=metadata,
    batch=batch,
    pretrained_model_name=pretrained,
    device=device,
    bank_repo_id=bank_repo_id,
    **shared,
)

# --- Flat soft routing (Top-H, then closed-form softmax) ---
theta_B, info = rb.run_flat_routing(
    model_bank_metadata_path=metadata,
    batch=batch,
    pretrained_model_name=pretrained,
    device=device,
    bank_repo_id=bank_repo_id,
    tau=1.0,
    H=2,
    **shared,
)

# --- Cone extrapolation (proposed, NOT validated on real data -- see README.md item 10) ---
theta_B, info = ce.run_cone_routing(
    model_bank_metadata_path=metadata,
    batch=batch,
    pretrained_model_name=pretrained,
    device=device,
    bank_repo_id=bank_repo_id,
    config={"H": 2, "num_iters": 100, "lr": 0.1, "num_random_starts": 5, "anchor_mode": "worst"},
    **shared,
)
```

`theta_B` is always a ready-to-use `transformers` model (already on
`device`). `info` differs per strategy — see §6.

### Hierarchical config knobs (`config=` dict, all optional — shown with defaults)

| Key | Default | Meaning |
|---|---|---|
| `H` | 2 | Number of source families kept after Top-H selection |
| `num_iters` | 100 | Exponentiated-gradient iterations per optimization start |
| `lr` | 0.1 | EG step size (fixed, no schedule) |
| `num_random_starts` | 5 | Random Dirichlet-sampled starting points, in addition to the uniform and flat-softmax starts |
| `dirichlet_concentration` | 1.0 | Concentration parameter for those random starts |
| `flat_tau` | 1.0 | Temperature for the flat-softmax starting point |

None of these are values validated in the paper draft — they're reasonable
defaults, not tuned hyperparameters. For a fast smoke test, cut `num_iters`
and `num_random_starts` down (e.g. 15 and 2); for a real run, the defaults
above (or larger) are more appropriate.

### Cone extrapolation config knobs (`eg_extrapolation.py`, in addition to the above)

| Key | Default | Meaning |
|---|---|---|
| `anchor_mode` | `"pretrained"` | Which candidate to relax the facet at: `"pretrained"` (`theta_0`), `"worst"` (largest batch loss), or `"all"` (try every candidate, keep the best — costs K× a single-anchor search) |
| `cone_lr` | `None` → `10 * lr` | **Starting** step size for the cone update (not interchangeable with plain `lr` — dropping EG's renormalization also drops an implicit convergence boost it provides). As of the backtracking line search below, this is only a starting point: it self-corrects within a run, so the exact value matters far less than it used to. |
| `max_log_step` | `None` → `10.0` | Clips `\|trial_lr * grad_i\|` before exponentiating. Without this, an overly aggressive trial step could overshoot so badly in one step that a non-anchor weight hits exactly floating-point `0.0` — an absorbing state the multiplicative update can never escape. Keep this on; only raise it if you have a specific reason to allow larger single-step moves. |
| `adaptive` | `True` | Backtracking line search: if a trial step doesn't decrease the loss by at least `armijo_c` times what the linearized gradient predicted for that step (an Armijo sufficient-decrease condition), shrink the trial step by `shrink_factor` and retry (up to `max_backtracks` times) before committing to a move; an accepted step grows back toward the original `cone_lr` by `grow_factor`. A plain "did the loss go down at all" check turns out *not* to be enough — a slowly-damping oscillation satisfies that at every step while making almost no progress for hundreds of iterations; only the stricter Armijo condition actually catches it. Set `False` to recover the old fixed-step behavior. |
| `shrink_factor`, `grow_factor`, `max_backtracks`, `armijo_c` | `0.5`, `1.2`, `10`, `0.1` | Line-search internals. `armijo_c` is the one worth knowing about: `0.0` reduces to plain non-increase backtracking (shown in `test_cone_convergence_study.py` to still fail to converge on some scenarios); `0.1` was sufficient for every case tested so far, converging in single digits to ~20 iterations regardless of `K` or how far the optimum sits outside the simplex. |

See `eg_extrapolation.py`'s module docstring, `test_eg_extrapolation_synthetic.py`, and the paper's Appendix C for the full derivation and what has (and hasn't) been validated so far.

## 5. Cost and memory notes

- **Hierarchical routing** loads/scores every row in the bank once
  (`compute_bank_batch_losses`, one checkpoint on GPU at a time, deleted
  after scoring), then keeps only `H` selected source checkpoints + `theta_0`
  as CPU state dicts for the optimization step. This is the cheaper of the
  two candidate-holding patterns.
- **Flat routing** is now Top-H screened the same way as Hierarchical
  (`select_routing_candidates` + `build_candidate_state_dicts`), so it no
  longer holds the whole bank's state dicts in memory at once — this
  changed from the original design, where it interpolated over every
  checkpoint. Its memory/compute profile is now close to Hierarchical's,
  minus the EG optimization loop.
- **Hard routing** is the cheapest: it scores every row (same pass as the
  others) but only ever loads the single winning checkpoint into memory as
  the returned model.
- Wall-clock time in practice is dominated by checkpoint download/load, not
  by the routing math itself — a run against 6 `gpt2-medium` checkpoints
  (2 sources × 3 λ) with a small test batch took about 3 minutes end to end
  on a single GTX 1080 Ti, most of it spent downloading/loading weights.

## 6. Reading the `info` dict

**Hierarchical** (`run_hierarchical_routing`):
- `candidate_names`: `["theta_0", <selected source names>...]`
- `weights`: optimized interpolation weight per candidate
- `batch_loss`: the composed model's loss on `B`
- `vertex_losses`: loss of each candidate used alone (i.e. weight=1 on just that one) — compare against `batch_loss` to see whether interpolation actually helped
- `source_relevance`: one row per source in the full bank, with its best λ and score (before Top-H filtering)
- `selected_sources`: which sources survived Top-H

**Hard** (`run_hard_routing`):
- `selected`: name of the winning checkpoint (or `"theta_0"`)
- `batch_loss`: its loss on `B`
- `all_losses`: every bank member's loss, for comparison

**Flat** (`run_flat_routing`):
- `candidate_names`: `["theta_0", <Top-H selected source names>...]` (not the full bank — see §5)
- `weights`: softmax weight per candidate
- `candidate_losses`: each selected candidate's raw loss (pre-softmax)
- `source_relevance`, `selected_sources`, `selected_lambdas`, `selected_candidates`: same meaning as Hierarchical's, since both share `select_routing_candidates`
- `tau`, `H`: the values used

**Cone extrapolation** (`run_cone_routing`, proposed extension):
- Everything Hierarchical's `info` has, plus:
- `used_cone`: whether the cone search actually beat the plain-simplex fallback (`False` means the returned model/loss is identical to what `run_hierarchical_routing` would have given — the cone found nothing better)
- `anchor_name`, `anchor_mode`: which candidate was relaxed, and by which rule
- `per_anchor_best_loss`: with `anchor_mode="all"`, the best loss found *per candidate tried* — the direct way to see whether a different anchor choice would have done better
- `cone_lr`, `max_log_step`: the actual values used after resolving the `None` defaults (see above) — always check these rather than assuming, especially `cone_lr` given it's not portable across scenarios with different gradient scales
- `baseline_loss`: the plain Hierarchical Routing loss the cone result was compared against (should equal `run_hierarchical_routing`'s own `batch_loss` on the same inputs)
