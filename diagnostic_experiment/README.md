# diagnostic_experiment

This is the real research pipeline of shiftlab-llm: GPU-heavy, real GPT-2-family
models, real Pile-proxy corpora. It is **not** connected to the toy
`shift`/`adapt`/`eval` framework described in the top-level README
(`src/shiftlab/core`, `shift/`, `adapt/`) — nothing here imports from
`shiftlab.core`, and every script here has its own `argparse` + YAML config
loader instead of going through `shiftlab.cli`. That separation is intentional
for now (see the top-level `README.md`); this file documents what exists on
this side.

Two research threads live here, sharing datasets and infrastructure but
answering different questions.

## Thread A — Difficulty shift (curriculum robustness)

Question: if a model is fine-tuned on a mixture of "easy" and "hard" examples
(ranked by the base model's own per-sample loss), how does its robustness to
harder test distributions change? A noise-injection (OCR-corruption) variant
studies the same question under synthetic character-level corruption instead
of loss-based difficulty.

There are **two independent implementations** of this idea. They are not
interchangeable — check which config schema you're using before assuming a
script will run against a given YAML.

- **`difficulty_shift/diagnostic_experiment.py`** — self-contained, single-file,
  ERM-only. Two-way easy/hard split. Config keys: `diagnostic.alpha`,
  `beta_train`, `betas_test`, `histogram_betas`. This is what the
  `configs/diagnostic/{wikitext,wikisource,histtext}/*.yaml` configs target
  (verified against `wikitext/distilgpt2_difficulty_shift.yaml`).
- **`training_pipeline.py` + `utils.py`** — newer, more general. Three-way
  easy/medium/hard split (`utils.split_easy_medium_hard`), multiple training
  `methods` in one run (ERM plus arbitrary named KL-DRO variants at different
  λ), and an OCR-noise-aware close/mid/far shift-severity evaluation
  (`utils.create_shift_datasets`). Config keys: `methods:` (a dict of named
  training runs) plus a differently-shaped `diagnostic:` block
  (`alpha_easy`, `alpha_hard`, `train_alpha`, `train_beta`, `close_noise`,
  `mid_noise`, `far_noise`, ...). This is what
  `configs/diagnostic/PG-19/distilgpt2_ocr_shift.yaml` targets (verified).
  Entry point: **`difficulty_shift/difficulty_shift.py`**, which — despite
  living next to `diagnostic_experiment.py` — imports `run_training` from
  `training_pipeline`, *not* from its sibling file. Two same-named
  `run_training` functions exist in this subtree with different signatures;
  make sure you're calling the one the config actually matches.

`experiments.py` (top-level, not inside `difficulty_shift/`) is effectively a
near-duplicate of `difficulty_shift/difficulty_shift.py` — same
`training_pipeline.run_training` call — plus commented-out calls to
`run_training_rho` / `run_training_lambd` / `run_training_lambd2`, which no
longer exist as live functions (see "Known dead code" below).

**Stale configs**: `configs/diagnostic/PG-19/distilgpt2_experiment2_phase1.yaml`
and `_phase2.yaml` use keys (`alpha_train`/`beta_train` at top level,
`training.lambdas: {small, medium, large}`) that match neither current
implementation — they match the commented-out pre-refactor functions. Running
them against the live `training_pipeline.py` will raise a `KeyError` on
`config["methods"]`.

## Thread B — Shift-aware model bank + Algorithm 2

Question: instead of retraining a bespoke robust model for every deployment
domain, can you (1) pre-train a small *bank* of models at different
robustness levels, (2) cheaply measure how "shifted" an incoming deployment
batch is, and (3) select/interpolate the right bank model for it without ever
training on that deployment domain?

**Core mechanism — KL-DRO and λ**: bank models are trained with a
distributionally-robust objective that upweights high-loss examples via
exponential tilting (`utils.KL_DRO_one_epoch`). **λ** is the robustness/
temperature coefficient (λ=0 ≈ plain ERM; larger λ trains against an
increasingly tail-focused reweighting of the data). **ρ** is the KL
divergence of that reweighting from uniform — a scalar summary of "how much
λ is tilting training," and it is reused more generally as a shift metric
between any two token distributions (see `shift_measurement.compute_kl` /
`compute_batch_token_distribution`).

**The pipeline, file by file, in the order data flows through them:**

1. **`models_bank.py`** — defines the 3 canonical *source* datasets
   (`datasets_config`: FreeLaw, PubMed Central, ArXiv — real Pile-proxy HF
   datasets, not the datasets used in Thread A). For each source, it
   precomputes and caches: the exact token distribution over a fixed token
   budget, per-sample pretrained losses, and the λ-ρ curve
   (`lambda_window.compute_lambda_rho_curve`) over a fine grid
   (`configs/diagnostic/models_bank.yaml`'s `lambda_grid`, 17 points). It then
   trains one `gpt2-medium` KL-DRO checkpoint per (source × λ) pair, where λ
   comes from the coarser `lambdas` list (9 values, 0.0–1.0) — these 9 are the
   actual trained bracket endpoints Algorithm 2 interpolates between. Output:
   `model_bank_metadata.csv`, recording each checkpoint's path, λ, source
   dataset, and pointers to that source's distribution/loss/curve artifacts.

2. **`erm_baselines.py`** — trains one plain-ERM (λ=0, ordinary MLE)
   `gpt2-medium` oracle **per deployment dataset** (PG-19, PubMed Abstracts,
   GitHub, Ubuntu IRC, ...) — note: *deployment*-side datasets, disjoint from
   the 3 bank sources above. These represent "what if we fully fine-tuned on
   the target domain" upper bounds. Output:
   `outputs/erm_baselines/oracle_erm_models_metadata.csv`.

3. **`algorithm_2.py`** — the decision procedure (`run_algorithm_2`), given a
   deployment batch B:
   - `select_closest_source`: token-KL distance from B to each bank source's
     precomputed distribution; pick the closest (`j* = argmin_j KL(p_B‖p_Dj)`).
   - `select_lambda_hat_from_curve`: look up that source's λ-ρ curve, find the
     λ whose curve-ρ is closest to B's measured KL.
   - `select_lambda_interval_from_trained_grid` /
     `select_model_interval_from_bank`: find the two *actually-trained* λ
     values (from the 9-point grid) bracketing λ̂.
   - `interpolation_utils.interpolate_models(beta=0.5)`: linearly interpolate
     the two endpoint checkpoints' weights to produce the deployed model
     θ_B, plus an `info` dict recording the full decision trail.

4. **`algo2_real_test.py`** — the large-scale validation harness. Loads real
   deployment datasets, splits each into selection batches (used only to
   drive Algorithm 2's decision) and a disjoint held-out evaluation set. For
   each selection batch it computes: Algorithm 2's interpolated model's loss,
   the ERM oracle's loss (from step 2), and a brute-force best-possible
   midpoint interpolation across all source×λ-interval combinations
   (`rank_interpolated_models`) — then reports "regret" (Algorithm 2 vs.
   oracle, vs. best-brute-force). This is the paper-style empirical
   validation of whether the cheap heuristic in step 3 is actually a good
   substitute for retraining.

5. **`hierarchical_routing.py`** and **`routing_baselines.py`** — the three
   deployment-time routing strategies from the paper's Section 4/5.1
   (Hierarchical (ours), Hard, Flat soft), runnable against either a local
   bank or the Hub-hosted `alignment-decision-lab/robustness-model-bank`.
   See **[ROUTING_README.md](ROUTING_README.md)** for a full usage guide
   (setup, HF auth gotchas, batch construction, config knobs, worked
   examples).

**Supporting/diagnostic files** (not in the main data-flow path, used to
validate pieces of the above in isolation):
- `shift_measurement.py` — shared metrics library (token-KL, embedding L2,
  diagonal/PCA Gaussian-KL) plus three standalone experiments runnable via
  `python shift_measurement.py`: noise-floor calibration, raw near/far shift
  comparison, and a check of how well each metric reproduces a literature
  reference shift ratio from an external Pile study. This is where the choice
  of "token-KL as the production metric" gets its empirical justification.
- `lambda_window.py` — the λ-ρ curve math (`compute_adversarial_weights`,
  `compute_kl_to_uniform`, `compute_lambda_rho_curve`) in isolation, plus a
  standalone script projecting fixed OWT2/PileCC/PubMed batches onto their
  own λ-ρ curves.
- `lambda_calibration.py`, `lambda_analysis_utils.py`, `lambda_model_analysis.py`
  — earlier/adjacent λ-focused diagnostics and plotting utilities.
- `interpolation_utils.py` — the actual weight-interpolation implementation
  (`interpolate_models`) plus helpers for training the required λ-bracket
  models and running/aggregating interpolation sweeps.
- `robustness_path.py` — asks a narrower geometric question than
  `algo2_real_test.py`: is linear weight interpolation between two λ-trained
  endpoints actually a good stand-in for a model *directly trained* at the
  intermediate λ? Trains the bracket endpoints, interpolates at several
  targets/widths, and compares against real intermediate-λ checkpoints using
  parameter-space distance, curvature ratio, and predictive-KL diagnostics.

## Known dead code / naming traps

- **`utils_old.py`** (976 lines) is fully dead — nothing imports it
  (`grep -rn "utils_old"` has no hits outside the file itself). It's the
  pre-refactor version of `utils.py`, containing `train_mixture`,
  `train_mixture_rho`, `train_mixture_lambd`, `train_mixture_lambd2`, etc.
- The top ~360 lines of `training_pipeline.py` are the commented-out
  pre-refactor `run_training_rho` / `run_training_lambd` / `run_training_lambd2`
  functions, corresponding 1:1 to the dead code in `utils_old.py`.
- `gpt2.py` (`src/shiftlab/train/gpt2.py`) is an unrelated 5-line smoke-test
  stub — loads `gpt2`, prints its type. Not used by anything here.
- Dataset config dicts (`owt2_config`, `pilecc_config`, `pubmed_config`, the
  bank `datasets_config`, the ERM `datasets_config`) are re-declared
  separately in `shift_measurement.py`, `lambda_calibration.py`,
  `lambda_window.py`, `models_bank.py`, and `erm_baselines.py` rather than
  sharing one source of truth. Note in particular that "which datasets count
  as sources vs. deployment domains" differs by file — `models_bank.py` uses
  3 sources (FreeLaw, PubMed Central, ArXiv); `erm_baselines.py` /
  `algo2_real_test.py` use a disjoint set of deployment domains (PG-19,
  PubMed Abstracts, GitHub, Ubuntu IRC, ...). Any other ad hoc script
  (e.g. exploratory curve-generation scripts not currently checked into this
  directory) may use yet another combination — always check the specific
  `datasets_config` in the file you're reading rather than assuming it
  matches another script's.

## Open question: forgetting / overfitting during bank and oracle training

Not yet investigated in this codebase, but worth flagging here since it
bears directly on both Thread A and Thread B: every training loop in this
subsystem (`train_one_bank_model`, `train_one_erm_model`, the difficulty-shift
`run_training`s) fine-tunes a pretrained GPT-2 checkpoint on a narrow,
bounded-size domain slice for multiple epochs, with nothing counterbalancing
drift away from the base model's general competence. `papers/Scaling-Laws-for-Forgetting-during-Finetuning.pdf`
(Bethune et al., ICML 2025) studies exactly this mechanism — fine-tuning on a
narrow target domain causes both overfitting (target-domain validation loss
eventually rises) and forgetting (loss on the original pretraining
distribution rises) — and shows that injecting as little as ~1% of
non-repeated, general-domain "anchor" data into the training mixture largely
prevents both, at negligible cost to target-domain performance.

A preliminary diagnostic (`curves/gpt2-owt2.png`) fine-tuning on four
different sources (Wikipedia, FreeLaw, ArXiv, OpenWebText2) for 4 epochs each
shows exactly this signature: loss on domains other than the training source
climbs over epochs in every case (forgetting), and in the OpenWebText2 run
specifically, even the source's own held-out loss climbs by epoch 4
(overfitting to a training slice too small for 4 full passes).

This raises a real question for Thread B in particular: the ERM oracle
models in `erm_baselines.py` are used as the "gold standard" that Algorithm
2 is compared against via oracle-regret metrics in `algo2_real_test.py`. If
those oracles are themselves already degraded by uncorrected forgetting/
overfitting during their own training, the regret numbers may not mean what
they're currently assumed to mean. Worth measuring (loss on a fixed
general-domain anchor set, before/after bank and oracle training, across the
λ grid) before deciding whether to integrate an injection-based fix — not yet
done as of this writing.

## Directory map

```
diagnostic_experiment/
├── experiments.py                    # near-duplicate entry point (see Thread A note)
├── training_pipeline.py              # active run_training (3-way mixture, methods dict) + dead pre-refactor code
├── utils.py                          # active shared helpers, training loops, plotting
├── utils_old.py                      # DEAD — pre-refactor version of utils.py
├── difficulty_shift/
│   ├── diagnostic_experiment.py      # self-contained ERM-only difficulty shift (targets wikitext/wikisource/histtext configs)
│   └── difficulty_shift.py           # entry point -> training_pipeline.run_training (targets PG-19 OCR config)
├── models_bank.py                    # Thread B: build the (source x lambda) checkpoint bank
├── erm_baselines.py                  # Thread B: per-deployment-domain ERM oracle models
├── algorithm_2.py                    # Thread B: the selection + interpolation decision procedure
├── algo2_real_test.py                # Thread B: end-to-end validation harness against oracle/brute-force baselines
├── hierarchical_routing.py           # Thread B: Hierarchical (ours) routing strategy -- see ROUTING_README.md
├── routing_baselines.py              # Thread B: Hard + Flat soft routing strategies -- see ROUTING_README.md
├── ROUTING_README.md                 # usage guide for the three routing strategies above
├── shift_measurement.py              # shared shift-metric library + standalone metric-validation experiments
├── lambda_window.py                  # lambda-rho curve math, standalone projection script
├── lambda_calibration.py             # adjacent lambda-focused diagnostics
├── lambda_analysis_utils.py          # adjacent lambda-focused diagnostics
├── lambda_model_analysis.py          # adjacent lambda-focused diagnostics
├── interpolation_utils.py            # weight interpolation implementation
└── robustness_path.py                # geometric validation of interpolation vs. directly-trained intermediate-lambda models
```
