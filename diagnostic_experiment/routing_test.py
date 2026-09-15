import os
import json
import math
import time
import copy
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling

import utils
import hierarchical_routing as hr
import routing_baselines as rb
import routing_PCA as rpca
import online_tent as ot
from shiftlab.data.load_datasets import load_dataset_from_subconfig


BANK_REPO_ID = "alignment-decision-lab/robustness-model-bank"
OUTPUT_ROOT = "outputs/experimental_pipeline"

PRIMARY_METHODS = [
    "pretrained", "best_single_ft", "mixed_ft", "static_tent_best_single", "tent_best_single",
    "hard", "flat", "static_hierarchical", "hierarchical", "tent_hierarchical", "oracle",
]
NONSTATIONARY_METHODS = [
    "pretrained", "best_single_ft", "mixed_ft", "static_tent_best_single", "tent_best_single",
    "hard", "flat", "static_hierarchical", "hierarchical", "tent_hierarchical",
]
METHOD_LABELS = {
    "pretrained": "Pretrained",
    "best_single_ft": "Best Single FT",
    "mixed_ft": "Mixed-source FT",
    "static_tent_best_single": "Static TENT (Best Single-FT)",
    "tent_best_single": "TENT (Best Single-FT)",
    "hard": "Hard Routing",
    "flat": "Flat Routing",
    "static_hierarchical": "Static Hierarchical Routing",
    "hierarchical": "Hierarchical Routing",
    "tent_hierarchical": "TENT (Hierarchical Routing)",
    "oracle": "Target-FT Oracle",
}


# ============================================================
# CONFIGURATION AND GENERAL HELPERS
# ============================================================

def get_model_config(EXPERIMENT_CONFIG, MODEL_REGISTRY):
    key = EXPERIMENT_CONFIG["model"]
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{key}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]


def get_model_name(EXPERIMENT_CONFIG, MODEL_REGISTRY):
    return get_model_config(EXPERIMENT_CONFIG, MODEL_REGISTRY)["model_name"]


def get_bank_prefix(EXPERIMENT_CONFIG, MODEL_REGISTRY):
    return get_model_config(EXPERIMENT_CONFIG, MODEL_REGISTRY)["bank_prefix"]


def get_hierarchical_config(EXPERIMENT_CONFIG, for_tent=False):
    cfg = dict(EXPERIMENT_CONFIG["hierarchical"])
    if for_tent:
        cfg["num_iters"] = EXPERIMENT_CONFIG.get("tent_hierarchical", {}).get("num_iters", cfg["num_iters"])
    return cfg


def get_mixed_ft_config(EXPERIMENT_CONFIG):
    return EXPERIMENT_CONFIG.get("mixed_ft")


def get_mixed_ft_subfolder(EXPERIMENT_CONFIG):
    mixed_ft = get_mixed_ft_config(EXPERIMENT_CONFIG)
    return None if mixed_ft is None else mixed_ft.get("subfolder")


def mixed_ft_source_name(source):
    return source.replace(" ", "_").replace("/", "_")


def validate_mixed_ft_config(EXPERIMENT_CONFIG, MODEL_REGISTRY):
    mixed_ft_methods = {"mixed_ft"}

    mixed_ft_required = False
    for protocol in ["primary_episodic", "nonstationary_stream"]:
        cfg = EXPERIMENT_CONFIG.get(protocol, {})
        if not cfg.get("enabled", False):
            continue
        methods = get_protocol_methods(EXPERIMENT_CONFIG, protocol)
        if any(method in mixed_ft_methods for method in methods):
            mixed_ft_required = True
            break

    if not mixed_ft_required:
        return

    mixed_ft = get_mixed_ft_config(EXPERIMENT_CONFIG)
    if mixed_ft is None:
        raise ValueError(
            "This experiment uses Mixed-source FT, but no 'mixed_ft' configuration was provided in EXPERIMENT_CONFIG."
        )

    subfolder = mixed_ft.get("subfolder")
    if not subfolder:
        raise ValueError("EXPERIMENT_CONFIG['mixed_ft']['subfolder'] is missing.")

    normalized_subfolder = subfolder.strip("/")
    parts = normalized_subfolder.split("/")

    if len(parts) < 3 or parts[-1] != "model":
        raise ValueError(
            "Invalid Mixed-FT subfolder. Expected a path of the form "
            "'<bank_prefix>/Mixed_FT_<sources>/model'. "
            f"Got: {subfolder}"
        )

    expected_prefix = get_bank_prefix(EXPERIMENT_CONFIG, MODEL_REGISTRY)
    actual_prefix = parts[0]
    if actual_prefix != expected_prefix:
        raise ValueError(
            "Mixed-FT checkpoint does not match the experiment model.\n"
            f"Experiment model: {EXPERIMENT_CONFIG['model']}\n"
            f"Expected bank prefix: {expected_prefix}\n"
            f"Configured bank prefix: {actual_prefix}\n"
            f"Mixed-FT subfolder: {subfolder}"
        )

    expected_mixed_name = "Mixed_FT_" + "_".join(
        mixed_ft_source_name(source) for source in EXPERIMENT_CONFIG["sources"]
    )
    actual_mixed_name = parts[-2]

    if actual_mixed_name != expected_mixed_name:
        raise ValueError(
            "Mixed-FT checkpoint does not match the experiment sources.\n"
            f"Experiment sources: {EXPERIMENT_CONFIG['sources']}\n"
            f"Expected Mixed-FT: {expected_mixed_name}\n"
            f"Configured Mixed-FT: {actual_mixed_name}\n"
            f"Mixed-FT subfolder: {subfolder}"
        )


def get_output_dir(EXPERIMENT_CONFIG):
    return os.path.join(OUTPUT_ROOT, EXPERIMENT_CONFIG["experiment_name"])


def protocol_dir(EXPERIMENT_CONFIG, protocol):
    return os.path.join(get_output_dir(EXPERIMENT_CONFIG), protocol)


def safe_name(name):
    return name.replace(" ", "_").replace("/", "_")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def save_status(output_dir, status, protocol=None, dataset=None, batch=None, error=None):
    payload = {"status": status, "protocol": protocol, "dataset": dataset, "batch": batch, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if error is not None:
        payload["error"] = str(error)
    save_json(os.path.join(output_dir, "status.json"), payload)


def get_protocol_methods(EXPERIMENT_CONFIG, protocol):
    cfg = EXPERIMENT_CONFIG[protocol]
    default = PRIMARY_METHODS if protocol == "primary_episodic" else NONSTATIONARY_METHODS
    return list(cfg.get("methods", default))


def validate_config(EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY):
    if EXPERIMENT_CONFIG.get("model") not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {EXPERIMENT_CONFIG.get('model')}")
    if not EXPERIMENT_CONFIG.get("sources"):
        raise ValueError("At least one source dataset is required.")
    if not EXPERIMENT_CONFIG.get("deployments"):
        raise ValueError("At least one deployment dataset is required.")
    for name in EXPERIMENT_CONFIG["sources"] + EXPERIMENT_CONFIG["deployments"]:
        if name not in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' is not in DATASET_REGISTRY.")
    if EXPERIMENT_CONFIG.get("context_length", 0) <= 1 or EXPERIMENT_CONFIG.get("batch_size", 0) <= 0:
        raise ValueError("context_length must be > 1 and batch_size must be > 0.")
    hier = EXPERIMENT_CONFIG.get("hierarchical", {})
    if hier.get("H", 0) <= 0 or hier.get("num_iters", 0) <= 0:
        raise ValueError("hierarchical.H and hierarchical.num_iters must be > 0.")
    tent_hier = EXPERIMENT_CONFIG.get("tent_hierarchical", {})
    if "num_iters" in tent_hier and tent_hier["num_iters"] <= 0:
        raise ValueError("tent_hierarchical.num_iters must be > 0.")
    tent = EXPERIMENT_CONFIG.get("tent", {})
    if tent.get("num_steps", 0) <= 0 or tent.get("lr", 0) <= 0:
        raise ValueError("tent.num_steps and tent.lr must be > 0.")

    valid = set(PRIMARY_METHODS + NONSTATIONARY_METHODS)
    for protocol in ["primary_episodic", "nonstationary_stream"]:
        cfg = EXPERIMENT_CONFIG.get(protocol, {})
        if not cfg.get("enabled", False):
            continue
        methods = get_protocol_methods(EXPERIMENT_CONFIG, protocol)
        unknown = set(methods) - valid
        if unknown:
            raise ValueError(f"Unknown methods in {protocol}: {sorted(unknown)}")
        n = cfg.get("num_batches") if protocol == "primary_episodic" else cfg.get("num_online_batches")
        if n is None or int(n) <= 0:
            raise ValueError(f"{protocol} requires a positive number of batches.")
        offset = int(cfg.get("deployment_offset_tokens", 512_000 if protocol == "primary_episodic" else 0))
        if offset % EXPERIMENT_CONFIG["context_length"] != 0:
            raise ValueError(f"{protocol}.deployment_offset_tokens must be divisible by context_length.")
        if protocol == "nonstationary_stream":
            overlap = set(EXPERIMENT_CONFIG["sources"]) & set(EXPERIMENT_CONFIG["deployments"])
            if overlap:
                raise ValueError(
                    "Non-stationary deployment datasets must be different from source datasets. "
                    f"Overlap found: {sorted(overlap)}"
                )
            if "oracle" in methods:
                raise ValueError("Target-FT Oracle is intentionally excluded from the non-stationary protocol.")

    validate_mixed_ft_config(EXPERIMENT_CONFIG, MODEL_REGISTRY)


def save_experiment_config(output_dir, EXPERIMENT_CONFIG, MODEL_REGISTRY):
    payload = {"experiment": EXPERIMENT_CONFIG, "model_config": get_model_config(EXPERIMENT_CONFIG, MODEL_REGISTRY), "bank_repo_id": BANK_REPO_ID}
    save_json(os.path.join(output_dir, "config.json"), payload)


# ============================================================
# MODEL BANK AND CHECKPOINT LOADING
# ============================================================

def build_model_bank_metadata(output_dir, EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY):
    model_cfg = get_model_config(EXPERIMENT_CONFIG, MODEL_REGISTRY)
    rows = []
    for dataset_name in EXPERIMENT_CONFIG["sources"]:
        bank_name = DATASET_REGISTRY[dataset_name]["bank_name"]
        for lambda_str in model_cfg["lambdas"]:
            rows.append({
                "dataset_name": dataset_name,
                "bank_name": bank_name,
                "lambda": float(lambda_str),
                "lambda_str": lambda_str,
                "subfolder": f"{model_cfg['bank_prefix']}/{bank_name}/lambda_{lambda_str}/model",
                "status": "trained",
            })
    bank_df = pd.DataFrame(rows)
    metadata_path = os.path.join(output_dir, "model_bank_metadata.csv")
    bank_df.to_csv(metadata_path, index=False)
    print(f"Built metadata for {len(bank_df)} checkpoints ({len(EXPERIMENT_CONFIG['sources'])} sources x {len(model_cfg['lambdas'])} lambdas).", flush=True)
    return metadata_path, bank_df


def load_hub_model(subfolder, device):
    model = AutoModelForCausalLM.from_pretrained(BANK_REPO_ID, subfolder=subfolder).to(device)
    model.eval()
    return model


def load_pretrained_model(device, EXPERIMENT_CONFIG, MODEL_REGISTRY):
    cfg = get_model_config(EXPERIMENT_CONFIG, MODEL_REGISTRY)
    subfolder = cfg.get("pretrained_subfolder")
    model = AutoModelForCausalLM.from_pretrained(BANK_REPO_ID, subfolder=subfolder).to(device) if subfolder else AutoModelForCausalLM.from_pretrained(cfg["model_name"]).to(device)
    model.eval()
    return model


def clear_model(model, device):
    if model is not None:
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def get_oracle_subfolder(dataset_name, EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY):
    oracle_name = DATASET_REGISTRY[dataset_name].get("oracle_name")
    if oracle_name is None:
        return None
    return f"{get_bank_prefix(EXPERIMENT_CONFIG, MODEL_REGISTRY)}/{oracle_name}/lambda_0.0/model"


# ============================================================
# DEPLOYMENT DATA
# ============================================================

def load_deployment_batches(dataset_name, tokenizer, num_batches, offset_tokens, EXPERIMENT_CONFIG, DATASET_REGISTRY):
    """Load consecutive deployment batches after an explicit token offset."""
    context_length, batch_size = EXPERIMENT_CONFIG["context_length"], EXPERIMENT_CONFIG["batch_size"]
    dataset_config = DATASET_REGISTRY[dataset_name]["dataset_config"]
    deployment_tokens = num_batches * batch_size * context_length
    total_max_tokens = offset_tokens + deployment_tokens
    print(f"\nLoading {dataset_name}: offset={offset_tokens:,} tokens, batches={num_batches}.", flush=True)

    training_config = {"dataset_offset": 0, "context_length": context_length, "max_tokens": total_max_tokens}
    dataset = load_dataset_from_subconfig(dataset_config=dataset_config, training_config=training_config)
    tokenization_config = {
        "dataset": {"text_column": dataset_config.get("text_column", "text")},
        "training": {"context_length": context_length, "max_tokens": total_max_tokens},
    }
    tokenized_dataset, dataset_stats = utils.tokenize_and_group_with_token_budget(dataset=dataset, tokenizer=tokenizer, config=tokenization_config)

    offset_blocks = offset_tokens // context_length
    deployment_blocks = num_batches * batch_size
    required_blocks = offset_blocks + deployment_blocks
    if len(tokenized_dataset) < required_blocks:
        raise RuntimeError(
            f"{dataset_name}: only {len(tokenized_dataset)} token blocks obtained, but {required_blocks} are required "
            f"({offset_blocks} skipped + {deployment_blocks} deployment)."
        )

    selected = tokenized_dataset.select(range(offset_blocks, required_blocks))
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    batches = list(DataLoader(selected, batch_size=batch_size, shuffle=False, collate_fn=collator))
    if len(batches) != num_batches:
        raise RuntimeError(f"{dataset_name}: expected {num_batches} batches, obtained {len(batches)}.")
    return batches, dataset_stats


def build_balanced_stream(deployments, num_online_batches, seed):
    """Balanced + random + reproducible domain sequence for the non-stationary stream."""
    if not deployments:
        raise ValueError("At least one non-stationary deployment dataset is required.")
    base, remainder = divmod(num_online_batches, len(deployments))
    sequence = []
    for name in deployments:
        sequence.extend([name] * base)
    rng = random.Random(seed)
    extras = list(deployments)
    rng.shuffle(extras)
    sequence.extend(extras[:remainder])
    rng.shuffle(sequence)
    return sequence


# ============================================================
# EVALUATION AND METRICS
# ============================================================

def evaluate_batch(model, batch, device):
    return utils.evaluation(model, [batch], device)


def loss_to_perplexity(loss):
    try:
        return math.exp(float(loss))
    except (OverflowError, ValueError, TypeError):
        return float("inf")


def oracle_gap_closed(loss_method, loss_pretrained, loss_oracle):
    values = [loss_method, loss_pretrained, loss_oracle]
    if any(v is None or pd.isna(v) for v in values):
        return float("nan")
    denominator = float(loss_pretrained) - float(loss_oracle)
    if abs(denominator) < 1e-12:
        return float("nan")
    return (float(loss_pretrained) - float(loss_method)) * 100.0 / denominator


def relative_loss_improvement(loss_method, loss_pretrained):
    """Percentage loss reduction relative to Pretrained on the same batch."""
    values = [loss_method, loss_pretrained]
    if any(v is None or pd.isna(v) for v in values) or abs(float(loss_pretrained)) < 1e-12:
        return float("nan")
    return (float(loss_pretrained) - float(loss_method)) * 100.0 / float(loss_pretrained)


def add_relative_loss_improvement(method_results):
    pre = next((r for r in method_results if r["method"] == "Pretrained"), None)
    pre_loss = pre.get("loss") if pre else None
    for result in method_results:
        result["relative_loss_improvement_pct"] = relative_loss_improvement(result.get("loss"), pre_loss)


def num_batch_tokens(batch):
    if "attention_mask" in batch:
        return int(batch["attention_mask"].sum().item())
    return int(batch["input_ids"].numel())


# ============================================================
# STATIC BASELINES
# ============================================================

def run_pretrained(batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY):
    start = time.time()
    model = load_pretrained_model(device, EXPERIMENT_CONFIG, MODEL_REGISTRY)
    loss, _, _, num_tokens = evaluate_batch(model, batch, device)
    total_time = time.time() - start
    clear_model(model, device)
    return {"method": "Pretrained", "loss": float(loss), "perplexity": loss_to_perplexity(loss), "total_time_sec": total_time, "num_tokens": num_tokens}


def select_best_single_ft(bank_df, first_batch, device, EXPERIMENT_CONFIG):
    """Select one source ERM on the first deployment batch, then keep it fixed."""
    erm_df = bank_df[bank_df["lambda"].astype(float) == 0.0].copy()
    expected = len(EXPERIMENT_CONFIG["sources"])
    if len(erm_df) != expected:
        raise RuntimeError(f"Expected {expected} source ERM checkpoints, found {len(erm_df)}.")
    batch_device = utils.move_batch_to_device(first_batch, device)
    rows, start = [], time.time()
    for _, row in erm_df.iterrows():
        model = hr.load_bank_checkpoint(row, BANK_REPO_ID).to(device)
        model.eval()
        with torch.no_grad():
            loss = model(**batch_device).loss.item()
        rows.append({"dataset_name": row["dataset_name"], "subfolder": row["subfolder"], "batch_loss": loss})
        clear_model(model, device)
    selection_time = time.time() - start
    scores_df = pd.DataFrame(rows)
    best = scores_df.loc[scores_df["batch_loss"].idxmin()]
    selected = {"dataset_name": best["dataset_name"], "subfolder": best["subfolder"], "selection_loss": float(best["batch_loss"])}
    return selected, selection_time, scores_df


def run_best_single_ft(batch, selected_model, selection_time, batch_id, device):
    start = time.time()
    model = load_hub_model(selected_model["subfolder"], device)
    loss, _, _, num_tokens = evaluate_batch(model, batch, device)
    evaluation_time = time.time() - start
    clear_model(model, device)
    return {
        "method": "Best Single FT", "loss": float(loss), "perplexity": loss_to_perplexity(loss),
        "total_time_sec": evaluation_time + (selection_time if batch_id == 0 else 0.0), "num_tokens": num_tokens,
        "selected_source": selected_model["dataset_name"], "selection_loss": selected_model["selection_loss"],
        "selection_recomputed": batch_id == 0,
    }


def run_mixed_ft(batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY):
    subfolder = get_mixed_ft_subfolder(EXPERIMENT_CONFIG)
    if subfolder is None:
        return {"method": "Mixed-source FT", "loss": float("nan"), "perplexity": float("nan"), "total_time_sec": float("nan"), "num_tokens": None, "status": "mixed_ft_not_configured"}
    start = time.time()
    model = load_hub_model(subfolder, device)
    loss, _, _, num_tokens = evaluate_batch(model, batch, device)
    total_time = time.time() - start
    clear_model(model, device)
    return {"method": "Mixed-source FT", "loss": float(loss), "perplexity": loss_to_perplexity(loss), "total_time_sec": total_time, "num_tokens": num_tokens}


# ============================================================
# TENT METHODS: SAME BATCH FOR ADAPTATION AND EVALUATION
# ============================================================
# online_tent.py only implements the adaptation mechanism. This runner chooses
# the initialization model and accounts for its deployment cost. Best Single-FT
# is selected once on B1; Hierarchical Routing is also solved once on B1 for
# TENT (Hierarchical Routing). Both TENT variants then carry their adapted
# LayerNorm parameters and optimizer state through subsequent batches.
# Static TENT performs the same Best Single-FT -> TENT operation on B1, then
# freezes the resulting model for B2...BT.


def init_tent_from_best_single(selected_model, selection_time, device, EXPERIMENT_CONFIG):
    start = time.time()
    model = load_hub_model(selected_model["subfolder"], device)
    state = ot.initialize_online_tent(model=model, device=device, config=EXPERIMENT_CONFIG["tent"])
    return state, selection_time + (time.time() - start)


def init_tent_from_model(model, initialization_time, device, EXPERIMENT_CONFIG):
    start = time.time()
    state = ot.initialize_online_tent(model=model, device=device, config=EXPERIMENT_CONFIG["tent"])
    return state, initialization_time + (time.time() - start)


def run_tent_batch(state, batch, batch_id, initialization_time, device, method_name):
    if state is None:
        raise RuntimeError(f"{method_name} state was not initialized on batch 1.")
    start = time.time()
    state, info = ot.run_online_tent_batch(state, batch, device)
    total_time = (time.time() - start) + (initialization_time if batch_id == 0 else 0.0)
    return {
        "method": method_name, "loss": float(info["loss_after"]), "perplexity": loss_to_perplexity(info["loss_after"]),
        "total_time_sec": total_time, "num_tokens": num_batch_tokens(batch), "loss_before": float(info["loss_before"]),
        "loss_after": float(info["loss_after"]), "loss_improvement": float(info["loss_improvement"]),
        "num_trainable_params": int(info["num_trainable_params"]), "num_adaptation_steps": int(info["num_adaptation_steps"]),
        "adaptation_lr": float(info["adaptation_lr"]), "state_carried": batch_id > 0,
        "num_batches_seen": int(info["num_batches_seen"]),
    }


def evaluate_static_tent(model, batch, device):
    start = time.time()
    loss, _, _, num_tokens = evaluate_batch(model, batch, device)
    return {
        "method": "Static TENT (Best Single-FT)", "loss": float(loss), "perplexity": loss_to_perplexity(loss),
        "total_time_sec": time.time() - start, "num_tokens": num_tokens, "state_carried": False, "adaptation_recomputed": False,
    }


def clear_online_tent(state, device):
    if state is not None:
        model = state.get("model")
        state.clear()
        clear_model(model, device)


# ============================================================
# ORACLE -- PRIMARY EPISODIC ONLY
# ============================================================

def run_oracle(dataset_name, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY):
    subfolder = get_oracle_subfolder(dataset_name, EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY)
    if subfolder is None:
        return {"method": "Target-FT Oracle", "loss": float("nan"), "perplexity": float("nan"), "total_time_sec": float("nan"), "num_tokens": None, "oracle_subfolder": None}
    start = time.time()
    model = load_hub_model(subfolder, device)
    loss, _, _, num_tokens = evaluate_batch(model, batch, device)
    total_time = time.time() - start
    clear_model(model, device)
    return {"method": "Target-FT Oracle", "loss": float(loss), "perplexity": loss_to_perplexity(loss), "total_time_sec": total_time, "num_tokens": num_tokens, "oracle_subfolder": subfolder}


# ============================================================
# ROUTING METHODS
# ============================================================

def compute_shared_routing_scores(bank_df, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY):
    """Physically score the bank once, while charging this cost to each routing method."""
    print("Scoring model bank once for all routing strategies...", flush=True)
    start = time.time()
    batch_device = utils.move_batch_to_device(batch, device)
    scored_bank_df = hr.compute_bank_batch_losses(bank_df=bank_df, batch=batch_device, device=device, bank_repo_id=BANK_REPO_ID)
    model_cfg = get_model_config(EXPERIMENT_CONFIG, MODEL_REGISTRY)
    pretrained_loss = rb.compute_pretrained_loss(
        pretrained_model_name=model_cfg["model_name"], pretrained_subfolder=model_cfg.get("pretrained_subfolder"),
        batch=batch_device, device=device, bank_repo_id=BANK_REPO_ID,
    )
    scoring_time = time.time() - start
    print(f"Shared scoring complete: {len(scored_bank_df)} bank checkpoints + theta_0 in {scoring_time:.1f}s.", flush=True)
    return scored_bank_df, pretrained_loss, scoring_time


def run_hard(metadata_path, bank_df, scored_bank_df, pretrained_loss, shared_scoring_time, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY):
    start = time.time()
    cfg = get_model_config(EXPERIMENT_CONFIG, MODEL_REGISTRY)
    model, info = rb.run_hard_routing(
        model_bank_metadata_path=metadata_path, batch=batch, pretrained_model_name=cfg["model_name"],
        pretrained_subfolder=cfg.get("pretrained_subfolder"), device=device, bank_repo_id=BANK_REPO_ID,
        bank_df=bank_df, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss,
    )
    method_time = time.time() - start
    loss = float(info["batch_loss"])
    result = {
        "method": "Hard Routing", "loss": loss, "perplexity": loss_to_perplexity(loss),
        "total_time_sec": shared_scoring_time + method_time, "method_time_sec": method_time,
        "shared_scoring_time_sec": shared_scoring_time, "selected_candidate": info.get("selected"),
    }
    clear_model(model, device)
    return result


def run_flat(metadata_path, bank_df, scored_bank_df, pretrained_loss, shared_scoring_time, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY):
    start = time.time()
    cfg = get_model_config(EXPERIMENT_CONFIG, MODEL_REGISTRY)
    model, info = rb.run_flat_routing(
        model_bank_metadata_path=metadata_path, batch=batch, pretrained_model_name=cfg["model_name"],
        pretrained_subfolder=cfg.get("pretrained_subfolder"), device=device, tau=EXPERIMENT_CONFIG["flat"]["tau"],
        H=EXPERIMENT_CONFIG["hierarchical"]["H"], bank_repo_id=BANK_REPO_ID, bank_df=bank_df,
        scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss,
    )
    loss, _, _, num_tokens = evaluate_batch(model, batch, device)
    method_time = time.time() - start
    result = {
        "method": "Flat Routing", "loss": float(loss), "perplexity": loss_to_perplexity(loss),
        "total_time_sec": shared_scoring_time + method_time, "method_time_sec": method_time,
        "shared_scoring_time_sec": shared_scoring_time, "num_tokens": num_tokens,
        "selected_candidates": str(info.get("candidate_names")), "weights": str(info.get("weights")),
    }
    clear_model(model, device)
    return result, info


def build_hierarchical_model(metadata_path, bank_df, scored_bank_df, pretrained_loss, shared_scoring_time, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY, hierarchical_config, method_name="Hierarchical Routing"):
    """Build a hierarchical composition and keep the returned model alive for optional static reuse."""
    start = time.time()
    cfg = get_model_config(EXPERIMENT_CONFIG, MODEL_REGISTRY)
    model, info = hr.run_hierarchical_routing(
        model_bank_metadata_path=metadata_path, batch=batch, pretrained_model_name=cfg["model_name"],
        pretrained_subfolder=cfg.get("pretrained_subfolder"), device=device, config=hierarchical_config,
        bank_repo_id=BANK_REPO_ID, bank_df=bank_df, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss,
    )
    method_time = time.time() - start
    loss = float(info["batch_loss"])
    result = {
        "method": method_name, "loss": loss, "perplexity": loss_to_perplexity(loss),
        "total_time_sec": shared_scoring_time + method_time, "method_time_sec": method_time,
        "shared_scoring_time_sec": shared_scoring_time, "selected_sources": str(info.get("selected_sources")),
        "selected_lambdas": str(info.get("selected_lambdas")), "weights": str(info.get("weights")),
        "best_start": info.get("best_start_name"), "best_iteration": info.get("best_iteration"),
        "best_is_vertex": info.get("best_is_vertex"), "routing_recomputed": True,
    }
    return model, result, info


def evaluate_static_hierarchical(model, batch, device):
    start = time.time()
    loss, _, _, num_tokens = evaluate_batch(model, batch, device)
    total_time = time.time() - start
    return {
        "method": "Static Hierarchical Routing", "loss": float(loss), "perplexity": loss_to_perplexity(loss),
        "total_time_sec": total_time, "num_tokens": num_tokens, "routing_recomputed": False,
    }


# ============================================================
# ROUTING DIAGNOSTICS AND OUTPUT HELPERS
# ============================================================

def save_method_details(batch_output_dir, method_key, result):
    pd.DataFrame([result]).to_csv(os.path.join(batch_output_dir, f"{method_key}_result.csv"), index=False)


def save_flat_details(batch_output_dir, info):
    names = info.get("candidate_names")
    if names is None:
        return
    losses, weights = info.get("candidate_losses"), info.get("weights")
    def value(container, name, idx):
        if isinstance(container, dict):
            return container.get(name)
        if isinstance(container, (list, tuple, np.ndarray)) and idx < len(container):
            return container[idx]
        return None
    rows = [{"candidate": name, "routing_loss": value(losses, name, i), "weight": value(weights, name, i)} for i, name in enumerate(names)]
    pd.DataFrame(rows).to_csv(os.path.join(batch_output_dir, "flat_weights.csv"), index=False)


def save_hierarchical_details(batch_output_dir, info):
    if info.get("source_relevance") is not None:
        pd.DataFrame(info["source_relevance"]).to_csv(os.path.join(batch_output_dir, "hierarchical_source_relevance.csv"), index=False)
    if info.get("optimization_trajectories") is not None:
        pd.DataFrame(info["optimization_trajectories"]).to_csv(os.path.join(batch_output_dir, "hierarchical_optimization_trajectories.csv"), index=False)


def run_pca_diagnostics(info, batch_output_dir, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY, label):
    """Preserve the original PCA + simplex/grid diagnostics, outside routing runtime."""
    if not EXPERIMENT_CONFIG.get("run_pca", True):
        return
    print(f"\nGenerating routing PCA for {label}...", flush=True)
    cfg = get_model_config(EXPERIMENT_CONFIG, MODEL_REGISTRY)
    rpca.run_routing_pca(
        info=info, output_dir=batch_output_dir, batch=batch, device=device, pretrained_model_name=cfg["model_name"],
        bank_repo_id=BANK_REPO_ID, pretrained_subfolder=cfg.get("pretrained_subfolder"), run_grid=True,
    )


def add_oracle_gap(method_results):
    pre = next((r for r in method_results if r["method"] == "Pretrained"), None)
    oracle = next((r for r in method_results if r["method"] == "Target-FT Oracle"), None)
    for result in method_results:
        result["oracle_gap_closed"] = oracle_gap_closed(result.get("loss"), pre.get("loss") if pre else None, oracle.get("loss") if oracle else None)


def build_base_row(EXPERIMENT_CONFIG, MODEL_REGISTRY, protocol, dataset, batch_id, offset_tokens):
    return {
        "experiment_name": EXPERIMENT_CONFIG["experiment_name"], "protocol": protocol, "model": EXPERIMENT_CONFIG["model"],
        "model_name": get_model_name(EXPERIMENT_CONFIG, MODEL_REGISTRY), "dataset": dataset, "batch_id": batch_id + 1,
        "context_length": EXPERIMENT_CONFIG["context_length"], "batch_size": EXPERIMENT_CONFIG["batch_size"],
        "deployment_offset_tokens": offset_tokens, "num_sources": len(EXPERIMENT_CONFIG["sources"]),
        "sources": str(EXPERIMENT_CONFIG["sources"]), "H": EXPERIMENT_CONFIG["hierarchical"]["H"],
        "hierarchical_num_iters": EXPERIMENT_CONFIG["hierarchical"]["num_iters"],
        "tent_hierarchical_num_iters": get_hierarchical_config(EXPERIMENT_CONFIG, for_tent=True)["num_iters"],
        "hierarchical_lr": EXPERIMENT_CONFIG["hierarchical"]["lr"], "flat_tau": EXPERIMENT_CONFIG["flat"]["tau"],
        "tent_lr": EXPERIMENT_CONFIG["tent"]["lr"], "tent_num_steps": EXPERIMENT_CONFIG["tent"]["num_steps"],
    }


def save_batch_outputs(batch_output_dir, base_row, method_results, include_oracle_gap):
    os.makedirs(batch_output_dir, exist_ok=True)
    mapping = {v: k for k, v in METHOD_LABELS.items()}
    for result in method_results:
        save_method_details(batch_output_dir, mapping[result["method"]], result)
    comparison = pd.DataFrame(method_results)
    preferred = ["method", "loss", "perplexity"] + (["oracle_gap_closed"] if include_oracle_gap else []) + ["total_time_sec", "num_tokens"]
    columns = [c for c in preferred if c in comparison.columns] + [c for c in comparison.columns if c not in preferred]
    comparison = comparison[columns]
    comparison.to_csv(os.path.join(batch_output_dir, "comparison_table.csv"), index=False)
    print("\nBatch comparison:")
    print(comparison[[c for c in preferred if c in comparison.columns]].to_string(index=False), flush=True)

    long_rows = []
    wide = copy.deepcopy(base_row)
    for result in method_results:
        row = copy.deepcopy(base_row)
        row.update(result)
        long_rows.append(row)
        prefix = mapping[result["method"]]
        wide.update({f"{prefix}_{k}": v for k, v in result.items() if k != "method"})
    pd.DataFrame([wide]).to_csv(os.path.join(batch_output_dir, "general_results.csv"), index=False)
    return wide, long_rows


def summarize_long_results(long_results, group_cols=("dataset", "method"), include_oracle_gap=True):
    df = pd.DataFrame(long_results)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in df.groupby(list(group_cols), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["num_batches"] = len(group)
        for col, out in [("loss", "loss"), ("perplexity", "perplexity"), ("total_time_sec", "total_time")]:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{out}_mean"] = values.mean()
            row[f"{out}_std"] = values.std()
        if include_oracle_gap and "oracle_gap_closed" in group.columns:
            values = pd.to_numeric(group["oracle_gap_closed"], errors="coerce")
            row["oracle_gap_closed_mean"] = values.mean()
            row["oracle_gap_closed_std"] = values.std()
        if "relative_loss_improvement_pct" in group.columns:
            values = pd.to_numeric(group["relative_loss_improvement_pct"], errors="coerce")
            row["relative_loss_improvement_pct_mean"] = values.mean()
            row["relative_loss_improvement_pct_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows)


def format_mean_std(mean, std, digits=4):
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def save_primary_paper_tables(summary_df, output_dir):
    specs = [
        ("loss", "loss_mean", "loss_std", 4),
        ("oracle_gap_closed", "oracle_gap_closed_mean", "oracle_gap_closed_std", 2),
        ("deployment_time", "total_time_mean", "total_time_std", 2),
    ]
    for name, mean_col, std_col, digits in specs:
        if mean_col not in summary_df.columns:
            continue
        table = summary_df.copy()
        table["value"] = [format_mean_std(m, s, digits) for m, s in zip(table[mean_col], table[std_col])]
        pivot = table.pivot(index="method", columns="dataset", values="value")
        order = [METHOD_LABELS[m] for m in PRIMARY_METHODS if METHOD_LABELS[m] in pivot.index]
        pivot = pivot.reindex(order)
        pivot.to_csv(os.path.join(output_dir, f"primary_episodic_{name}_table.csv"))
        with open(os.path.join(output_dir, f"primary_episodic_{name}_table.tex"), "w", encoding="utf-8") as f:
            f.write(pivot.to_latex(escape=False))


def save_nonstationary_table(summary_df, output_dir):
    if summary_df.empty:
        return
    rows = []
    for _, row in summary_df.iterrows():
        rows.append({
            "Method": row["method"],
            "Loss": format_mean_std(row["loss_mean"], row["loss_std"], 4),
            "Relative Loss Improvement vs Pretrained (%)": format_mean_std(
                row.get("relative_loss_improvement_pct_mean", float("nan")),
                row.get("relative_loss_improvement_pct_std", float("nan")), 2,
            ),
            "Deployment Time / Batch (s)": format_mean_std(row["total_time_mean"], row["total_time_std"], 2),
        })
    table = pd.DataFrame(rows)
    order = [METHOD_LABELS[m] for m in NONSTATIONARY_METHODS]
    table["_order"] = table["Method"].map({name: i for i, name in enumerate(order)})
    table = table.sort_values("_order").drop(columns="_order")
    table.to_csv(os.path.join(output_dir, "nonstationary_overall_comparison_table.csv"), index=False)
    with open(os.path.join(output_dir, "nonstationary_overall_comparison_table.tex"), "w", encoding="utf-8") as f:
        f.write(table.to_latex(index=False, escape=False))

def save_method_comparison_table(output_dir):
    rows = [
        {"Method": "Pretrained", "Deployment signal": "None", "State": "Fixed", "Frequency": "Never"},
        {"Method": "Best Single FT", "Deployment signal": "First batch", "State": "Fixed after B1", "Frequency": "Once"},
        {"Method": "Mixed-source FT", "Deployment signal": "None", "State": "Fixed", "Frequency": "Never"},
        {"Method": "Static TENT (Best Single-FT)", "Deployment signal": "First batch", "State": "TENT-adapted then frozen", "Frequency": "Once"},
        {"Method": "TENT (Best Single-FT)", "Deployment signal": "Current batch", "State": "Carried across batches", "Frequency": "Every batch after B1 selection"},
        {"Method": "Hard Routing", "Deployment signal": "Current batch", "State": "Recomputed", "Frequency": "Every batch"},
        {"Method": "Flat Routing", "Deployment signal": "Current batch", "State": "Recomputed", "Frequency": "Every batch"},
        {"Method": "Static Hierarchical Routing", "Deployment signal": "First batch", "State": "Frozen after B1", "Frequency": "Once"},
        {"Method": "Hierarchical Routing", "Deployment signal": "Current batch", "State": "Recomputed", "Frequency": "Every batch"},
        {"Method": "TENT (Hierarchical Routing)", "Deployment signal": "B1 routing + current batch", "State": "TENT state carried after B1", "Frequency": "Hierarchical once, TENT every batch"},
        {"Method": "Target-FT Oracle", "Deployment signal": "Target training data", "State": "Fixed", "Frequency": "Offline; Primary only"},
    ]
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "method_comparison.csv"), index=False)


# ============================================================
# PRIMARY EPISODIC PROTOCOL
# ============================================================
# Scientific question:
#   On a stationary deployment condition, how well does each deployment method
#   perform on the incoming batch itself?
#
# Protocol:
#   * Each deployment dataset D is evaluated independently.
#   * Batches start after deployment_offset_tokens (512k by default) because
#     the Target-FT Oracle may have seen earlier target tokens during training.
#   * The same incoming batch B is used for selection/adaptation and evaluation;
#     this is intentional transductive test-time adaptation.
#   * Best Single-FT is selected once on B1. TENT (Best Single-FT) starts from
#     that checkpoint and then adapts online; Static TENT performs only the B1
#     update and freezes the resulting model.
#   * Static Hierarchical Routing is solved on B1 and frozen. Per-batch
#     Hierarchical Routing is recomputed on every batch.
#   * TENT (Hierarchical Routing) performs its own shorter Hierarchical Routing
#     optimization on B1, then continues with online TENT only.
#   * Target-FT Oracle is evaluated here, so Oracle Gap Closed is computed per
#     batch and only then averaged.
#   * PCA/grid diagnostics are preserved on B1 of every deployment dataset and
#     are not counted in Hierarchical Routing runtime.


def run_primary_episodic(tokenizer, metadata_path, bank_df, device, EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY):
    cfg = EXPERIMENT_CONFIG["primary_episodic"]
    if not cfg.get("enabled", False):
        return None
    methods = get_protocol_methods(EXPERIMENT_CONFIG, "primary_episodic")
    out = protocol_dir(EXPERIMENT_CONFIG, "primary_episodic")
    os.makedirs(out, exist_ok=True)
    save_status(out, "running", protocol="primary_episodic")
    offset = int(cfg.get("deployment_offset_tokens", 512_000))
    num_batches = int(cfg["num_batches"])
    all_wide, all_long = [], []

    for dataset_name in EXPERIMENT_CONFIG["deployments"]:
        print(f"\n\n################ PRIMARY EPISODIC: {dataset_name} ################", flush=True)
        batches, stats = load_deployment_batches(dataset_name, tokenizer, num_batches, offset, EXPERIMENT_CONFIG, DATASET_REGISTRY)
        dataset_dir = os.path.join(out, safe_name(dataset_name))
        os.makedirs(dataset_dir, exist_ok=True)
        save_json(os.path.join(dataset_dir, "dataset_stats.json"), stats)

        best_single, best_single_time = None, 0.0
        best_needed = any(m in methods for m in ["best_single_ft", "static_tent_best_single", "tent_best_single"])
        if best_needed:
            best_single, best_single_time, scores = select_best_single_ft(bank_df, batches[0], device, EXPERIMENT_CONFIG)
            scores.to_csv(os.path.join(dataset_dir, "best_single_ft_selection.csv"), index=False)

        static_tent_state = tent_best_state = tent_hier_state = None
        static_tent_init_time = tent_best_init_time = tent_hier_init_time = 0.0
        if "static_tent_best_single" in methods:
            static_tent_state, static_tent_init_time = init_tent_from_best_single(best_single, best_single_time, device, EXPERIMENT_CONFIG)
        if "tent_best_single" in methods:
            tent_best_state, tent_best_init_time = init_tent_from_best_single(best_single, best_single_time, device, EXPERIMENT_CONFIG)

        static_hier_model = None
        dataset_wide, dataset_long = [], []
        try:
            for batch_id, batch in enumerate(batches):
                save_status(out, "running", protocol="primary_episodic", dataset=dataset_name, batch=batch_id + 1)
                batch_dir = os.path.join(dataset_dir, f"batch_{batch_id + 1:02d}")
                os.makedirs(batch_dir, exist_ok=True)
                results = []

                if "pretrained" in methods:
                    results.append(run_pretrained(batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY))
                if "best_single_ft" in methods:
                    results.append(run_best_single_ft(batch, best_single, best_single_time, batch_id, device))
                if "mixed_ft" in methods:
                    results.append(run_mixed_ft(batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY))

                if "static_tent_best_single" in methods:
                    if batch_id == 0:
                        result = run_tent_batch(static_tent_state, batch, batch_id, static_tent_init_time, device, "Static TENT (Best Single-FT)")
                        result["adaptation_recomputed"] = True
                        results.append(result)
                    else:
                        results.append(evaluate_static_tent(static_tent_state["model"], batch, device))
                if "tent_best_single" in methods:
                    results.append(run_tent_batch(tent_best_state, batch, batch_id, tent_best_init_time, device, "TENT (Best Single-FT)"))
                if "oracle" in methods:
                    results.append(run_oracle(dataset_name, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY))

                routing_needed = any(m in methods for m in ["hard", "flat", "hierarchical"]) or (
                    batch_id == 0 and any(m in methods for m in ["static_hierarchical", "tent_hierarchical"])
                )
                scored = pretrained_loss = scoring_time = None
                if routing_needed:
                    scored, pretrained_loss, scoring_time = compute_shared_routing_scores(bank_df, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY)
                    scored.to_csv(os.path.join(batch_dir, "bank_batch_losses.csv"), index=False)

                if "hard" in methods:
                    results.append(run_hard(metadata_path, bank_df, scored, pretrained_loss, scoring_time, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY))
                if "flat" in methods:
                    flat_result, flat_info = run_flat(metadata_path, bank_df, scored, pretrained_loss, scoring_time, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY)
                    results.append(flat_result)
                    save_flat_details(batch_dir, flat_info)

                # Normal HR remains the source for Static HR and B1 PCA/grid diagnostics.
                hier_info = None
                hier_b1_needed = batch_id == 0 and any(m in methods for m in ["hierarchical", "static_hierarchical", "tent_hierarchical"])
                if hier_b1_needed:
                    hier_model, hier_result, hier_info = build_hierarchical_model(
                        metadata_path, bank_df, scored, pretrained_loss, scoring_time, batch, device,
                        EXPERIMENT_CONFIG, MODEL_REGISTRY, hierarchical_config=get_hierarchical_config(EXPERIMENT_CONFIG),
                    )
                    if "hierarchical" in methods:
                        results.append(hier_result)
                    if "tent_hierarchical" in methods:
                        # TENT has its own B1 HR solve; only bank scores are shared.
                        tent_hier_model, tent_hier_result, tent_hier_info = build_hierarchical_model(
                            metadata_path, bank_df, scored, pretrained_loss, scoring_time, batch, device,
                            EXPERIMENT_CONFIG, MODEL_REGISTRY, hierarchical_config=get_hierarchical_config(EXPERIMENT_CONFIG, for_tent=True),
                        )
                        tent_hier_state, tent_hier_init_time = init_tent_from_model(
                            tent_hier_model, tent_hier_result["total_time_sec"], device, EXPERIMENT_CONFIG,
                        )
                        results.append(run_tent_batch(
                            tent_hier_state, batch, batch_id, tent_hier_init_time, device, "TENT (Hierarchical Routing)",
                        ))
                    if "static_hierarchical" in methods:
                        static_hier_model = hier_model
                        static_result = copy.deepcopy(hier_result)
                        static_result["method"] = "Static Hierarchical Routing"
                        static_result["routing_recomputed"] = True
                        results.append(static_result)
                    else:
                        clear_model(hier_model, device)
                    save_hierarchical_details(batch_dir, hier_info)
                    run_pca_diagnostics(hier_info, batch_dir, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY, f"{dataset_name} - primary batch 1")
                elif batch_id > 0:
                    if "static_hierarchical" in methods:
                        if static_hier_model is None:
                            raise RuntimeError("Static Hierarchical model was not initialized on batch 1.")
                        results.append(evaluate_static_hierarchical(static_hier_model, batch, device))
                    if "hierarchical" in methods:
                        hier_model, hier_result, hier_info = build_hierarchical_model(
                            metadata_path, bank_df, scored, pretrained_loss, scoring_time, batch, device,
                            EXPERIMENT_CONFIG, MODEL_REGISTRY, hierarchical_config=get_hierarchical_config(EXPERIMENT_CONFIG),
                        )
                        results.append(hier_result)
                        save_hierarchical_details(batch_dir, hier_info)
                        clear_model(hier_model, device)
                    if "tent_hierarchical" in methods:
                        results.append(run_tent_batch(tent_hier_state, batch, batch_id, tent_hier_init_time, device, "TENT (Hierarchical Routing)"))

                add_oracle_gap(results)
                base = build_base_row(EXPERIMENT_CONFIG, MODEL_REGISTRY, "primary_episodic", dataset_name, batch_id, offset)
                wide, long_rows = save_batch_outputs(batch_dir, base, results, include_oracle_gap=True)
                dataset_wide.append(wide)
                dataset_long.extend(long_rows)
                all_wide.append(wide)
                all_long.extend(long_rows)

                if EXPERIMENT_CONFIG.get("progressive_save", True):
                    pd.DataFrame(dataset_wide).to_csv(os.path.join(dataset_dir, "all_batches_results.csv"), index=False)
                    pd.DataFrame(dataset_long).to_csv(os.path.join(dataset_dir, "all_method_results.csv"), index=False)
                    pd.DataFrame(all_wide).to_csv(os.path.join(out, "all_batches_results.csv"), index=False)
                    pd.DataFrame(all_long).to_csv(os.path.join(out, "all_method_results.csv"), index=False)

            dataset_summary = summarize_long_results(dataset_long, include_oracle_gap=True)
            dataset_summary.to_csv(os.path.join(dataset_dir, "summary_results.csv"), index=False)
        finally:
            clear_model(static_hier_model, device)
            clear_online_tent(static_tent_state, device)
            clear_online_tent(tent_best_state, device)
            clear_online_tent(tent_hier_state, device)

    summary = summarize_long_results(all_long, include_oracle_gap=True)
    summary.to_csv(os.path.join(out, "summary_results.csv"), index=False)
    save_primary_paper_tables(summary, out)
    save_status(out, "completed", protocol="primary_episodic")
    return {"output_dir": out, "summary": os.path.join(out, "summary_results.csv")}


# ============================================================
# NON-STATIONARY STREAM PROTOCOL
# ============================================================
# Scientific question:
#   Can methods follow rapid changes in deployment distribution when consecutive
#   incoming batches come from different domains?
#
# Protocol:
#   * No Target-FT Oracle is used; offset therefore defaults to 0.
#   * Source and deployment datasets must be disjoint.
#   * A balanced random stream is generated with a fixed seed and saved before
#     evaluation. The same batch is used for adaptation and evaluation.
#   * Best Single-FT is selected once on stream B1. Static TENT freezes after
#     its B1 update; online TENT keeps adapting across all domain switches.
#   * Static Hierarchical Routing is solved only on B1 and frozen. Hard, Flat
#     and ordinary Hierarchical Routing reroute on every incoming batch.
#   * TENT (Hierarchical Routing) performs its own shorter Hierarchical Routing
#     optimization on B1, then keeps only its online TENT state through the stream.
#   * Relative Loss Improvement is computed per batch against Pretrained before
#     averaging: 100 * (L_pretrained - L_method) / L_pretrained.
#   * PCA/grid diagnostics are preserved on stream B1 only and are not counted
#     in Hierarchical Routing runtime.


def run_nonstationary_stream(tokenizer, metadata_path, bank_df, device, EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY):
    cfg = EXPERIMENT_CONFIG["nonstationary_stream"]
    if not cfg.get("enabled", False):
        return None
    methods = get_protocol_methods(EXPERIMENT_CONFIG, "nonstationary_stream")
    out = protocol_dir(EXPERIMENT_CONFIG, "nonstationary_stream")
    os.makedirs(out, exist_ok=True)
    save_status(out, "running", protocol="nonstationary_stream")

    num_online = int(cfg["num_online_batches"])
    seed = int(cfg.get("seed", EXPERIMENT_CONFIG.get("seed", 42)))
    offset = int(cfg.get("deployment_offset_tokens", 0))
    sampling = cfg.get("sampling", "balanced_random")
    if sampling != "balanced_random":
        raise ValueError(f"Unsupported non-stationary sampling '{sampling}'. Expected 'balanced_random'.")

    sequence = build_balanced_stream(EXPERIMENT_CONFIG["deployments"], num_online, seed)
    pd.DataFrame({"batch_id": range(1, num_online + 1), "dataset": sequence}).to_csv(os.path.join(out, "nonstationary_stream_sequence.csv"), index=False)
    counts = Counter(sequence)

    dataset_batches, dataset_positions = {}, defaultdict(int)
    for dataset_name, needed in counts.items():
        batches, stats = load_deployment_batches(dataset_name, tokenizer, needed, offset, EXPERIMENT_CONFIG, DATASET_REGISTRY)
        dataset_batches[dataset_name] = batches
        save_json(os.path.join(out, f"dataset_stats_{safe_name(dataset_name)}.json"), stats)

    first_dataset = sequence[0]
    first_batch = dataset_batches[first_dataset][0]
    best_single, best_single_time = None, 0.0
    best_needed = any(m in methods for m in ["best_single_ft", "static_tent_best_single", "tent_best_single"])
    if best_needed:
        best_single, best_single_time, scores = select_best_single_ft(bank_df, first_batch, device, EXPERIMENT_CONFIG)
        scores.to_csv(os.path.join(out, "best_single_ft_selection.csv"), index=False)

    static_tent_state = tent_best_state = tent_hier_state = None
    static_tent_init_time = tent_best_init_time = tent_hier_init_time = 0.0
    if "static_tent_best_single" in methods:
        static_tent_state, static_tent_init_time = init_tent_from_best_single(best_single, best_single_time, device, EXPERIMENT_CONFIG)
    if "tent_best_single" in methods:
        tent_best_state, tent_best_init_time = init_tent_from_best_single(best_single, best_single_time, device, EXPERIMENT_CONFIG)

    static_hier_model = None
    all_wide, all_long = [], []
    try:
        for stream_id, dataset_name in enumerate(sequence):
            idx = dataset_positions[dataset_name]
            batch = dataset_batches[dataset_name][idx]
            dataset_positions[dataset_name] += 1
            save_status(out, "running", protocol="nonstationary_stream", dataset=dataset_name, batch=stream_id + 1)
            batch_dir = os.path.join(out, f"batch_{stream_id + 1:03d}_{safe_name(dataset_name)}")
            os.makedirs(batch_dir, exist_ok=True)
            results = []

            if "pretrained" in methods:
                results.append(run_pretrained(batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY))
            if "best_single_ft" in methods:
                results.append(run_best_single_ft(batch, best_single, best_single_time, stream_id, device))
            if "mixed_ft" in methods:
                results.append(run_mixed_ft(batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY))
            if "static_tent_best_single" in methods:
                if stream_id == 0:
                    result = run_tent_batch(static_tent_state, batch, stream_id, static_tent_init_time, device, "Static TENT (Best Single-FT)")
                    result["adaptation_recomputed"] = True
                    results.append(result)
                else:
                    results.append(evaluate_static_tent(static_tent_state["model"], batch, device))
            if "tent_best_single" in methods:
                results.append(run_tent_batch(tent_best_state, batch, stream_id, tent_best_init_time, device, "TENT (Best Single-FT)"))

            routing_needed = any(m in methods for m in ["hard", "flat", "hierarchical"]) or (
                stream_id == 0 and any(m in methods for m in ["static_hierarchical", "tent_hierarchical"])
            )
            scored = pretrained_loss = scoring_time = None
            if routing_needed:
                scored, pretrained_loss, scoring_time = compute_shared_routing_scores(bank_df, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY)
                scored.to_csv(os.path.join(batch_dir, "bank_batch_losses.csv"), index=False)

            if "hard" in methods:
                results.append(run_hard(metadata_path, bank_df, scored, pretrained_loss, scoring_time, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY))
            if "flat" in methods:
                flat_result, flat_info = run_flat(metadata_path, bank_df, scored, pretrained_loss, scoring_time, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY)
                results.append(flat_result)
                save_flat_details(batch_dir, flat_info)

            hier_b1_needed = stream_id == 0 and any(m in methods for m in ["hierarchical", "static_hierarchical", "tent_hierarchical"])
            if hier_b1_needed:
                hier_model, hier_result, hier_info = build_hierarchical_model(
                    metadata_path, bank_df, scored, pretrained_loss, scoring_time, batch, device,
                    EXPERIMENT_CONFIG, MODEL_REGISTRY, hierarchical_config=get_hierarchical_config(EXPERIMENT_CONFIG),
                )
                if "hierarchical" in methods:
                    results.append(hier_result)
                if "tent_hierarchical" in methods:
                    # TENT has its own B1 HR solve; only bank scores are shared.
                    tent_hier_model, tent_hier_result, tent_hier_info = build_hierarchical_model(
                        metadata_path, bank_df, scored, pretrained_loss, scoring_time, batch, device,
                        EXPERIMENT_CONFIG, MODEL_REGISTRY, hierarchical_config=get_hierarchical_config(EXPERIMENT_CONFIG, for_tent=True),
                    )
                    tent_hier_state, tent_hier_init_time = init_tent_from_model(
                        tent_hier_model, tent_hier_result["total_time_sec"], device, EXPERIMENT_CONFIG,
                    )
                    results.append(run_tent_batch(
                        tent_hier_state, batch, stream_id, tent_hier_init_time, device, "TENT (Hierarchical Routing)",
                    ))
                if "static_hierarchical" in methods:
                    static_hier_model = hier_model
                    static_result = copy.deepcopy(hier_result)
                    static_result["method"] = "Static Hierarchical Routing"
                    static_result["routing_recomputed"] = True
                    results.append(static_result)
                else:
                    clear_model(hier_model, device)
                save_hierarchical_details(batch_dir, hier_info)
                run_pca_diagnostics(hier_info, batch_dir, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY, f"non-stationary stream batch 1 ({dataset_name})")
            elif stream_id > 0:
                if "static_hierarchical" in methods:
                    if static_hier_model is None:
                        raise RuntimeError("Static Hierarchical model was not initialized on stream batch 1.")
                    results.append(evaluate_static_hierarchical(static_hier_model, batch, device))
                if "hierarchical" in methods:
                    hier_model, hier_result, hier_info = build_hierarchical_model(
                        metadata_path, bank_df, scored, pretrained_loss, scoring_time, batch, device,
                        EXPERIMENT_CONFIG, MODEL_REGISTRY, hierarchical_config=get_hierarchical_config(EXPERIMENT_CONFIG),
                    )
                    results.append(hier_result)
                    save_hierarchical_details(batch_dir, hier_info)
                    clear_model(hier_model, device)
                if "tent_hierarchical" in methods:
                    results.append(run_tent_batch(tent_hier_state, batch, stream_id, tent_hier_init_time, device, "TENT (Hierarchical Routing)"))

            add_relative_loss_improvement(results)
            base = build_base_row(EXPERIMENT_CONFIG, MODEL_REGISTRY, "nonstationary_stream", dataset_name, stream_id, offset)
            base["stream_batch_id"] = stream_id + 1
            base["dataset_local_batch_id"] = idx + 1
            wide, long_rows = save_batch_outputs(batch_dir, base, results, include_oracle_gap=False)
            all_wide.append(wide)
            all_long.extend(long_rows)

            if EXPERIMENT_CONFIG.get("progressive_save", True):
                pd.DataFrame(all_wide).to_csv(os.path.join(out, "nonstationary_batch_results_wide.csv"), index=False)
                pd.DataFrame(all_long).to_csv(os.path.join(out, "nonstationary_batch_results.csv"), index=False)
    finally:
        clear_model(static_hier_model, device)
        clear_online_tent(static_tent_state, device)
        clear_online_tent(tent_best_state, device)
        clear_online_tent(tent_hier_state, device)

    summary = summarize_long_results(all_long, group_cols=("method",), include_oracle_gap=False)
    summary.to_csv(os.path.join(out, "summary_results.csv"), index=False)
    save_nonstationary_table(summary, out)
    save_status(out, "completed", protocol="nonstationary_stream")
    return {"output_dir": out, "summary": os.path.join(out, "summary_results.csv")}


# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================
# experimental_pipeline.py should only define EXPERIMENT_CONFIG dictionaries
# and call this function. This runner creates one top-level directory per
# experiment, then one subdirectory per protocol:
#
# outputs/experimental_pipeline/<experiment_name>/
#   config.json
#   model_bank_metadata.csv
#   primary_episodic/
#   nonstationary_stream/
#
# Therefore a single pipeline launch can run experiment_1, experiment_2, ...
# sequentially, while every protocol writes progressive results to its own
# directory as soon as batches finish.


def run_experiment(EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY):
    validate_config(EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY)
    set_seed(EXPERIMENT_CONFIG.get("seed", 42))
    output_dir = get_output_dir(EXPERIMENT_CONFIG)
    os.makedirs(output_dir, exist_ok=True)
    save_experiment_config(output_dir, EXPERIMENT_CONFIG, MODEL_REGISTRY)
    save_method_comparison_table(output_dir)
    save_status(output_dir, "running")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = get_model_name(EXPERIMENT_CONFIG, MODEL_REGISTRY)
    print("\n============================================================")
    print("ROUTING EXPERIMENT")
    print("============================================================")
    print(f"Experiment:  {EXPERIMENT_CONFIG['experiment_name']}")
    print(f"Device:      {device}")
    print(f"Model:       {model_name}")
    print(f"Sources:     {EXPERIMENT_CONFIG['sources']}")
    print(f"Deployments: {EXPERIMENT_CONFIG['deployments']}")
    print(f"Output:      {output_dir}")
    print("============================================================", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    metadata_path, bank_df = build_model_bank_metadata(output_dir, EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY)

    results = {"experiment_name": EXPERIMENT_CONFIG["experiment_name"], "model": EXPERIMENT_CONFIG["model"], "output_dir": output_dir}
    start = time.time()
    try:
        if EXPERIMENT_CONFIG.get("primary_episodic", {}).get("enabled", False):
            results["primary_episodic"] = run_primary_episodic(tokenizer, metadata_path, bank_df, device, EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY)
        if EXPERIMENT_CONFIG.get("nonstationary_stream", {}).get("enabled", False):
            results["nonstationary_stream"] = run_nonstationary_stream(tokenizer, metadata_path, bank_df, device, EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY)
        results["status"] = "completed"
        results["total_time_sec"] = time.time() - start
        save_status(output_dir, "completed")
        save_json(os.path.join(output_dir, "experiment_summary.json"), results)
    except Exception as exc:
        results["status"] = "failed"
        results["error"] = str(exc)
        results["total_time_sec"] = time.time() - start
        save_status(output_dir, "failed", error=exc)
        save_json(os.path.join(output_dir, "experiment_summary.json"), results)
        raise

    print("\n============================================================")
    print("EXPERIMENT COMPLETE")
    print("============================================================")
    print(f"Experiment: {EXPERIMENT_CONFIG['experiment_name']}")
    print(f"Output:     {output_dir}")
    print("============================================================", flush=True)
    return results