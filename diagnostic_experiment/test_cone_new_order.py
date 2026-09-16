"""Primary-episodic comparison of routing order and Cone extrapolation.

GPT-2 Small, Experiment-2 source bank, PG-19 deployment.

Methods:
    Pretrained
    Hierarchical Routing
    Hierarchical Routing - New Order
    Cone Routing
    Cone Routing - New Order
    Target-FT Oracle

Protocol:
    - 50 consecutive PG-19 deployment batches after a 512k-token offset.
    - Routing is recomputed independently on every incoming batch.
    - Loss, deployment time and Oracle Gap Closed are summarized as mean ± std.
    - PCA/grid diagnostics are generated only for standard Hierarchical Routing
      and Hierarchical Routing - New Order on batch 1.
    - PCA is intentionally not generated for Cone methods yet.

Current-order bank losses are physically computed once per batch and shared by
Hierarchical and Cone, while their scoring cost is charged to both methods.
New-order methods perform their own selective scoring internally because the
set of evaluated lambda checkpoints depends on the ERM Top-H screening.
"""

import os
import time
import pandas as pd
import torch
from transformers import AutoTokenizer

import routing_test as rt
import hierarchical_routing as hr
import hierarchical_routing_new_order as hr_new
import eg_extrapolation as eg
import eg_extrapolation_new_order as eg_new
import routing_PCA as rpca
from experimental_pipeline import MODEL_REGISTRY, DATASET_REGISTRY


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENT_CONFIG = {
    "experiment_name": "test_cone_new_order_exp2_pg19",
    "model": "small",

    "sources": [
        "GitHub",
        "FreeLaw",
        "StackExchange",
        "DM Mathematics",
    ],

    "deployments": [
        "PG-19",
    ],

    "context_length": 512,
    "batch_size": 16,

    "hierarchical": {
        "H": 3,
        "num_iters": 20,
        "lr": 0.1,
        "num_random_starts": 5,
        "dirichlet_concentration": 1.0,
        "flat_tau": 1.0,
        "seed": 42,
    },

    "cone": {
        "H": 3,
        "num_iters": 20,
        "lr": 0.1,
        "cone_lr": None,
        "max_log_step": None,
        "num_random_starts": 5,
        "dirichlet_concentration": 1.0,
        "flat_tau": 1.0,
        "seed": 42,
        "anchor_mode": "worst",
    },

    "primary_episodic": {
        "num_batches": 1,# 50,
        "deployment_offset_tokens": 512_000,
    },

    "run_pca": True,
    "progressive_save": True,
    "seed": 42,
}

BANK_REPO_ID = rt.BANK_REPO_ID
OUTPUT_ROOT = os.path.join(rt.OUTPUT_ROOT, EXPERIMENT_CONFIG["experiment_name"])
METHOD_ORDER = [
    "Pretrained",
    "Hierarchical Routing",
    "Hierarchical Routing - New Order",
    "Cone Routing",
    "Cone Routing - New Order",
    "Target-FT Oracle",
]


# ============================================================
# HELPERS
# ============================================================

def clear_model(model, device):
    rt.clear_model(model, device)


def routing_result(method, info, total_time, extra=None):
    result = {
        "method": method, "loss": float(info["batch_loss"]),
        "perplexity": rt.loss_to_perplexity(info["batch_loss"]),
        "total_time_sec": float(total_time),
        "selected_sources": str(info.get("selected_sources")),
        "selected_lambdas": str(info.get("selected_lambdas")),
        "weights": str(info.get("weights")),
    }
    if extra:
        result.update(extra)
    return result


def save_info(batch_dir, prefix, info):
    rt.save_json(os.path.join(batch_dir, f"{prefix}_info.json"), info)
    if info.get("source_relevance") is not None:
        pd.DataFrame(info["source_relevance"]).to_csv(os.path.join(batch_dir, f"{prefix}_source_relevance.csv"), index=False)
    trajectories = info.get("optimization_trajectories")
    if trajectories is not None:
        pd.DataFrame(trajectories).to_csv(os.path.join(batch_dir, f"{prefix}_optimization_trajectories.csv"), index=False)


def save_cone_info(batch_dir, prefix, info):
    rt.save_json(os.path.join(batch_dir, f"{prefix}_info.json"), info)
    if info.get("source_relevance") is not None:
        pd.DataFrame(info["source_relevance"]).to_csv(os.path.join(batch_dir, f"{prefix}_source_relevance.csv"), index=False)
    if info.get("cone_trajectories") is not None:
        pd.DataFrame(info["cone_trajectories"]).to_csv(os.path.join(batch_dir, f"{prefix}_cone_trajectories.csv"), index=False)
    if info.get("baseline_trajectories") is not None:
        pd.DataFrame(info["baseline_trajectories"]).to_csv(os.path.join(batch_dir, f"{prefix}_baseline_trajectories.csv"), index=False)


def run_pca(info, batch_dir, batch, device, model_cfg, label):
    if not EXPERIMENT_CONFIG["run_pca"]:
        return
    print(f"Generating PCA/grid: {label}", flush=True)
    rpca.run_routing_pca(
        info=info, output_dir=os.path.join(batch_dir, label), batch=batch, device=device,
        pretrained_model_name=model_cfg["model_name"], bank_repo_id=BANK_REPO_ID,
        pretrained_subfolder=model_cfg.get("pretrained_subfolder"), run_grid=True,
    )


# ============================================================
# ROUTING METHODS
# ============================================================

def run_current_hierarchical(metadata_path, bank_df, scored_bank_df, pretrained_loss, scoring_time,
                             batch, device, model_cfg):
    start = time.time()
    model, info = hr.run_hierarchical_routing(
        model_bank_metadata_path=metadata_path, batch=batch, pretrained_model_name=model_cfg["model_name"],
        pretrained_subfolder=model_cfg.get("pretrained_subfolder"), device=device,
        config=EXPERIMENT_CONFIG["hierarchical"], bank_repo_id=BANK_REPO_ID,
        bank_df=bank_df, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss,
    )
    method_time = time.time() - start
    result = routing_result(
        "Hierarchical Routing", info, scoring_time + method_time,
        {"shared_scoring_time_sec": scoring_time, "method_time_sec": method_time,
         "best_start": info.get("best_start_name"), "best_iteration": info.get("best_iteration"),
         "best_is_vertex": info.get("best_is_vertex")},
    )
    return model, result, info


def run_new_order_hierarchical(metadata_path, bank_df, pretrained_loss, batch, device, model_cfg):
    start = time.time()
    model, info = hr_new.run_hierarchical_routing_new_order(
        model_bank_metadata_path=metadata_path, batch=batch, pretrained_model_name=model_cfg["model_name"],
        pretrained_subfolder=model_cfg.get("pretrained_subfolder"), device=device,
        config=EXPERIMENT_CONFIG["hierarchical"], bank_repo_id=BANK_REPO_ID,
        bank_df=bank_df, pretrained_loss=pretrained_loss,
    )
    total_time = time.time() - start
    result = routing_result(
        "Hierarchical Routing - New Order", info, total_time,
        {"method_time_sec": total_time, "best_start": info.get("best_start_name"),
         "best_iteration": info.get("best_iteration"), "best_is_vertex": info.get("best_is_vertex"),
         "screening_top_h": str(info.get("screening_top_h"))},
    )
    return model, result, info


def run_current_cone(metadata_path, bank_df, scored_bank_df, pretrained_loss, scoring_time,
                     batch, device, model_cfg):
    start = time.time()
    model, info = eg.run_cone_routing(
        model_bank_metadata_path=metadata_path, batch=batch, pretrained_model_name=model_cfg["model_name"],
        pretrained_subfolder=model_cfg.get("pretrained_subfolder"), device=device,
        config=EXPERIMENT_CONFIG["cone"], bank_repo_id=BANK_REPO_ID,
        bank_df=bank_df, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss,
    )
    method_time = time.time() - start
    result = routing_result(
        "Cone Routing", info, scoring_time + method_time,
        {"shared_scoring_time_sec": scoring_time, "method_time_sec": method_time,
         "used_cone": info.get("used_cone"), "anchor_name": info.get("anchor_name"),
         "cone_loss": info.get("cone_loss"), "baseline_loss": info.get("baseline_loss"),
         "cone_best_start": info.get("cone_best_start_name"), "cone_best_iteration": info.get("cone_best_iteration")},
    )
    return model, result, info


def run_new_order_cone(metadata_path, bank_df, pretrained_loss, batch, device, model_cfg):
    start = time.time()
    model, info = eg_new.run_cone_routing_new_order(
        model_bank_metadata_path=metadata_path, batch=batch, pretrained_model_name=model_cfg["model_name"],
        pretrained_subfolder=model_cfg.get("pretrained_subfolder"), device=device,
        config=EXPERIMENT_CONFIG["cone"], bank_repo_id=BANK_REPO_ID,
        bank_df=bank_df, pretrained_loss=pretrained_loss,
    )
    total_time = time.time() - start
    result = routing_result(
        "Cone Routing - New Order", info, total_time,
        {"method_time_sec": total_time, "used_cone": info.get("used_cone"),
         "anchor_name": info.get("anchor_name"), "cone_loss": info.get("cone_loss"),
         "baseline_loss": info.get("baseline_loss"), "cone_best_start": info.get("cone_best_start_name"),
         "cone_best_iteration": info.get("cone_best_iteration"),
         "screening_top_h": str(info.get("screening_top_h"))},
    )
    return model, result, info


# ============================================================
# OUTPUTS
# ============================================================

def add_oracle_gap(results):
    pre = next(r for r in results if r["method"] == "Pretrained")
    oracle = next(r for r in results if r["method"] == "Target-FT Oracle")
    for result in results:
        result["oracle_gap_closed"] = rt.oracle_gap_closed(result["loss"], pre["loss"], oracle["loss"])


def save_batch(batch_dir, batch_id, results):
    os.makedirs(batch_dir, exist_ok=True)
    comparison = pd.DataFrame(results)
    order = {name: i for i, name in enumerate(METHOD_ORDER)}
    comparison["_order"] = comparison["method"].map(order)
    comparison = comparison.sort_values("_order").drop(columns="_order")
    comparison.to_csv(os.path.join(batch_dir, "comparison_table.csv"), index=False)

    print(f"\nBatch {batch_id} comparison:")
    cols = ["method", "loss", "perplexity", "oracle_gap_closed", "total_time_sec"]
    print(comparison[[c for c in cols if c in comparison.columns]].to_string(index=False), flush=True)
    return comparison


def summarize(results_df):
    rows = []
    for method, group in results_df.groupby("method", sort=False):
        row = {"method": method, "num_batches": len(group)}
        for col, name in [("loss", "loss"), ("perplexity", "perplexity"),
                          ("total_time_sec", "deployment_time"), ("oracle_gap_closed", "oracle_gap_closed")]:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{name}_mean"], row[f"{name}_std"] = values.mean(), values.std()
        rows.append(row)

    summary = pd.DataFrame(rows)
    order = {name: i for i, name in enumerate(METHOD_ORDER)}
    summary["_order"] = summary["method"].map(order)
    return summary.sort_values("_order").drop(columns="_order")


def format_mean_std(mean, std, digits):
    return rt.format_mean_std(mean, std, digits)


def save_paper_tables(summary_df):
    specs = [
        ("loss", "loss_mean", "loss_std", 4),
        ("deployment_time", "deployment_time_mean", "deployment_time_std", 2),
        ("oracle_gap_closed", "oracle_gap_closed_mean", "oracle_gap_closed_std", 2),
    ]
    for name, mean_col, std_col, digits in specs:
        table = summary_df[["method", mean_col, std_col]].copy()
        table["value"] = [format_mean_std(m, s, digits) for m, s in zip(table[mean_col], table[std_col])]
        table[["method", "value"]].to_csv(os.path.join(OUTPUT_ROOT, f"{name}_table.csv"), index=False)
        with open(os.path.join(OUTPUT_ROOT, f"{name}_table.tex"), "w", encoding="utf-8") as f:
            f.write(table[["method", "value"]].to_latex(index=False, escape=False))


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    rt.set_seed(EXPERIMENT_CONFIG["seed"])
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = MODEL_REGISTRY[EXPERIMENT_CONFIG["model"]]
    dataset_name = EXPERIMENT_CONFIG["deployments"][0]
    primary_cfg = EXPERIMENT_CONFIG["primary_episodic"]

    print("============================================================")
    print("CONE + NEW ORDER TEST")
    print("============================================================")
    print(f"Model:       {model_cfg['model_name']}")
    print(f"Sources:     {EXPERIMENT_CONFIG['sources']}")
    print(f"Deployment:  {dataset_name}")
    print(f"Batches:     {primary_cfg['num_batches']}")
    print(f"H:           {EXPERIMENT_CONFIG['hierarchical']['H']}")
    print(f"Iterations:  {EXPERIMENT_CONFIG['hierarchical']['num_iters']}")
    print(f"Device:      {device}")
    print(f"Output:      {OUTPUT_ROOT}")
    print("============================================================", flush=True)

    rt.save_json(os.path.join(OUTPUT_ROOT, "config.json"), {
        "experiment": EXPERIMENT_CONFIG, "model_config": model_cfg, "bank_repo_id": BANK_REPO_ID,
    })

    metadata_path, bank_df = rt.build_model_bank_metadata(
        OUTPUT_ROOT, EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batches, stats = rt.load_deployment_batches(
        dataset_name=dataset_name, tokenizer=tokenizer, num_batches=primary_cfg["num_batches"],
        offset_tokens=primary_cfg["deployment_offset_tokens"], EXPERIMENT_CONFIG=EXPERIMENT_CONFIG,
        DATASET_REGISTRY=DATASET_REGISTRY,
    )
    rt.save_json(os.path.join(OUTPUT_ROOT, "dataset_stats.json"), stats)

    all_results = []
    for batch_id, batch in enumerate(batches, start=1):
        print(f"\n\n################ BATCH {batch_id:02d}/{len(batches)} ################", flush=True)
        batch_dir = os.path.join(OUTPUT_ROOT, f"batch_{batch_id:02d}")
        os.makedirs(batch_dir, exist_ok=True)

        # Current Order: full-bank scores are shared physically by Hierarchical and Cone.
        scored_bank_df, pretrained_loss, shared_scoring_time = rt.compute_shared_routing_scores(
            bank_df, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY
        )
        scored_bank_df.to_csv(os.path.join(batch_dir, "bank_batch_losses.csv"), index=False)

        results = [rt.run_pretrained(batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY)]

        hier_model, hier_result, hier_info = run_current_hierarchical(
            metadata_path, bank_df, scored_bank_df, pretrained_loss, shared_scoring_time,
            batch, device, model_cfg,
        )
        results.append(hier_result)

        hr_new_model, hr_new_result, hr_new_info = run_new_order_hierarchical(
            metadata_path, bank_df, pretrained_loss, batch, device, model_cfg
        )
        results.append(hr_new_result)

        cone_model, cone_result, cone_info = run_current_cone(
            metadata_path, bank_df, scored_bank_df, pretrained_loss, shared_scoring_time,
            batch, device, model_cfg,
        )
        results.append(cone_result)

        cone_new_model, cone_new_result, cone_new_info = run_new_order_cone(
            metadata_path, bank_df, pretrained_loss, batch, device, model_cfg
        )
        results.append(cone_new_result)

        oracle_result = rt.run_oracle(
            dataset_name, batch, device, EXPERIMENT_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY
        )
        results.append(oracle_result)
        add_oracle_gap(results)

        save_info(batch_dir, "hierarchical", hier_info)
        save_info(batch_dir, "hierarchical_new_order", hr_new_info)
        save_cone_info(batch_dir, "cone", cone_info)
        save_cone_info(batch_dir, "cone_new_order", cone_new_info)

        # PCA/grid only for the two simplex Hierarchical methods, and only on B1.
        if batch_id == 1:
            run_pca(hier_info, batch_dir, batch, device, model_cfg, "pca_hierarchical")
            run_pca(hr_new_info, batch_dir, batch, device, model_cfg, "pca_hierarchical_new_order")

        comparison = save_batch(batch_dir, batch_id, results)
        all_results.extend(comparison.to_dict("records"))
        pd.DataFrame(all_results).to_csv(os.path.join(OUTPUT_ROOT, "all_method_results.csv"), index=False)

        for model in [hier_model, hr_new_model, cone_model, cone_new_model]:
            clear_model(model, device)

    all_df = pd.DataFrame(all_results)
    summary = summarize(all_df)
    summary.to_csv(os.path.join(OUTPUT_ROOT, "summary_results.csv"), index=False)
    save_paper_tables(summary)

    print("\n============================================================")
    print("FINAL SUMMARY — 50 BATCHES")
    print("============================================================")
    display = summary.copy()
    display["Loss"] = [format_mean_std(m, s, 4) for m, s in zip(display["loss_mean"], display["loss_std"])]
    display["Deployment Time (s)"] = [format_mean_std(m, s, 2) for m, s in zip(display["deployment_time_mean"], display["deployment_time_std"])]
    display["Oracle Gap Closed (%)"] = [format_mean_std(m, s, 2) for m, s in zip(display["oracle_gap_closed_mean"], display["oracle_gap_closed_std"])]
    print(display[["method", "Loss", "Deployment Time (s)", "Oracle Gap Closed (%)"]].to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_ROOT}")
    print("============================================================", flush=True)


if __name__ == "__main__":
    main()