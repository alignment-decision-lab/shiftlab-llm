import os
import time

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

import utils
import hierarchical_routing as hr
import routing_baselines as rb
import routing_PCA as rpca
from shiftlab.data.load_datasets import load_dataset_from_subconfig


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "gpt2"
CONTEXT_LENGTH = 512
BATCH_SIZE = 16
NUM_ROUTING_BATCHES = 5
DEPLOYMENT_OFFSET_TOKENS = 512_000

BANK_REPO_ID = "alignment-decision-lab/robustness-model-bank"
MIXED_FT_SUBFOLDER = "gpt2-small/Mixed_FT_ArXiv_FreeLaw_PubMed_Central/model"

HIERARCHICAL_CONFIG = {
    "H": 3,
    "num_iters": 100,
    "lr": 0.1,
    "num_random_starts": 5,
    "dirichlet_concentration": 1.0,
    "flat_tau": 1.0,
    "seed": 42,
}

FLAT_TAU = 1.0
RUN_PCA = True
OUTPUT_DIR = "outputs/routing_test"

GPT2_SMALL_SOURCES = ["ArXiv", "FreeLaw", "PubMed Central"]

GPT2_SMALL_LAMBDAS = [
    "0.0", "0.0001", "0.001", "0.01", "0.1",
    "0.3", "0.5", "1.0", "1e-05", "2.0",
]

DEPLOYMENT_CONFIGS = {
    "PG-19": {
        "dataset_config": {
            "type": "hf_text",
            "name": "emozilla/pg19",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_subfolder": "gpt2-small/Gutenberg (PG-19)/lambda_0.0/model",
    },
    "PubMed Abstracts": {
        "dataset_config": {
            "type": "hf_text",
            "name": "timaeus/pile-pubmed_abstracts",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_subfolder": "gpt2-small/PubMed Abstracts/lambda_0.0/model",
    },
    "GitHub": {
        "dataset_config": {
            "type": "hf_text",
            "name": "timaeus/pile-github",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_subfolder": "gpt2-small/Github/lambda_0.0/model",
    },
    "Ubuntu IRC": {
        "dataset_config": {
            "type": "hf_text",
            "name": "common-pile/ubuntu_irc",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_subfolder": None,
    },
}


# ============================================================
# MODEL BANK
# ============================================================

def build_gpt2_small_bank_metadata():
    rows = []
    for dataset_name in GPT2_SMALL_SOURCES:
        for lambda_str in GPT2_SMALL_LAMBDAS:
            rows.append({
                "dataset_name": dataset_name,
                "lambda": float(lambda_str),
                "subfolder": f"gpt2-small/{dataset_name}/lambda_{lambda_str}/model",
                "status": "trained",
            })

    bank_df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata_path = os.path.join(OUTPUT_DIR, "gpt2_small_model_bank_metadata.csv")
    bank_df.to_csv(metadata_path, index=False)

    print(f"Built metadata for {len(bank_df)} GPT-2 Small checkpoints.", flush=True)
    return metadata_path, bank_df


# ============================================================
# DEPLOYMENT DATA
# ============================================================

def load_deployment_data(dataset_name, dataset_config, tokenizer):
    """Build consecutive incoming batches after the first 512,000 tokens."""
    print(f"\nLoading deployment dataset: {dataset_name}", flush=True)

    routing_max_tokens = NUM_ROUTING_BATCHES * BATCH_SIZE * CONTEXT_LENGTH
    total_max_tokens = DEPLOYMENT_OFFSET_TOKENS + routing_max_tokens

    # Load enough data to tokenize both the skipped prefix and routing batches.
    training_config = {
        "dataset_offset": 0,
        "context_length": CONTEXT_LENGTH,
        "max_tokens": total_max_tokens,
    }

    dataset = load_dataset_from_subconfig(
        dataset_config=dataset_config,
        training_config=training_config,
    )

    tokenization_config = {
        "dataset": {"text_column": "text"},
        "training": {
            "context_length": CONTEXT_LENGTH,
            "max_tokens": total_max_tokens,
        },
    }

    tokenized_dataset, dataset_stats = utils.tokenize_and_group_with_token_budget(
        dataset=dataset,
        tokenizer=tokenizer,
        config=tokenization_config,
    )

    # 512,000 is exactly 1,000 blocks of length 512.
    if DEPLOYMENT_OFFSET_TOKENS % CONTEXT_LENGTH != 0:
        raise ValueError(
            "DEPLOYMENT_OFFSET_TOKENS must be divisible by CONTEXT_LENGTH."
        )

    offset_blocks = DEPLOYMENT_OFFSET_TOKENS // CONTEXT_LENGTH
    routing_size = NUM_ROUTING_BATCHES * BATCH_SIZE
    required_blocks = offset_blocks + routing_size

    if len(tokenized_dataset) < required_blocks:
        raise RuntimeError(
            f"{dataset_name}: only {len(tokenized_dataset)} token blocks obtained, "
            f"but {required_blocks} are required "
            f"({offset_blocks} skipped + {routing_size} routing)."
        )

    routing_dataset = tokenized_dataset.select(
        range(offset_blocks, offset_blocks + routing_size)
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    routing_loader = DataLoader(
        routing_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )
    routing_batches = list(routing_loader)

    if len(routing_batches) != NUM_ROUTING_BATCHES:
        raise RuntimeError(
            f"{dataset_name}: expected {NUM_ROUTING_BATCHES} routing batches, "
            f"obtained {len(routing_batches)}."
        )

    print(
        f"{dataset_name}: skipped exactly {DEPLOYMENT_OFFSET_TOKENS:,} tokens "
        f"({offset_blocks} blocks), then built "
        f"{NUM_ROUTING_BATCHES} consecutive batches "
        f"({routing_max_tokens:,} tokens).",
        flush=True,
    )

    return routing_batches, dataset_stats


# ============================================================
# HELPERS
# ============================================================

def evaluate_batch(model, batch, device):
    return utils.evaluation(model, [batch], device)


def clear_model(model, device):
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_hub_model(subfolder, device):
    model = AutoModelForCausalLM.from_pretrained(
        BANK_REPO_ID,
        subfolder=subfolder,
    ).to(device)
    model.eval()
    return model


# ============================================================
# BEST SINGLE FT
# ============================================================

def select_best_single_ft(bank_df, routing_batch, device):
    """Select the best source ERM (lambda=0) on the first incoming batch.

    Only the three source ERM checkpoints are evaluated, so the measured
    selection time reflects the actual cost of the Best Single FT baseline.
    """
    erm_df = bank_df[
        bank_df["lambda"].astype(float) == 0.0
    ].copy()

    if len(erm_df) != len(GPT2_SMALL_SOURCES):
        raise RuntimeError(
            f"Expected {len(GPT2_SMALL_SOURCES)} source ERMs, "
            f"found {len(erm_df)}."
        )

    batch_device = utils.move_batch_to_device(routing_batch, device)
    rows = []

    selection_start = time.time()

    for _, row in erm_df.iterrows():
        model = hr.load_bank_checkpoint(row, BANK_REPO_ID).to(device)
        model.eval()

        with torch.no_grad():
            loss = model(**batch_device).loss.item()

        rows.append({
            "dataset_name": row["dataset_name"],
            "subfolder": row["subfolder"],
            "batch_loss": loss,
        })

        clear_model(model, device)

    selection_time = time.time() - selection_start

    scored_erm_df = pd.DataFrame(rows)
    best_row = scored_erm_df.loc[scored_erm_df["batch_loss"].idxmin()]

    best_single_ft = {
        "dataset_name": best_row["dataset_name"],
        "subfolder": best_row["subfolder"],
        "selection_loss": float(best_row["batch_loss"]),
    }

    return best_single_ft, selection_time, scored_erm_df


# ============================================================
# STATIC MODEL EVALUATION
# ============================================================

def evaluate_pretrained(routing_batch, device):
    start = time.time()

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    loss, _, _, num_tokens = evaluate_batch(model, routing_batch, device)

    total_time = time.time() - start
    clear_model(model, device)
    return loss, total_time, num_tokens


def evaluate_best_single_ft(routing_batch, best_single_ft, device):
    start = time.time()

    model = load_hub_model(best_single_ft["subfolder"], device)
    loss, _, _, num_tokens = evaluate_batch(model, routing_batch, device)

    total_time = time.time() - start
    clear_model(model, device)
    return loss, total_time, num_tokens


def evaluate_mixed_ft(routing_batch, device):
    start = time.time()

    model = load_hub_model(MIXED_FT_SUBFOLDER, device)
    loss, _, _, num_tokens = evaluate_batch(model, routing_batch, device)

    total_time = time.time() - start
    clear_model(model, device)
    return loss, total_time, num_tokens


def evaluate_oracle(routing_batch, oracle_subfolder, device):
    if oracle_subfolder is None:
        return float("nan"), float("nan"), None

    start = time.time()

    model = load_hub_model(oracle_subfolder, device)
    loss, _, _, num_tokens = evaluate_batch(model, routing_batch, device)

    total_time = time.time() - start
    clear_model(model, device)
    return loss, total_time, num_tokens


# ============================================================
# ONE ROUTING BATCH
# ============================================================

def run_one_batch(
    dataset_name,
    batch_id,
    routing_batch,
    metadata_path,
    bank_df,
    best_single_ft,
    best_single_selection_time,
    oracle_subfolder,
    device,
):
    print("\n============================================================")
    print(f"{dataset_name} - BATCH {batch_id + 1}/{NUM_ROUTING_BATCHES}")
    print("============================================================", flush=True)

    batch_output_dir = os.path.join(
        OUTPUT_DIR,
        dataset_name.replace(" ", "_"),
        f"batch_{batch_id + 1:02d}",
    )
    os.makedirs(batch_output_dir, exist_ok=True)

    result = {
        "dataset": dataset_name,
        "batch_id": batch_id + 1,
    }

    # --------------------------------------------------------
    # Pretrained
    # --------------------------------------------------------

    pretrained_batch_loss, pretrained_time, num_tokens = evaluate_pretrained(
        routing_batch, device
    )

    result.update({
        "pretrained_loss": pretrained_batch_loss,
        "pretrained_total_time_sec": pretrained_time,
        "routing_num_tokens": num_tokens,
    })

    # --------------------------------------------------------
    # Best Single FT
    # --------------------------------------------------------

    best_single_loss, best_single_eval_time, _ = evaluate_best_single_ft(
        routing_batch,
        best_single_ft,
        device,
    )

    # The model selection is paid once, on the first incoming batch.
    best_single_total_time = best_single_eval_time
    if batch_id == 0:
        best_single_total_time += best_single_selection_time

    result.update({
        "best_single_ft_loss": best_single_loss,
        "best_single_ft_total_time_sec": best_single_total_time,
        "best_single_ft_source": best_single_ft["dataset_name"],
    })

    # --------------------------------------------------------
    # Mixed FT
    # --------------------------------------------------------

    mixed_ft_loss, mixed_ft_time, _ = evaluate_mixed_ft(
        routing_batch, device
    )

    result.update({
        "mixed_ft_loss": mixed_ft_loss,
        "mixed_ft_total_time_sec": mixed_ft_time,
    })

    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------

    oracle_loss, oracle_time, _ = evaluate_oracle(
        routing_batch,
        oracle_subfolder,
        device,
    )

    result.update({
        "oracle_loss": oracle_loss,
        "oracle_total_time_sec": oracle_time,
        "oracle_subfolder": oracle_subfolder,
    })

    # --------------------------------------------------------
    # Shared scoring for routing methods
    # --------------------------------------------------------

    print("Scoring model bank once for all routing strategies...", flush=True)

    scoring_start = time.time()
    routing_batch_device = utils.move_batch_to_device(
        routing_batch, device
    )

    scored_bank_df = hr.compute_bank_batch_losses(
        bank_df=bank_df,
        batch=routing_batch_device,
        device=device,
        bank_repo_id=BANK_REPO_ID,
    )

    pretrained_loss = rb.compute_pretrained_loss(
        pretrained_model_name=MODEL_NAME,
        batch=routing_batch_device,
        device=device,
    )

    shared_scoring_time = time.time() - scoring_start
    result["shared_scoring_time_sec"] = shared_scoring_time

    print(
        f"Shared scoring complete: {len(scored_bank_df)} bank checkpoints "
        f"+ theta_0 in {shared_scoring_time:.1f}s.",
        flush=True,
    )

    # --------------------------------------------------------
    # Hierarchical Routing
    # --------------------------------------------------------

    method_start = time.time()

    hierarchical_model, hierarchical_info = hr.run_hierarchical_routing(
        model_bank_metadata_path=metadata_path,
        batch=routing_batch,
        pretrained_model_name=MODEL_NAME,
        device=device,
        config=HIERARCHICAL_CONFIG,
        bank_repo_id=BANK_REPO_ID,
        bank_df=bank_df,
        scored_bank_df=scored_bank_df,
        pretrained_loss=pretrained_loss,
    )

    # Stop here: PCA/grid are diagnostics, not routing runtime.
    hierarchical_method_time = time.time() - method_start
    hierarchical_total_time = (
        shared_scoring_time + hierarchical_method_time
    )

    result.update({
        "hierarchical_loss": float(hierarchical_info["batch_loss"]),
        "hierarchical_method_time_sec": hierarchical_method_time,
        "hierarchical_total_time_sec": hierarchical_total_time,
        "hierarchical_best_start": hierarchical_info["best_start_name"],
        "hierarchical_best_iteration": hierarchical_info["best_iteration"],
        "hierarchical_best_is_vertex": hierarchical_info["best_is_vertex"],
        "hierarchical_selected_sources": str(
            hierarchical_info["selected_sources"]
        ),
        "hierarchical_selected_lambdas": str(
            hierarchical_info["selected_lambdas"]
        ),
        "hierarchical_weights": str(
            hierarchical_info["weights"]
        ),
    })

    pd.DataFrame(
        hierarchical_info["source_relevance"]
    ).to_csv(
        os.path.join(
            batch_output_dir,
            "hierarchical_source_relevance.csv",
        ),
        index=False,
    )

    pd.DataFrame(
        hierarchical_info["optimization_trajectories"]
    ).to_csv(
        os.path.join(
            batch_output_dir,
            "hierarchical_optimization_trajectories.csv",
        ),
        index=False,
    )

    clear_model(hierarchical_model, device)

    # --------------------------------------------------------
    # PCA + Grid diagnostic
    # OUTSIDE hierarchical runtime
    # --------------------------------------------------------

    if RUN_PCA and batch_id == 0:
        rpca.run_routing_pca(
            info=hierarchical_info,
            output_dir=batch_output_dir,
            batch=routing_batch,
            device=device,
            pretrained_model_name=MODEL_NAME,
            bank_repo_id=BANK_REPO_ID,
            run_grid=True,
        )

    # --------------------------------------------------------
    # Hard Routing
    # --------------------------------------------------------

    method_start = time.time()

    hard_model, hard_info = rb.run_hard_routing(
        model_bank_metadata_path=metadata_path,
        batch=routing_batch,
        pretrained_model_name=MODEL_NAME,
        device=device,
        bank_repo_id=BANK_REPO_ID,
        bank_df=bank_df,
        scored_bank_df=scored_bank_df,
        pretrained_loss=pretrained_loss,
    )

    hard_method_time = time.time() - method_start
    hard_total_time = shared_scoring_time + hard_method_time

    result.update({
        "hard_loss": float(hard_info["batch_loss"]),
        "hard_method_time_sec": hard_method_time,
        "hard_total_time_sec": hard_total_time,
        "hard_selected": hard_info["selected"],
    })

    clear_model(hard_model, device)

    # --------------------------------------------------------
    # Flat Routing
    # --------------------------------------------------------

    method_start = time.time()

    flat_model, flat_info = rb.run_flat_routing(
        model_bank_metadata_path=metadata_path,
        batch=routing_batch,
        pretrained_model_name=MODEL_NAME,
        device=device,
        tau=FLAT_TAU,
        H=HIERARCHICAL_CONFIG["H"],
        bank_repo_id=BANK_REPO_ID,
        bank_df=bank_df,
        scored_bank_df=scored_bank_df,
        pretrained_loss=pretrained_loss,
    )

    # The Flat mixture loss must be measured on the actual composed model.
    flat_loss, _, _, _ = evaluate_batch(
        flat_model,
        routing_batch,
        device,
    )

    flat_method_time = time.time() - method_start
    flat_total_time = shared_scoring_time + flat_method_time

    result.update({
        "flat_loss": flat_loss,
        "flat_method_time_sec": flat_method_time,
        "flat_total_time_sec": flat_total_time,
        "flat_selected_candidates": str(
            flat_info["candidate_names"]
        ),
        "flat_weights": str(
            flat_info["weights"]
        ),
    })

    pd.DataFrame([
        {
            "candidate": name,
            "routing_loss": flat_info["candidate_losses"][name],
            "weight": flat_info["weights"][name],
        }
        for name in flat_info["candidate_names"]
    ]).to_csv(
        os.path.join(
            batch_output_dir,
            "flat_weights.csv",
        ),
        index=False,
    )

    clear_model(flat_model, device)

    # --------------------------------------------------------
    # Compact comparison
    # --------------------------------------------------------

    comparison_df = pd.DataFrame([
        {
            "method": "Pretrained",
            "batch_loss": result["pretrained_loss"],
            "total_time_sec": result["pretrained_total_time_sec"],
        },
        {
            "method": "Best Single FT",
            "batch_loss": result["best_single_ft_loss"],
            "total_time_sec": result["best_single_ft_total_time_sec"],
        },
        {
            "method": "Mixed FT",
            "batch_loss": result["mixed_ft_loss"],
            "total_time_sec": result["mixed_ft_total_time_sec"],
        },
        {
            "method": "Hard Routing",
            "batch_loss": result["hard_loss"],
            "total_time_sec": result["hard_total_time_sec"],
        },
        {
            "method": "Flat Routing",
            "batch_loss": result["flat_loss"],
            "total_time_sec": result["flat_total_time_sec"],
        },
        {
            "method": "Hierarchical Routing",
            "batch_loss": result["hierarchical_loss"],
            "total_time_sec": result["hierarchical_total_time_sec"],
        },
        {
            "method": "Oracle",
            "batch_loss": result["oracle_loss"],
            "total_time_sec": result["oracle_total_time_sec"],
        },
    ])

    comparison_df.to_csv(
        os.path.join(
            batch_output_dir,
            "comparison_table.csv",
        ),
        index=False,
    )

    pd.DataFrame([result]).to_csv(
        os.path.join(
            batch_output_dir,
            "general_results.csv",
        ),
        index=False,
    )

    print("\nBatch comparison:")
    print(comparison_df.to_string(index=False), flush=True)

    return result


# ============================================================
# SUMMARY
# ============================================================

def build_dataset_summary(dataset_name, dataset_results):
    df = pd.DataFrame(dataset_results)

    methods = {
        "Pretrained": (
            "pretrained_loss",
            "pretrained_total_time_sec",
        ),
        "Best Single FT": (
            "best_single_ft_loss",
            "best_single_ft_total_time_sec",
        ),
        "Mixed FT": (
            "mixed_ft_loss",
            "mixed_ft_total_time_sec",
        ),
        "Hard Routing": (
            "hard_loss",
            "hard_total_time_sec",
        ),
        "Flat Routing": (
            "flat_loss",
            "flat_total_time_sec",
        ),
        "Hierarchical Routing": (
            "hierarchical_loss",
            "hierarchical_total_time_sec",
        ),
        "Oracle": (
            "oracle_loss",
            "oracle_total_time_sec",
        ),
    }

    rows = []

    for method, (loss_col, time_col) in methods.items():
        rows.append({
            "dataset": dataset_name,
            "method": method,
            "num_batches": len(df),
            "loss_mean": df[loss_col].mean(),
            "loss_std": df[loss_col].std(),
            "total_time_mean_sec": df[time_col].mean(),
            "total_time_std_sec": df[time_col].std(),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    metadata_path, bank_df = build_gpt2_small_bank_metadata()

    print("\nRouting model bank:")
    print(
        bank_df[
            ["dataset_name", "lambda", "subfolder"]
        ].to_string(index=False),
        flush=True,
    )

    all_results = []
    all_summaries = []

    for dataset_name, config in DEPLOYMENT_CONFIGS.items():

        # ====================================================
        # DEPLOYMENT BATCHES
        # ====================================================

        routing_batches, dataset_stats = load_deployment_data(
            dataset_name=dataset_name,
            dataset_config=config["dataset_config"],
            tokenizer=tokenizer,
        )

        dataset_dir = os.path.join(
            OUTPUT_DIR,
            dataset_name.replace(" ", "_"),
        )
        os.makedirs(dataset_dir, exist_ok=True)

        # ====================================================
        # BEST SINGLE FT SELECTION
        #
        # Select once on the first incoming batch, using only
        # the three source ERM checkpoints (lambda=0).
        # The selected model is then fixed for all later batches.
        # ====================================================

        print(
            f"\nSelecting Best Single FT for {dataset_name} "
            f"from the three source ERM checkpoints...",
            flush=True,
        )

        best_single_ft, best_single_selection_time, best_single_scores_df = (
            select_best_single_ft(
                bank_df=bank_df,
                routing_batch=routing_batches[0],
                device=device,
            )
        )

        best_single_scores_df.to_csv(
            os.path.join(
                dataset_dir,
                "best_single_ft_selection.csv",
            ),
            index=False,
        )

        print(
            f"Best Single FT: {best_single_ft['dataset_name']} "
            f"(selection loss={best_single_ft['selection_loss']:.6f}, "
            f"selection time={best_single_selection_time:.1f}s)",
            flush=True,
        )

        # ====================================================
        # ROUTING BATCHES
        # ====================================================

        dataset_results = []

        for batch_id, routing_batch in enumerate(routing_batches):

            result = run_one_batch(
                dataset_name=dataset_name,
                batch_id=batch_id,
                routing_batch=routing_batch,
                metadata_path=metadata_path,
                bank_df=bank_df,
                best_single_ft=best_single_ft,
                best_single_selection_time=best_single_selection_time,
                oracle_subfolder=config["oracle_subfolder"],
                device=device,
            )

            dataset_results.append(result)
            all_results.append(result)

            # Save progressively in case the experiment is interrupted.
            pd.DataFrame(dataset_results).to_csv(
                os.path.join(
                    dataset_dir,
                    "all_batches_results.csv",
                ),
                index=False,
            )

            pd.DataFrame(all_results).to_csv(
                os.path.join(
                    OUTPUT_DIR,
                    "all_batches_results.csv",
                ),
                index=False,
            )

        # ====================================================
        # DATASET SUMMARY
        # ====================================================

        summary_df = build_dataset_summary(
            dataset_name,
            dataset_results,
        )

        all_summaries.append(summary_df)

        summary_df.to_csv(
            os.path.join(
                dataset_dir,
                "summary_results.csv",
            ),
            index=False,
        )

        pd.concat(
            all_summaries,
            ignore_index=True,
        ).to_csv(
            os.path.join(
                OUTPUT_DIR,
                "summary_results.csv",
            ),
            index=False,
        )

    print("\nRouting experiments complete.", flush=True)

if __name__ == "__main__":
    main()