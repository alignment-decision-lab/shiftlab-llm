import torch
import os
import pandas as pd
import utils
import algorithm_2 as algo2
import shift_measurement as sm
import interpolation_utils as iu
from shiftlab.data.load_datasets import load_dataset_from_subconfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader
import time
#---------------- Source datasets ----------------

source_configs = {
    "FreeLaw": {
        "dataset": {
            "type": "hf_text",
            "name": "timaeus/pile-freelaw",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "evaluation": {
            "dataset_size": 1000,
            "dataset_offset": 0,
            "batch_size": 16,
            "context_length": 512,
        },
    },

    "PubMed Central": {
        "dataset": {
            "type": "hf_text",
            "name": "datajuicer/the-pile-pubmed-central-refined-by-data-juicer",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "evaluation": {
            "dataset_size": 1000,
            "dataset_offset": 0,
            "batch_size": 16,
            "context_length": 512,
        },
    },

    "ArXiv": {
        "dataset": {
            "type": "hf_text",
            "name": "timaeus/pile-arxiv",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "evaluation": {
            "dataset_size": 1000,
            "dataset_offset": 0,
            "batch_size": 16,
            "context_length": 512,
        },
    },
}


# ---------------- Deployment datasets ----------------

deployment_configs = {
    "PG-19": {
        "dataset": {
            "type": "hf_text",
            "name": "emozilla/pg19",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "evaluation": {
            "dataset_size": 1000,
            "dataset_offset": 10000,
            "batch_size": 16,
            "context_length": 512,
        }
    },

     "PubMed Abstracts": {
    "dataset": {
        "type": "hf_text",
        "name": "timaeus/pile-pubmed_abstracts",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
    "evaluation": {
        "dataset_size": 1000,
        "dataset_offset": 50000,
        "batch_size": 16,
        "context_length": 512,
    },
},

"GitHub": {
    "dataset": {
        "type": "hf_text",
        "name": "timaeus/pile-github",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
    "evaluation": {
        "dataset_size": 1000,
        "dataset_offset": 10000,
        "batch_size": 16,
        "context_length": 512,
    },
},

"Ubuntu IRC": {
        "dataset": {
            "type": "hf_text",
            "name": "common-pile/ubuntu_irc",
            "split": "train",
            "text_column": "text",
            "streaming": True,

        },
        "evaluation": {
        "dataset_size": 1000,
        "dataset_offset": 10000,
        "batch_size": 16,
        "context_length": 512,
    },
    },

    # "YouTube Subtitles": {
    #     "dataset": {
    #         "type": "hf_text",
    #         "name": "suolyer/pile_youtubesubtitles",
    #         "split": "test",
    #         "text_column": "text",
    #         "streaming": True,

    #     },
    #     "evaluation": {
    #     "dataset_size": 200,
    #     "dataset_offset": 0,
    #     "batch_size": 16,
    #     "context_length": 512,
    # },
    # },
}

# ---------------- UTILITIES ----------------

def get_erm_model_path(erm_df, dataset_name):
    """Return the saved ERM model path associated with a dataset."""
    rows = erm_df[erm_df["dataset_name"] == dataset_name]

    if len(rows) == 0:
        raise ValueError(f"No ERM model found for dataset: {dataset_name}")

    if len(rows) > 1:
        raise ValueError(
            f"Several ERM models found for dataset: {dataset_name}. "
            "The metadata must contain exactly one final model "
            "per dataset."
        )

    model_path = rows.iloc[0]["model_dir"]

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"ERM model directory does not exist: {model_path}")

    return model_path

def load_deployment_data(deployment_cfg, tokenizer, data_collator):
    """
    Load a deployment dataset and split it into several selection
    batches followed by a fixed held-out evaluation subset.
    """
    evaluation_cfg = deployment_cfg["evaluation"]
    num_selection_batches = 10
    num_selection_examples = num_selection_batches * evaluation_cfg["batch_size"]
    dataset = load_dataset_from_subconfig(deployment_cfg["dataset"], evaluation_cfg)

    tokenization_config = {
        "training": {
            "context_length": evaluation_cfg["context_length"],
        }
    }

    tokenized_dataset = utils.tokenize_and_group_dataset(
        dataset,
        tokenizer,
        tokenization_config,
    )
    if len(tokenized_dataset) <= num_selection_examples:
        raise ValueError(
            f"Deployment dataset contains "
            f"{len(tokenized_dataset)} examples, but at least "
            f"{num_selection_examples + 1} are required for "
            f"{num_selection_batches} selection batches and a "
            "non-empty held-out set."
        )
            
    selection_dataset = tokenized_dataset.select(range(num_selection_examples))
    evaluation_dataset = tokenized_dataset.select(range(num_selection_examples, len(tokenized_dataset)))

    selection_dataloader = DataLoader(
        selection_dataset,
        batch_size=evaluation_cfg["batch_size"],
        shuffle=False,
        collate_fn=data_collator,
    )

    selection_batches = list(selection_dataloader)
    if len(selection_batches) != num_selection_batches:
        raise ValueError(
            f"Expected {num_selection_batches} complete selection "
            f"batches, but obtained {len(selection_batches)}."
        )

    evaluation_dataloader = DataLoader(
        evaluation_dataset,
        batch_size=evaluation_cfg["batch_size"],
        shuffle=False,
        collate_fn=data_collator,
    )

    return selection_batches, evaluation_dataloader

def to_float(value):
    """Convert a value to float, handling torch tensors."""
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    return float(value)

def load_source_token_distributions(bank_df):
    source_distributions = {}

    for source_name in sorted(
        bank_df["dataset_name"].unique()
    ):
        source_rows = bank_df[
            bank_df["dataset_name"] == source_name
        ]

        distribution_paths = (
            source_rows["source_distribution_path"]
            .dropna()
            .unique()
            .tolist()
        )

        if len(distribution_paths) != 1:
            raise ValueError(
                f"Expected exactly one source distribution "
                f"for {source_name}, found "
                f"{len(distribution_paths)}."
            )

        distribution_path = distribution_paths[0]

        if not os.path.isfile(distribution_path):
            raise FileNotFoundError(
                f"Source distribution not found: "
                f"{distribution_path}"
            )

        source_distributions[source_name] = torch.load(
            distribution_path,
            map_location="cpu",
        )

    return source_distributions

def create_distance_analysis_dataloader(
    dataset_cfg,
    tokenizer,
    data_collator,
):
    """
    Load and tokenize one complete dataset sample used to compute
    dataset-level shift statistics.
    """
    evaluation_cfg = dataset_cfg["evaluation"]

    dataset = load_dataset_from_subconfig(
        dataset_cfg["dataset"],
        evaluation_cfg,
    )

    tokenization_config = {
        "training": {
            "context_length": evaluation_cfg["context_length"],
        }
    }

    tokenized_dataset = utils.tokenize_and_group_dataset(
        dataset,
        tokenizer,
        tokenization_config,
    )

    if len(tokenized_dataset) == 0:
        raise ValueError(
            f"Empty dataset for configuration: "
            f"{dataset_cfg['dataset']['name']}"
        )

    return DataLoader(
        tokenized_dataset,
        batch_size=evaluation_cfg["batch_size"],
        shuffle=False,
        collate_fn=data_collator,
    )

# ------- METRICS -------

def compute_oracle_regret(oracle, algo2):
    """Compute the oracle regret."""
    return algo2 - oracle

def prepare_dataset_distance_analysis(
    dataset_dataloaders,
    deployment_names,
    source_names,
    source_token_distributions,
    reference_model,
    device,
    vocab_size,
    token_distribution_max_tokens=500_000,
    max_embedding_tokens=20000,
    max_pca_tokens_per_dataset=5000,
    pca_components_list=(5, 10, 20),
    token_epsilon=1e-8,
    gaussian_epsilon=1e-5,
):
    """
    Precompute all dataset statistics and dataset-to-source distances
    before running the batch-level experiment.
    """
    all_stats = sm.precompute_all_dataset_statistics(
        dataset_dataloaders=dataset_dataloaders,
        model=reference_model,
        device=device,
        vocab_size=vocab_size,
        source_token_distributions=(
            source_token_distributions
        ),
        token_distribution_max_tokens=(
            token_distribution_max_tokens
        ),
        max_embedding_tokens=max_embedding_tokens,
        token_epsilon=token_epsilon,
        gaussian_epsilon=gaussian_epsilon,
    )

    max_pca_components = max(pca_components_list)

    source_stats_for_pca = {
        source_name: all_stats[source_name]
        for source_name in source_names
    }

    pca_mean, pca_components = (
        sm.fit_common_pca_from_dataset_statistics(
            all_stats=source_stats_for_pca,
            n_components=max_pca_components,
            max_tokens_per_dataset=max_pca_tokens_per_dataset,
        )
    )

    all_stats = sm.add_pca_statistics(
        all_stats=all_stats,
        pca_mean=pca_mean,
        pca_components=pca_components,
        components_list=pca_components_list,
        epsilon=gaussian_epsilon,
    )

    dataset_distances = sm.compute_dataset_to_source_distances(
        all_stats=all_stats,
        deployment_names=deployment_names,
        source_names=source_names,
        pca_components_list=pca_components_list,
    )
    

    return {
        "dataset_statistics": all_stats,
        "dataset_distances": dataset_distances,
        "pca_mean": pca_mean,
        "pca_components": pca_components,
    }

def rank_interpolated_models(
    bank_df,
    deployment_name,
    selection_batch,
    evaluation_dataloader,
    source_distance_metrics,
    selected_source,
    selected_lambda_left,
    selected_lambda_right,
    algo2_selection_loss,
    algo2_selection_ppl,
    algo2_eval_loss,
    algo2_eval_ppl,
    algorithm_config,
    device,
    output_dir,
    tolerance=1e-12,
):
    """
    Evaluate and rank all midpoint interpolations from the model bank.

    With 3 sources and 9 bank models per source, including the
    ERM model represented by lambda=0:
        3 x 8 adjacent intervals = 24 interpolated models.

    The final ranking is based on selection loss.
    """
    ranking_rows = []

    source_names = sorted(
        bank_df["dataset_name"].unique()
    )

    for source_name in source_names:
        source_metrics = source_distance_metrics[source_name]
        source_rows = bank_df[
            bank_df["dataset_name"] == source_name
        ]

        trained_lambdas = sorted(
            source_rows["lambda"]
            .astype(float)
            .unique()
            .tolist()
        )

        if len(trained_lambdas) < 2:
            raise ValueError(
                f"Source {source_name} must contain at least "
                "two trained lambda values."
            )

        for lambda_left, lambda_right in zip(
            trained_lambdas[:-1],
            trained_lambdas[1:],
        ):
            is_algo2_choice = (
                source_name == selected_source
                and abs(
                    float(lambda_left)
                    - float(selected_lambda_left)
                ) <= tolerance
                and abs(
                    float(lambda_right)
                    - float(selected_lambda_right)
                ) <= tolerance
            )

            # Reuse the already evaluated Algorithm 2 model.
            if is_algo2_choice:
                selection_loss = algo2_selection_loss
                selection_ppl = algo2_selection_ppl
                eval_loss = algo2_eval_loss
                eval_ppl = algo2_eval_ppl

            else:
                left_row, right_row = (
                    algo2.select_model_interval_from_bank(
                        bank_df=bank_df,
                        selected_dataset=source_name,
                        lambda_left=lambda_left,
                        lambda_right=lambda_right,
                    )
                )

                model_left = AutoModelForCausalLM.from_pretrained(left_row["save_dir"]).to(device)
                model_right = AutoModelForCausalLM.from_pretrained(right_row["save_dir"]).to(device)

                interpolated_model = iu.interpolate_models(
                    model_left=model_left,
                    model_right=model_right,
                    beta=0.5,
                    config=algorithm_config,
                    device=device,
                )

                selection_loss, selection_ppl, _, _ = utils.evaluation(
                    interpolated_model,
                    [selection_batch],
                    device=device,
                )

                eval_loss, eval_ppl, _, _ = utils.evaluation(
                    interpolated_model,
                    evaluation_dataloader,
                    device=device,
                )

                del interpolated_model
                del model_left
                del model_right

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            ranking_rows.append(
                {
                    "source_dataset": source_name,

                    "dataset_to_source_token_kl": source_metrics[
                        "dataset_to_source_token_kl"
                    ],
                    "batch_to_source_token_kl": source_metrics[
                        "batch_to_source_token_kl"
                    ],

                    "dataset_to_source_embedding_mean_l2": source_metrics[
                        "dataset_to_source_embedding_mean_l2"
                    ],
                    "batch_to_source_embedding_mean_l2": source_metrics[
                        "batch_to_source_embedding_mean_l2"
                    ],

                    "dataset_to_source_diag_gaussian_kl": source_metrics[
                        "dataset_to_source_diag_gaussian_kl"
                    ],
                    "batch_to_source_diag_gaussian_kl": source_metrics[
                        "batch_to_source_diag_gaussian_kl"
                    ],

                    "dataset_to_source_pca_gaussian_kl_5": source_metrics[
                        "dataset_to_source_pca_gaussian_kl_5"
                    ],
                    "batch_to_source_pca_gaussian_kl_5": source_metrics[
                        "batch_to_source_pca_gaussian_kl_5"
                    ],

                    "dataset_to_source_pca_gaussian_kl_10": source_metrics[
                        "dataset_to_source_pca_gaussian_kl_10"
                    ],
                    "batch_to_source_pca_gaussian_kl_10": source_metrics[
                        "batch_to_source_pca_gaussian_kl_10"
                    ],

                    "dataset_to_source_pca_gaussian_kl_20": source_metrics[
                        "dataset_to_source_pca_gaussian_kl_20"
                    ],
                    "batch_to_source_pca_gaussian_kl_20": source_metrics[
                        "batch_to_source_pca_gaussian_kl_20"
                    ],

                    "lambda_left": float(lambda_left),
                    "lambda_right": float(lambda_right),
                    "lambda_midpoint": (
                        float(lambda_left) + float(lambda_right)
                    ) / 2.0,

                    "selection_loss": float(selection_loss),
                    "selection_ppl": float(selection_ppl),
                    "eval_loss": float(eval_loss),
                    "eval_ppl": float(eval_ppl),
                    "is_algo2_choice": bool(is_algo2_choice),
                }
            )
            

    ranking_df = pd.DataFrame(ranking_rows)

    if len(ranking_df) != 24:
        raise ValueError(
            "Expected 24 interpolated candidates, "
            f"but found {len(ranking_df)}."
        )

    num_algo2_matches = int(
        ranking_df["is_algo2_choice"].sum()
    )

    if num_algo2_matches != 1:
        raise ValueError(
            "Exactly one interpolation must match Algorithm 2. "
            f"Found {num_algo2_matches}."
        )

    ranking_df = ranking_df.sort_values(
        by="selection_loss",
        ascending=True,
    ).reset_index(drop=True)

    ranking_df["rank"] = range(1, len(ranking_df) + 1)

    ranking_df = ranking_df[
        [
            "rank",
            "source_dataset",

            "dataset_to_source_token_kl",
            "batch_to_source_token_kl",

            "dataset_to_source_embedding_mean_l2",
            "batch_to_source_embedding_mean_l2",

            "dataset_to_source_diag_gaussian_kl",
            "batch_to_source_diag_gaussian_kl",

            "dataset_to_source_pca_gaussian_kl_5",
            "batch_to_source_pca_gaussian_kl_5",

            "dataset_to_source_pca_gaussian_kl_10",
            "batch_to_source_pca_gaussian_kl_10",

            "dataset_to_source_pca_gaussian_kl_20",
            "batch_to_source_pca_gaussian_kl_20",

            "lambda_left",
            "lambda_right",
            "lambda_midpoint",

            "selection_loss",
            "selection_ppl",
            "eval_loss",
            "eval_ppl",

            "is_algo2_choice",
        ]
    ]

    os.makedirs(output_dir, exist_ok=True)

    ranking_csv_path = os.path.join(
        output_dir,
        "ranking.csv",
    )

    ranking_df.to_csv(
        ranking_csv_path,
        index=False,
    )

    best_row = ranking_df.iloc[0]

    algo2_row = ranking_df[
        ranking_df["is_algo2_choice"]
    ].iloc[0]

    return {
        "ranking_csv_path": ranking_csv_path,
        "num_candidates": int(len(ranking_df)),

        "best_source": best_row["source_dataset"],
        "best_lambda_left": float(best_row["lambda_left"]),
        "best_lambda_right": float(best_row["lambda_right"]),
        "best_lambda_midpoint": float(best_row["lambda_midpoint"]),
        "best_selection_loss": float(best_row["selection_loss"]),
        "best_selection_ppl": float(best_row["selection_ppl"]),
        "best_eval_loss": float(best_row["eval_loss"]),
        "best_eval_ppl": float(best_row["eval_ppl"]),

        "algo2_rank": int(algo2_row["rank"]),
        "algo2_is_best": bool(algo2_row["rank"] == 1),
        "algo2_eval_gap_to_selection_best_loss": float(
            algo2_row["eval_loss"] - best_row["eval_loss"]
        ),
        "algo2_eval_gap_to_selection_best_ppl": float(
            algo2_row["eval_ppl"] - best_row["eval_ppl"]
        ),
    }

# ---------------- MAIN EXPERIMENT FUNCTION ----------------

def run_real_experiment(
    model_bank_metadata_path,
    erm_models_metadata_path,
    deployment_configs,
    prepared_distance_analysis,
    reference_model,
    device,
):
    """
    Run Algorithm 2 on several deployment datasets and several
    independent deployment batches.

    For each batch, compare the interpolation selected by Algorithm 2
    with:
    1. the best midpoint interpolation according to selection-batch loss;
    2. the target-domain ERM oracle.

    A fixed held-out subset is shared across the selection batches of
    the same deployment dataset.
    """

    model_name = "gpt2-medium"
    root_output_dir = "outputs/real_experiment"
    os.makedirs(root_output_dir, exist_ok=True)

    algorithm_config = {
        "models": {
            "name": model_name,
        }
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    vocab_size = tokenizer.vocab_size

    erm_df = pd.read_csv(erm_models_metadata_path)

    required_columns = {
        "dataset_name",
        "model_dir",
    }

    missing_columns = required_columns - set(erm_df.columns)

    if missing_columns:
        raise ValueError(
            "ERM models metadata CSV is missing required columns: "
            f"{missing_columns}"
        )

    bank_df = pd.read_csv(model_bank_metadata_path)
    source_names = sorted(bank_df["dataset_name"].unique())

    source_statistics = {
        source_name: prepared_distance_analysis[
            "dataset_statistics"
        ][source_name]
        for source_name in source_names
    }

    dataset_distances = prepared_distance_analysis[
        "dataset_distances"
    ]

    pca_mean = prepared_distance_analysis["pca_mean"]
    pca_components = prepared_distance_analysis["pca_components"]

    all_results = []

    for deployment_name, deployment_cfg in deployment_configs.items():
        print(
            f"\n===== Deployment dataset: {deployment_name} =====",
            flush=True,
        )

        safe_deployment_name = (
            deployment_name
            .replace(" ", "_")
            .replace("/", "_")
        )

        deployment_output_dir = os.path.join(
            root_output_dir,
            safe_deployment_name,
        )
        os.makedirs(deployment_output_dir, exist_ok=True)

        selection_batches, evaluation_dataloader = load_deployment_data(
            deployment_cfg,
            tokenizer,
            data_collator,
        )

        print(
            f"Number of selection batches: {len(selection_batches)}",
            flush=True,
        )

        # The target oracle is the same for all batches of this dataset.
        oracle_erm_path = get_erm_model_path(
            erm_df,
            deployment_name,
        )

        oracle_model = AutoModelForCausalLM.from_pretrained(
            oracle_erm_path
        ).to(device)

        # The held-out subset is also fixed across all selection batches.
        oracle_eval_loss, oracle_eval_ppl, _, eval_num_tokens = (
            utils.evaluation(
                oracle_model,
                evaluation_dataloader,
                device=device,
            )
        )

        deployment_results = []

        for batch_id, selection_batch in enumerate(
            selection_batches,
            start=1,
        ):
            print(
                f"\n--- {deployment_name} | batch {batch_id:02d} ---",
                flush=True,
            )
            batch_start = time.time()

            batch_output_dir = os.path.join(
                deployment_output_dir,
                f"batch_{batch_id:02d}",
            )
            os.makedirs(batch_output_dir, exist_ok=True)

            selection_batch = utils.move_batch_to_device(
                selection_batch,
                device,
            )
            #---------------- Batch-to-source distances ----------------
            batch_distances = sm.compute_batch_to_source_distances(
                batch=selection_batch,
                source_statistics=source_statistics,
                reference_model=reference_model,
                device=device,
                pca_mean=pca_mean,
                pca_components=pca_components,
                vocab_size=vocab_size,
                pca_components_list=(5, 10, 20),
            )
            source_distance_metrics = {}

            for source_name in source_names:
                dataset_source_distances = (
                    dataset_distances[deployment_name][source_name]
                )

                batch_source_distances = batch_distances[source_name]

                source_distance_metrics[source_name] = {
                    "dataset_to_source_token_kl": float(
                        dataset_source_distances["token_kl"]
                    ),
                    "batch_to_source_token_kl": float(
                        batch_source_distances["token_kl"]
                    ),

                    "dataset_to_source_embedding_mean_l2": float(
                        dataset_source_distances["embedding_mean_l2"]
                    ),
                    "batch_to_source_embedding_mean_l2": float(
                        batch_source_distances["embedding_mean_l2"]
                    ),

                    "dataset_to_source_diag_gaussian_kl": float(
                        dataset_source_distances["diag_gaussian_kl"]
                    ),
                    "batch_to_source_diag_gaussian_kl": float(
                        batch_source_distances["diag_gaussian_kl"]
                    ),
                }

                for n_components in (5, 10, 20):
                    metric_name = (
                        f"pca_gaussian_kl_{n_components}"
                    )

                    source_distance_metrics[source_name][
                        f"dataset_to_source_{metric_name}"
                    ] = float(
                        dataset_source_distances[metric_name]
                    )

                    source_distance_metrics[source_name][
                        f"batch_to_source_{metric_name}"
                    ] = float(
                        batch_source_distances[metric_name]
                    )

            # ---------------- Algorithm 2 ----------------

            theta_B, info = algo2.run_algorithm_2(
                model_bank_metadata_path=model_bank_metadata_path,
                deployment_batch=selection_batch,
                vocab_size=vocab_size,
                config=algorithm_config,
                device=device,
            )
            for source_name in source_names:
                source_distance_metrics[source_name][
                    "batch_to_source_token_kl"
                ] = float(info["kl_values"][source_name])

            selected_source = info["selected_dataset"]

            print(
                f"Selected source: {selected_source}",
                flush=True,
            )

            # ---------------- Selection-batch metrics ----------------

            algo2_loss, algo2_ppl, _, selection_num_tokens = (
                utils.evaluation(
                    theta_B,
                    [selection_batch],
                    device=device,
                )
            )

            oracle_loss, oracle_ppl, _, _ = utils.evaluation(
                oracle_model,
                [selection_batch],
                device=device,
            )

            # ---------------- Fixed held-out metrics ----------------

            algo2_eval_loss, algo2_eval_ppl, _, _ = (
                utils.evaluation(
                    theta_B,
                    evaluation_dataloader,
                    device=device,
                )
            )

            # ---------------- Ranking ----------------

            ranking_info = rank_interpolated_models(
                bank_df=bank_df,
                deployment_name=deployment_name,
                selection_batch=selection_batch,
                evaluation_dataloader=evaluation_dataloader,
                source_distance_metrics=source_distance_metrics,
                selected_source=selected_source,
                selected_lambda_left=info["lambda_left"],
                selected_lambda_right=info["lambda_right"],
                algo2_selection_loss=algo2_loss,
                algo2_selection_ppl=algo2_ppl,
                algo2_eval_loss=algo2_eval_loss,
                algo2_eval_ppl=algo2_eval_ppl,
                algorithm_config=algorithm_config,
                device=device,
                output_dir=batch_output_dir,
            )

            # ---------------- Regrets ----------------

            best_selection_oracle_regret_loss = (
                compute_oracle_regret(
                    oracle_loss,
                    ranking_info["best_selection_loss"],
                )
            )

            best_selection_oracle_regret_ppl = (
                compute_oracle_regret(
                    oracle_ppl,
                    ranking_info["best_selection_ppl"],
                )
            )

            best_eval_oracle_regret_loss = (
                compute_oracle_regret(
                    oracle_eval_loss,
                    ranking_info["best_eval_loss"],
                )
            )

            best_eval_oracle_regret_ppl = (
                compute_oracle_regret(
                    oracle_eval_ppl,
                    ranking_info["best_eval_ppl"],
                )
            )

            algo2_selection_oracle_regret_loss = (
                compute_oracle_regret(
                    oracle_loss,
                    algo2_loss,
                )
            )

            algo2_selection_oracle_regret_ppl = (
                compute_oracle_regret(
                    oracle_ppl,
                    algo2_ppl,
                )
            )

            algo2_eval_oracle_regret_loss = (
                compute_oracle_regret(
                    oracle_eval_loss,
                    algo2_eval_loss,
                )
            )

            algo2_eval_oracle_regret_ppl = (
                compute_oracle_regret(
                    oracle_eval_ppl,
                    algo2_eval_ppl,
                )
            )

            # ---------------- Batch result ----------------

            result = {
                "deployment_dataset": deployment_name,
                "batch_id": batch_id,

                # Algorithm 2 decision
                "selected_source": selected_source,
                "rho_B": to_float(info["rho_B"]),
                "lambda_hat": float(info["lambda_hat"]),
                "lambda_left": float(info["lambda_left"]),
                "lambda_right": float(info["lambda_right"]),
                "lambda_midpoint": (
                    float(info["lambda_left"])
                    + float(info["lambda_right"])
                ) / 2.0,

                "selection_num_tokens": int(
                    selection_num_tokens
                ),
                "eval_num_tokens": int(eval_num_tokens),

                # Selection-batch metrics
                "algo2_selection_loss": float(algo2_loss),
                "algo2_selection_ppl": float(algo2_ppl),

                "best_interpolation_selection_loss": float(
                    ranking_info["best_selection_loss"]
                ),
                "best_interpolation_selection_ppl": float(
                    ranking_info["best_selection_ppl"]
                ),

                "oracle_selection_loss": float(oracle_loss),
                "oracle_selection_ppl": float(oracle_ppl),

                "algo2_selection_regret_to_best_loss": float(
                    algo2_loss
                    - ranking_info["best_selection_loss"]
                ),
                "algo2_selection_regret_to_best_ppl": float(
                    algo2_ppl
                    - ranking_info["best_selection_ppl"]
                ),

                "algo2_selection_oracle_regret_loss": float(
                    algo2_selection_oracle_regret_loss
                ),
                "algo2_selection_oracle_regret_ppl": float(
                    algo2_selection_oracle_regret_ppl
                ),

                "best_selection_oracle_regret_loss": float(
                    best_selection_oracle_regret_loss
                ),
                "best_selection_oracle_regret_ppl": float(
                    best_selection_oracle_regret_ppl
                ),

                # Held-out metrics
                "algo2_eval_loss": float(algo2_eval_loss),
                "algo2_eval_ppl": float(algo2_eval_ppl),

                "best_interpolation_eval_loss": float(
                    ranking_info["best_eval_loss"]
                ),
                "best_interpolation_eval_ppl": float(
                    ranking_info["best_eval_ppl"]
                ),

                "oracle_eval_loss": float(oracle_eval_loss),
                "oracle_eval_ppl": float(oracle_eval_ppl),

                "algo2_eval_gap_to_selection_best_loss": float(
                    ranking_info[
                        "algo2_eval_gap_to_selection_best_loss"
                    ]
                ),
                "algo2_eval_gap_to_selection_best_ppl": float(
                    ranking_info[
                        "algo2_eval_gap_to_selection_best_ppl"
                    ]
                ),

                "algo2_eval_oracle_regret_loss": float(
                    algo2_eval_oracle_regret_loss
                ),
                "algo2_eval_oracle_regret_ppl": float(
                    algo2_eval_oracle_regret_ppl
                ),

                "best_eval_oracle_regret_loss": float(
                    best_eval_oracle_regret_loss
                ),
                "best_eval_oracle_regret_ppl": float(
                    best_eval_oracle_regret_ppl
                ),

                # Ranking
                "algo2_rank": int(
                    ranking_info["algo2_rank"]
                ),
                "algo2_is_best": bool(
                    ranking_info["algo2_is_best"]
                ),
                "num_interpolations": int(
                    ranking_info["num_candidates"]
                ),

                "best_source": ranking_info["best_source"],
                "best_lambda_left": float(
                    ranking_info["best_lambda_left"]
                ),
                "best_lambda_right": float(
                    ranking_info["best_lambda_right"]
                ),
                "best_lambda_midpoint": float(
                    ranking_info["best_lambda_midpoint"]
                ),

                "interpolation_ranking_csv_path": (
                    ranking_info["ranking_csv_path"]
                ),
            }

            # Dataset-to-source and batch-to-source distance metrics
            for source_name, metrics in source_distance_metrics.items():
                safe_source_name = (
                    source_name
                    .replace(" ", "_")
                    .replace("/", "_")
                )

                for metric_name, metric_value in metrics.items():
                    result[
                        f"{metric_name}_{safe_source_name}"
                    ] = float(metric_value)
                

            # One-line CSV for this specific batch
            batch_general_csv_path = os.path.join(
                batch_output_dir,
                "general_results.csv",
            )

            pd.DataFrame([result]).to_csv(
                batch_general_csv_path,
                index=False,
            )

            deployment_results.append(result)
            all_results.append(result)

            # Incremental dataset-level CSV
            deployment_results_path = os.path.join(
                deployment_output_dir,
                "all_batches_results.csv",
            )

            pd.DataFrame(deployment_results).to_csv(
                deployment_results_path,
                index=False,
            )

            # Incremental global CSV
            global_results_path = os.path.join(
                root_output_dir,
                "all_batches_results.csv",
            )

            pd.DataFrame(all_results).to_csv(
                global_results_path,
                index=False,
            )

            print(
                f"Best interpolation: "
                f"source={ranking_info['best_source']}, "
                f"interval=["
                f"{ranking_info['best_lambda_left']}, "
                f"{ranking_info['best_lambda_right']}], "
                f"selection_loss="
                f"{ranking_info['best_selection_loss']:.6f}",
                flush=True,
            )

            print(
                f"Algorithm 2 rank: "
                f"{ranking_info['algo2_rank']}/"
                f"{ranking_info['num_candidates']}",
                flush=True,
            )

            print(
                f"Batch results saved at: "
                f"{batch_general_csv_path}",
                flush=True,
            )

            del theta_B
            del selection_batch

            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            elapsed = time.time() - batch_start
            print(
                f"Batch {batch_id:02d} completed in "
                f"{elapsed / 60:.2f} minutes.",
                flush=True,
            )

        # The oracle and held-out dataloader are shared by all batches.
        del oracle_model
        del evaluation_dataloader
        del selection_batches

        if device.type == "cuda":
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)

    final_csv_path = os.path.join(
        root_output_dir,
        "all_batches_results.csv",
    )

    results_df.to_csv(
        final_csv_path,
        index=False,
    )

    print(
        f"\nGlobal results saved at: {final_csv_path}",
        flush=True,
    )

    return results_df

if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    start_time = time.time()
    model_name = "gpt2-medium"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name
    ).to(device)

    reference_model.eval()

    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    
    model_bank_metadata_path = (
        "outputs/model_bank/model_bank_metadata.csv"
    )

    bank_df = pd.read_csv(model_bank_metadata_path)

    source_token_distributions = (
        load_source_token_distributions(bank_df)
    )

    precomputed_path = os.path.join(
        "outputs",
        "real_experiment",
        "precomputed_distance_analysis.pt",
    )

    os.makedirs(
        os.path.dirname(precomputed_path),
        exist_ok=True,
    )

    if os.path.exists(precomputed_path):
        print(
            f"Loading precomputed distance analysis from: "
            f"{precomputed_path}",
            flush=True,
        )

        prepared_distance_analysis = torch.load(
            precomputed_path,
            map_location="cpu",
            weights_only=False,
        )

    else:
        dataset_dataloaders = {}

        for dataset_name, dataset_cfg in source_configs.items():
            print(
                f"Preparing source dataset: {dataset_name}",
                flush=True,
            )

            dataset_dataloaders[dataset_name] = (
                create_distance_analysis_dataloader(
                    dataset_cfg=dataset_cfg,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                )
            )

        for dataset_name, dataset_cfg in deployment_configs.items():
            print(
                f"Preparing deployment dataset: {dataset_name}",
                flush=True,
            )

            dataset_dataloaders[dataset_name] = (
                create_distance_analysis_dataloader(
                    dataset_cfg=dataset_cfg,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                )
            )

        prepared_distance_analysis = (
            prepare_dataset_distance_analysis(
                dataset_dataloaders=dataset_dataloaders,
                deployment_names=list(
                    deployment_configs.keys()
                ),
                source_names=list(
                    source_configs.keys()
                ),
                source_token_distributions=(
                    source_token_distributions
                ),
                reference_model=reference_model,
                device=device,
                vocab_size=tokenizer.vocab_size,
                token_distribution_max_tokens=500_000,
                max_embedding_tokens=20000,
                max_pca_tokens_per_dataset=5000,
                pca_components_list=(5, 10, 20),
            )
        )

        torch.save(
            prepared_distance_analysis,
            precomputed_path,
        )
        del dataset_dataloaders

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"Precomputed distance analysis saved at: "
            f"{precomputed_path}",
            flush=True,
        )

    # -----------------------------------------
    # Batch-level real experiment
    # -----------------------------------------

    results_df = run_real_experiment(
        model_bank_metadata_path=model_bank_metadata_path,
        erm_models_metadata_path=(
            "outputs/erm_baselines/"
            "oracle_erm_models_metadata.csv"
        ),
        deployment_configs=deployment_configs,
        prepared_distance_analysis=(
            prepared_distance_analysis
        ),
        reference_model=reference_model,
        device=device,
    )

    end_time = time.time()

    print(
        f"Done. Total time: "
        f"{end_time - start_time:.2f} seconds --> "
        f"{(end_time - start_time) / 60:.2f} minutes",
        flush=True,
    )