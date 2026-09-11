import os
import time
import traceback

import pandas as pd

import routing_test


# ============================================================
# SHARED REGISTRIES
# ============================================================
#
# MODEL_REGISTRY and DATASET_REGISTRY describe resources that are shared
# by every experiment. They are defined once here and passed unchanged to
# routing_test.run_experiment(...).
#
# Only EXPERIMENT_CONFIG changes from one experiment to another.

MODEL_REGISTRY = {
    "tiny": {
        "model_name": "sshleifer/tiny-gpt2",
        "bank_prefix": "tiny-gpt2",
        "pretrained_subfolder": "tiny-gpt2/pretrained/model",
        "lambdas": ["0.0", "1e-05", "0.0001", "0.001", "0.01", "0.1", "0.3", "0.5", "1.0", "2.0"],
    },
    "small": {
        "model_name": "gpt2",
        "bank_prefix": "gpt2-small",
        "pretrained_subfolder": "gpt2-small/pretrained/model",
        "lambdas": ["0.0", "1e-05", "0.0001", "0.001", "0.01", "0.1", "0.3", "0.5", "1.0", "2.0"],
    },
    "medium": {
        "model_name": "gpt2-medium",
        "bank_prefix": "gpt2Medium",
        "pretrained_subfolder": None,
        "lambdas": ["0.0", "0.02", "0.05", "0.10", "0.20", "0.50", "0.70", "1.00", "1.50", "2.00"],
    },
    "large": {
        "model_name": "gpt2-large",
        "bank_prefix": "gpt2-large",
        "pretrained_subfolder": None,
        "lambdas": ["0.0", "0.02", "0.05", "0.10", "0.20", "0.50", "0.70", "1.00", "1.50", "2.00"],
    },
    "xlarge": {
        "model_name": "gpt2-xl",
        "bank_prefix": "gpt2-xl",
        "pretrained_subfolder": None,
        "lambdas": ["0.0", "0.02", "0.05", "0.10", "0.20", "0.50", "0.70", "1.00", "1.50", "2.00"],
    },
}


DATASET_REGISTRY = {
    "ArXiv": {
        "bank_name": "ArXiv",
        "dataset_config": {
            "type": "hf_text",
            "name": "timaeus/pile-arxiv",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": "ArXiv",
    },
    "DM Mathematics": {
        "bank_name": "DM Mathematics",
        "dataset_config": {
            "type": "hf_text",
            "name": "timaeus/pile-dm_mathematics",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": "DM Mathematics",
    },
    "EuroParl": {
        "bank_name": "EuroParl",
        "dataset_config": {
            "type": "translation",
            "name": "Helsinki-NLP/europarl",
            "config": "en-fr",
            "split": "train",
            "language": "en",
            "streaming": True,
        },
        "oracle_name": "EuroParl",
    },
    "FreeLaw": {
        "bank_name": "FreeLaw",
        "dataset_config": {
            "type": "hf_text",
            "name": "timaeus/pile-freelaw",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": "FreeLaw",
    },
    "GitHub": {
        "bank_name": "Github",
        "dataset_config": {
            "type": "hf_text",
            "name": "timaeus/pile-github",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": "Github",
    },
    "PG-19": {
        "bank_name": "Gutenberg (PG-19)",
        "dataset_config": {
            "type": "hf_text",
            "name": "emozilla/pg19",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": "Gutenberg (PG-19)",
    },
    "OWT2": {
        "bank_name": "OpenWebText2",
        "dataset_config": {
            "type": "hf_text",
            "name": "suolyer/pile_openwebtext2",
            "split": "validation",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": "OpenWebText2",
    },
    "PubMed Abstracts": {
        "bank_name": "PubMed Abstracts",
        "dataset_config": {
            "type": "hf_text",
            "name": "timaeus/pile-pubmed_abstracts",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": "PubMed Abstracts",
    },
    "PubMed Central": {
        "bank_name": "PubMed Central",
        "dataset_config": {
            "type": "hf_text",
            "name": "datajuicer/the-pile-pubmed-central-refined-by-data-juicer",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": "PubMed Central",
    },
    "StackExchange": {
        "bank_name": "StackExchange",
        "dataset_config": {
            "type": "hf_text",
            "name": "flax-sentence-embeddings/stackexchange_title_body_jsonl",
            "split": "train",
            "text_column": "texts",
            "streaming": True,
        },
        "oracle_name": "StackExchange",
    },
    "Wikipedia": {
        "bank_name": "Wikipedia (en)",
        "dataset_config": {
            "type": "hf_text",
            "name": "wikimedia/wikipedia",
            "config": "20231101.en",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": "Wikipedia (en)",
    },
    "BookCorpus": {
        "bank_name": "BookCorpus",
        "dataset_config": {
            "type": "hf_text",
            "name": "Yuti/bookcorpus",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": None,
    },
    "Pile-CC": {
        "bank_name": "Pile Common Crawl",
        "dataset_config": {
            "type": "hf_text",
            "name": "timaeus/pile-pile-cc",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": None,
    },
    "Ubuntu IRC": {
        "bank_name": "Ubuntu IRC",
        "dataset_config": {
            "type": "hf_text",
            "name": "common-pile/ubuntu_irc",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": None,
    },
    "Hacker News": {
        "bank_name": "Hacker News",
        "dataset_config": {
            "type": "hf_text",
            "name": "timaeus/pile-hackernews",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": None,
    },
    "Enron Emails": {
        "bank_name": "Enron Emails",
        "dataset_config": {
            "type": "hf_text",
            "name": "timaeus/pile-enron_emails",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": None,
    },
    "YouTube Subtitles": {
        "bank_name": "YouTube Subtitles",
        "dataset_config": {
            "type": "hf_text",
            "name": "suolyer/pile_youtubesubtitles",
            "split": "validation",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": None,
    },
    "USPTO": {
        "bank_name": "USPTO",
        "dataset_config": {
            "type": "hf_text",
            "name": "common-pile/uspto_filtered",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
        "oracle_name": None,
    },
}


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================
#
# Every experiment has its own EXPERIMENT_CONFIG. The model and dataset
# registries above remain unchanged.
#
# Each experiment can independently choose:
#   - the model size;
#   - the source domains used to build the routing bank;
#   - the deployment domains;
#   - routing/Tent hyperparameters;
#   - whether Primary Episodic and/or Non-stationary Stream are run;
#   - the methods enabled inside each protocol.
#
# Primary Episodic:
#   Each deployment dataset is evaluated independently. The default 512k-token
#   offset keeps evaluation data separate from the portion potentially used by
#   target-FT Oracle training. Online Tent carries state across batches of the
#   same dataset, then resets before the next deployment dataset.
#   A deployment with oracle_name=None is still valid: routing_test.py records
#   the Oracle and Oracle Gap Closed as NaN for that dataset, while all other
#   methods continue to run normally.
#
# Non-stationary Stream:
#   Batches from the deployment domains are interleaved in a balanced random
#   order. There is no Oracle and no Oracle Gap Closed. The stream starts at
#   token offset 0. Online Tent keeps its state across domain switches.
#   Sources and non-stationary deployment domains MUST be disjoint.

EXPERIMENT_1_CONFIG = {
    "experiment_name": "exp1_arxiv_freelaw_pubmedcentral",
    "model": "small",

    "sources": [
        "ArXiv",
        "FreeLaw",
        "PubMed Central",
    ],

    # Mixed FT must have been trained uniformly on exactly the source set above.
    # routing_test.py verifies the model prefix and source names from this path.
    "mixed_ft": {
        "subfolder": "gpt2-small/Mixed_FT_ArXiv_FreeLaw_PubMed_Central/model",
    },
    "deployments": [
        "PG-19",
        "PubMed Abstracts",
        "GitHub",
    ],

    "context_length": 512,
    "batch_size": 16,

    "hierarchical": {
        "H": 3,
        "num_iters": 5,
        "lr": 0.1,
        "num_random_starts": 5,
        "dirichlet_concentration": 1.0,
        "seed": 42,
    },
    "flat": {
        "tau": 1.0,
    },
    "tent": {
        "lr": 1e-3,
        "num_steps": 1,
    },

    # PCA/grid diagnostics are generated only for the first batch used by
    # normal Hierarchical Routing. Their runtime is not counted as routing time.
    "run_pca": True,
    "progressive_save": True,
    "seed": 42,

    "primary_episodic": {
        "enabled": True,
        "num_batches": 50,
        "deployment_offset_tokens": 512_000,
        "methods": [
            "pretrained",
            "best_single_ft",
            "mixed_ft",
            "episodic_tent",
            "online_tent",
            "hard",
            "flat",
            "static_hierarchical",
            "hierarchical",
            "oracle",
        ],
    },

    "nonstationary_stream": {
        "enabled": True,
        "num_online_batches": 21,
        "deployment_offset_tokens": 0,
        "sampling": "balanced_random",
        "seed": 42,
        "methods": [
            "pretrained",
            "best_single_ft",
            "mixed_ft",
            "episodic_tent",
            "online_tent",
            "hard",
            "flat",
            "static_hierarchical",
            "hierarchical",
        ],
    },
}



EXPERIMENT_2_CONFIG = {
    "experiment_name": "exp2_github_freelaw_stackexchange_dm",
    "model": "small",

    "sources": [
        "GitHub",
        "FreeLaw",
        "StackExchange",
        "DM Mathematics",
    ],

    # Expected Mixed-FT checkpoint for this exact source set.
    "mixed_ft": {
        "subfolder": "gpt2-small/Mixed_FT_GitHub_FreeLaw_StackExchange_DM_Mathematics/model",
    },
    "deployments": [
        "PG-19",
        "PubMed Abstracts",
        "Wikipedia",
    ],

    "context_length": 512,
    "batch_size": 16,

    "hierarchical": {
        "H": 3,
        "num_iters": 5,
        "lr": 0.1,
        "num_random_starts": 5,
        "dirichlet_concentration": 1.0,
        "seed": 42,
    },
    "flat": {
        "tau": 1.0,
    },
    "tent": {
        "lr": 1e-3,
        "num_steps": 1,
    },

    "run_pca": True,
    "progressive_save": True,
    "seed": 42,

    "primary_episodic": {
        "enabled": True,
        "num_batches": 50,
        "deployment_offset_tokens": 512_000,
        "methods": [
            "pretrained",
            "best_single_ft",
            "mixed_ft",
            "episodic_tent",
            "online_tent",
            "hard",
            "flat",
            "static_hierarchical",
            "hierarchical",
            "oracle",
        ],
    },

    "nonstationary_stream": {
        "enabled": True,
        "num_online_batches": 21,
        "deployment_offset_tokens": 0,
        "sampling": "balanced_random",
        "seed": 42,
        "methods": [
            "pretrained",
            "best_single_ft",
            "mixed_ft",
            "episodic_tent",
            "online_tent",
            "hard",
            "flat",
            "static_hierarchical",
            "hierarchical",
        ],
    },
}

EXPERIMENT_3_CONFIG = {
    "experiment_name": "exp3_full_bank",
    "model": "small",

    "sources": [
        "ArXiv",
        "DM Mathematics",
        "EuroParl",
        "FreeLaw",
        "GitHub",
        "PG-19",
        "OWT2",
        "PubMed Abstracts",
        "PubMed Central",
        "StackExchange",
        "Wikipedia",
    ],

    # Expected Mixed-FT checkpoint for this exact source set.
    "mixed_ft": {
        "subfolder": "gpt2-small/Mixed_FT_ArXiv_DM_Mathematics_EuroParl_FreeLaw_GitHub_PG-19_OWT2_PubMed_Abstracts_PubMed_Central_StackExchange_Wikipedia/model",
    },
    "deployments": [
        "Ubuntu IRC",
        "YouTube Subtitles",
        "Enron Emails",
    ],

    "context_length": 512,
    "batch_size": 16,

    "hierarchical": {
        "H": 2,
        "num_iters": 5,
        "lr": 0.1,
        "num_random_starts": 5,
        "dirichlet_concentration": 1.0,
        "seed": 42,
    },
    "flat": {
        "tau": 1.0,
    },
    "tent": {
        "lr": 1e-3,
        "num_steps": 1,
    },

    "run_pca": True,
    "progressive_save": True,
    "seed": 42,

    "primary_episodic": {
        "enabled": True,
        "num_batches": 50,
        "deployment_offset_tokens": 512_000,
        "methods": [
            "pretrained",
            "best_single_ft",
            "mixed_ft",
            "episodic_tent",
            "online_tent",
            "hard",
            "flat",
            "static_hierarchical",
            "hierarchical",
            "oracle",
        ],
    },

    "nonstationary_stream": {
        "enabled": True,
        "num_online_batches": 21,
        "deployment_offset_tokens": 0,
        "sampling": "balanced_random",
        "seed": 42,
        "methods": [
            "pretrained",
            "best_single_ft",
            "mixed_ft",
            "episodic_tent",
            "online_tent",
            "hard",
            "flat",
            "static_hierarchical",
            "hierarchical",
        ],
    },
}


# Only configurations listed here are executed, in this exact order.
EXPERIMENTS = [
    EXPERIMENT_1_CONFIG,
    EXPERIMENT_2_CONFIG,
    EXPERIMENT_3_CONFIG,
]


# ============================================================
# PIPELINE ORCHESTRATOR
# ============================================================
#
# This file deliberately contains no routing/adaptation implementation.
# All scientific logic lives in routing_test.py and its imported modules.
#
# The orchestrator runs experiments sequentially. routing_test.py saves
# protocol/batch results progressively, so results from experiment N remain
# inspectable while experiment N+1 is still running.
#
# A failure in one experiment is recorded in pipeline_summary.csv and does not
# prevent the following configured experiments from running.

OUTPUT_ROOT = routing_test.OUTPUT_ROOT


def save_pipeline_summary(rows):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_ROOT, "pipeline_summary.csv"), index=False)


def main():
    pipeline_rows = []
    pipeline_start = time.time()

    print("============================================================")
    print("EXPERIMENTAL PIPELINE")
    print("============================================================")
    print(f"Number of experiments: {len(EXPERIMENTS)}")
    print(f"Output root:           {OUTPUT_ROOT}")
    print("============================================================", flush=True)

    for experiment_id, experiment_config in enumerate(EXPERIMENTS, start=1):
        name = experiment_config["experiment_name"]
        start = time.time()

        print("\n\n############################################################")
        print(f"EXPERIMENT {experiment_id}/{len(EXPERIMENTS)}: {name}")
        print("############################################################", flush=True)

        row = {
            "experiment_id": experiment_id,
            "experiment_name": name,
            "model": experiment_config["model"],
            "sources": str(experiment_config["sources"]),
            "deployments": str(experiment_config["deployments"]),
            "primary_episodic_enabled": experiment_config.get("primary_episodic", {}).get("enabled", False),
            "nonstationary_stream_enabled": experiment_config.get("nonstationary_stream", {}).get("enabled", False),
            "status": "running",
            "total_time_sec": None,
            "output_dir": os.path.join(OUTPUT_ROOT, name),
            "error": None,
        }
        pipeline_rows.append(row)
        save_pipeline_summary(pipeline_rows)

        try:
            result = routing_test.run_experiment(
                EXPERIMENT_CONFIG=experiment_config,
                MODEL_REGISTRY=MODEL_REGISTRY,
                DATASET_REGISTRY=DATASET_REGISTRY,
            )
            row["status"] = result.get("status", "completed")
            row["output_dir"] = result.get("output_dir", row["output_dir"])

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"\nExperiment '{name}' FAILED.")
            traceback.print_exc()

        finally:
            row["total_time_sec"] = time.time() - start
            save_pipeline_summary(pipeline_rows)

    total_time = time.time() - pipeline_start
    completed = sum(row["status"] == "completed" for row in pipeline_rows)
    failed = sum(row["status"] == "failed" for row in pipeline_rows)

    print("\n============================================================")
    print("PIPELINE COMPLETE")
    print("============================================================")
    print(f"Completed: {completed}/{len(EXPERIMENTS)}")
    print(f"Failed:    {failed}/{len(EXPERIMENTS)}")
    print(f"Time:      {total_time:.1f}s")
    print(f"Summary:   {os.path.join(OUTPUT_ROOT, 'pipeline_summary.csv')}")
    print("============================================================", flush=True)


if __name__ == "__main__":
    main()
