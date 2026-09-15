import os
import gc
import time
import math

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

from huggingface_hub import snapshot_download
import tempfile

import utils
import shift_measurement as sm
from shiftlab.data.load_datasets import load_dataset_from_subconfig

from lambda_effect_analysis import (
    SOURCE_NAME,
    SOURCE_CONFIG,
    TRAINING_CONFIG,
)


# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "gpt2"
MODEL_TAG = "gpt2_small" if MODEL_NAME == "gpt2" else MODEL_NAME.replace("-", "_")
MODEL_DISPLAY_NAME = "GPT-2 Small" if MODEL_NAME == "gpt2" else MODEL_NAME.replace("-", " ").upper()
CONFIG = {
    "models": {
        "name": MODEL_NAME,
    },
    "dataset": SOURCE_CONFIG,
    "training": TRAINING_CONFIG,
}

LAMBDAS = [
    0.0,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    0.1,
    0.3,
    0.5,
    1.0,
    2.0,
]

OUTPUT_DIR = f"outputs/robustness_efficiency_{MODEL_TAG}"

SEED = 42

CONTEXT_LENGTH = 512

NOISE_LEVELS = [
    0.05,
    0.1,
    0.3,
    0.5,
    0.8,
]

NOISY_DATASET_NAMES = {
    0.05: "ArXiv Noise 5%",
    0.1: "ArXiv Noise 10%",
    0.3: "ArXiv Noise 30%",
    0.5: "ArXiv Noise 50%",
    0.8: "ArXiv Noise 80%",
}

# ============================================================
# HUGGING FACE MODEL BANK
# ============================================================

HF_REPO_ID = "alignment-decision-lab/robustness-model-bank"
HF_MODEL_BANK_ROOT = "gpt2-small"
# ------------------------------------------------------------
# Evaluation budget
#
# ArXiv training used:
#
#     5,120,000 tokens
#     val_split_ratio = 0.1
#
# Therefore the held-out ArXiv validation split contains
# approximately:
#
#     512,000 tokens
#
# We evaluate every deployment dataset on the same budget:
# 500 sequences × 512 tokens = 256,000 input tokens.
# ------------------------------------------------------------

EVAL_MAX_TOKENS = 256_000
EVAL_NUM_SEQUENCES = EVAL_MAX_TOKENS // CONTEXT_LENGTH
if EVAL_MAX_TOKENS % CONTEXT_LENGTH != 0:
    raise ValueError(
        "EVAL_MAX_TOKENS must be divisible by CONTEXT_LENGTH."
    )

BATCH_SIZE = 16


# ============================================================
# DEPLOYMENT DATASETS
# ============================================================
#
# IMPORTANT:
#
# The KL values below correspond to:
#
#     KL(deployment || ArXiv)
#
# They are therefore read from the ArXiv COLUMN of:
#
# outputs/dataset_analysis/asymetric_token_kl_matrix.csv
#
# Selected shift levels:
#
# ArXiv              0.000
# DM Mathematics     3.036
# PubMed Abstracts   5.008
# BookCorpus         8.014
# YouTube Subtitles 10.970
#
# ============================================================

DEPLOYMENT_DATASETS = {
    "ArXiv": {
        "shift_level": "Source",
        "token_kl": 0.0,
        "config": SOURCE_CONFIG,
        "shift_type": "source",
        "noise_probability": None,
    },

    "DM Mathematics": {
        "shift_level": "Mild",
        "token_kl": 3.036307639789287,
        "shift_type": "natural_domain",
        "noise_probability": None,
        "config": {
            "type": "hf_text",
            "name": "timaeus/pile-dm_mathematics",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
    },

    "PubMed Abstracts": {
        "shift_level": "Moderate",
        "token_kl": 5.007714111933455,
        "shift_type": "natural_domain",
        "noise_probability": None,
        "config": {
            "type": "hf_text",
            "name": "timaeus/pile-pubmed_abstracts",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
    },

    "BookCorpus": {
        "shift_level": "Large",
        "token_kl": 8.013702781272432,
        "shift_type": "natural_domain",
        "noise_probability": None,
        "config": {
            "type": "hf_text",
            "name": "Yuti/bookcorpus",
            "split": "train",
            "text_column": "text",
            "streaming": True,
        },
    },

    "YouTube Subtitles": {
        "shift_level": "Strong",
        "token_kl": 10.970024091709192,
        "shift_type": "natural_domain",
        "noise_probability": None,
        "config": {
            "type": "hf_text",
            "name": "suolyer/pile_youtubesubtitles",
            "split": "validation",
            "text_column": "text",
            "streaming": True,
        },
    },
}
for noise_level, dataset_name in NOISY_DATASET_NAMES.items():
    DEPLOYMENT_DATASETS[dataset_name] = {
        "shift_level": f"Noise {int(noise_level * 100)}%",
        "token_kl": None,  # computed later
        "config": None,
        "noise_probability": noise_level,
        "shift_type": "synthetic_noise",
    }


# Fixed order used in tables and figures.
DATASET_ORDER = [
    "ArXiv",
    "DM Mathematics",
    "PubMed Abstracts",
    "BookCorpus",
    "YouTube Subtitles",
    "ArXiv Noise 5%",
    "ArXiv Noise 10%",
    "ArXiv Noise 30%",
    "ArXiv Noise 50%",
    "ArXiv Noise 80%",
]


# ============================================================
# TOKENIZER / COLLATOR
# ============================================================

def setup_tokenizer_and_collator():
    """
    Load only the tokenizer and data collator.

    We do not load a base language model here because all trained
    models are loaded later from the existing ArXiv model bank.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return tokenizer, data_collator


# ============================================================
# ARXIV HELD-OUT DATA
# ============================================================

def prepare_arxiv_heldout_dataloader(tokenizer, data_collator):
    """
    Reconstruct exactly the ArXiv train/validation split used when
    the lambda model bank was trained.

    This is important because evaluating on the beginning of ArXiv
    again would include examples used for fine-tuning.

    The original setup was:

        max_tokens = 5,120,000
        context_length = 512
        val_split_ratio = 0.1
        seed = 42

    We therefore rebuild the same tokenized dataset and the same
    deterministic split, then retain only the validation dataloader.
    """

    print(
        "\n"
        "============================================================\n"
        "Preparing ArXiv held-out evaluation set\n"
        "============================================================",
        flush=True,
    )

    raw_dataset = load_dataset_from_subconfig(SOURCE_CONFIG, TRAINING_CONFIG)

    tokenized_dataset, dataset_stats = utils.tokenize_and_group_with_token_budget(dataset=raw_dataset, tokenizer=tokenizer, config=CONFIG)
    (
        _,
        _,
        val_dataloader,
        _,
    ) = utils.create_training_dataloaders(
        tokenized_dataset=tokenized_dataset,
        data_collator=data_collator,
        training_config=TRAINING_CONFIG,
    )
    val_dataset = val_dataloader.dataset

    if len(val_dataset) < EVAL_NUM_SEQUENCES:
        raise ValueError(
            f"ArXiv validation set contains only "
            f"{len(val_dataset)} sequences, but "
            f"{EVAL_NUM_SEQUENCES} are required."
        )

    eval_dataset = val_dataset.select(range(EVAL_NUM_SEQUENCES))

    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)
    num_sequences = len(eval_dataset)
    effective_tokens = num_sequences * CONTEXT_LENGTH

    print(
        "\n===== ArXiv held-out evaluation =====\n"
        f"Sequences: {num_sequences:,}\n"
        f"Context length: {CONTEXT_LENGTH:,}\n"
        f"Effective input tokens: {effective_tokens:,}\n"
        "=====================================\n",
        flush=True,
    )

    stats = {
        "deployment_dataset": "ArXiv",
        "shift_level": "Source",
        "shift_type": "source",
        "noise_probability": None,
        "token_kl_deployment_to_arxiv": 0.0,
        "num_sequences": num_sequences,
        "context_length": CONTEXT_LENGTH,
        "effective_tokens": effective_tokens,
        "evaluation_source": "held_out_training_validation_split",
    }

    return eval_dataloader, eval_dataset, stats

def prepare_noisy_arxiv_dataloader(clean_eval_dataset, noise_level, tokenizer, data_collator):
    """
    Build a noisy version of exactly the same held-out ArXiv
    examples used for clean evaluation.

    Noise is applied after decoding the clean held-out sequences
    to text, then the corrupted texts are tokenized again.
    """

    dataset_name = NOISY_DATASET_NAMES[noise_level]

    print(
        "\n"
        "============================================================\n"
        f"Preparing {dataset_name}\n"
        f"Noise probability: {noise_level}\n"
        "============================================================",
        flush=True,
    )
    # --------------------------------------------------------
    # Decode the held-out ArXiv sequences
    # --------------------------------------------------------
    noisy_texts = []

    for idx in range(len(clean_eval_dataset)):
        example = clean_eval_dataset[idx]
        text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        noisy_example = utils.add_ocr_noise({"text": text}, p=noise_level, seed=SEED + idx)
        noisy_texts.append(noisy_example["text"])

    # --------------------------------------------------------
    # Rebuild a Hugging Face text dataset
    # --------------------------------------------------------

    noisy_raw_dataset = Dataset.from_dict({"text": noisy_texts})
    evaluation_training_config = {
        "seed": SEED,
        "context_length": CONTEXT_LENGTH,
        "max_tokens": EVAL_MAX_TOKENS,
        "batch_size": BATCH_SIZE,
    }

    evaluation_config = {
        "models": {"name": MODEL_NAME},
        "dataset": {"text_column": "text"},
        "training": evaluation_training_config,
    }

    # --------------------------------------------------------
    # Tokenize corrupted text with same evaluation budget
    # --------------------------------------------------------

    noisy_tokenized_dataset, tokenization_stats = (
        utils.tokenize_and_group_with_token_budget(
            dataset=noisy_raw_dataset,
            tokenizer=tokenizer,
            config=evaluation_config,
        )
    )

    dataloader = DataLoader(noisy_tokenized_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)

    stats = {
        "deployment_dataset": dataset_name,
        "shift_level": f"Noise {int(noise_level * 100)}%",
        "shift_type": "synthetic_noise",
        "noise_probability": noise_level,
        "token_kl_deployment_to_arxiv": None,
        "num_sequences": len(noisy_tokenized_dataset),
        "context_length": CONTEXT_LENGTH,
        "effective_tokens": tokenization_stats["effective_tokens"],
        "num_documents_used": tokenization_stats["num_documents_used"],
        "raw_tokens_seen": tokenization_stats["raw_tokens_seen"],
        "evaluation_source": "noisy_arxiv_heldout",
    }

    print(
        f"{dataset_name}: "
        f"{len(noisy_tokenized_dataset):,} sequences | "
        f"{tokenization_stats['effective_tokens']:,} tokens",
        flush=True,
    )
    return dataloader, stats


# ============================================================
# OTHER DEPLOYMENT DATASETS
# ============================================================

def prepare_shifted_dataloader(dataset_name, dataset_info, tokenizer, data_collator):
    """
    Build a fixed-token-budget evaluation set for one deployment
    dataset.

    All non-ArXiv deployment datasets use exactly EVAL_MAX_TOKENS
    whenever enough data are available.
    """

    print(
        "\n"
        "============================================================\n"
        f"Preparing deployment dataset: {dataset_name}\n"
        f"Shift level: {dataset_info['shift_level']}\n"
        f"Original selection KL(deployment || ArXiv): "
        f"{dataset_info['token_kl']:.6f}\n"
        "============================================================",
        flush=True,
    )

    dataset_config = dataset_info["config"]

    evaluation_training_config = {
        "seed": SEED,
        "context_length": CONTEXT_LENGTH,
        "max_tokens": EVAL_MAX_TOKENS,
        "batch_size": BATCH_SIZE,
    }

    evaluation_config = {
        "models": {
            "name": MODEL_NAME,
        },
        "dataset": dataset_config,
        "training": evaluation_training_config,
    }

    raw_dataset = load_dataset_from_subconfig(dataset_config, evaluation_training_config)
    tokenized_dataset, tokenization_stats = (
        utils.tokenize_and_group_with_token_budget(
            dataset=raw_dataset,
            tokenizer=tokenizer,
            config=evaluation_config,
        )
    )
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)

    stats = {
        "deployment_dataset": dataset_name,
        "shift_level": dataset_info["shift_level"],
        "shift_type": dataset_info["shift_type"],
        "noise_probability": None,
        "token_kl_deployment_to_arxiv": dataset_info["token_kl"],
        "num_sequences": len(tokenized_dataset),
        "context_length": CONTEXT_LENGTH,
        "effective_tokens": tokenization_stats["effective_tokens"],
        "num_documents_used": tokenization_stats["num_documents_used"],
        "raw_tokens_seen": tokenization_stats["raw_tokens_seen"],
        "evaluation_source": "fixed_token_budget",
    }

    print(
        f"{dataset_name}: "
        f"{len(tokenized_dataset):,} sequences | "
        f"{tokenization_stats['effective_tokens']:,} tokens",
        flush=True,
    )
    return dataloader, stats


# ============================================================
# PREPARE ALL EVALUATION DATA
# ============================================================

def prepare_all_evaluation_dataloaders(tokenizer, data_collator):
    """
    Prepare:
        - clean held-out ArXiv;
        - four natural deployment shifts;
        - five progressively corrupted versions of held-out ArXiv.
    """
    dataloaders = {}
    stats_rows = []

    # ========================================================
    # 1. CLEAN HELD-OUT ARXIV
    # ========================================================
    (
        arxiv_dataloader,
        clean_arxiv_eval_dataset,
        arxiv_stats,
    ) = prepare_arxiv_heldout_dataloader(
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    dataloaders["ArXiv"] = arxiv_dataloader
    stats_rows.append(arxiv_stats)

    # ========================================================
    # 2. NATURAL DOMAIN SHIFTS
    # ========================================================
    natural_dataset_names = [
        "DM Mathematics",
        "PubMed Abstracts",
        "BookCorpus",
        "YouTube Subtitles",
    ]
    for dataset_name in natural_dataset_names:
        dataset_info = DEPLOYMENT_DATASETS[dataset_name]
        dataloader, stats = prepare_shifted_dataloader(
            dataset_name=dataset_name,
            dataset_info=dataset_info,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        dataloaders[dataset_name] = dataloader
        stats_rows.append(stats)

    # ========================================================
    # 3. SYNTHETIC ARXIV NOISE
    # ========================================================
    for noise_level in NOISE_LEVELS:
        dataset_name = NOISY_DATASET_NAMES[noise_level]

        dataloader, stats = (
            prepare_noisy_arxiv_dataloader(
                clean_eval_dataset=clean_arxiv_eval_dataset,
                noise_level=noise_level,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
        )
        dataloaders[dataset_name] = dataloader
        stats_rows.append(stats)

    return dataloaders, stats_rows

# ============================================================
# TOKEN-KL DISTANCES COMPUTATION
# ===========================================================

def compute_all_token_kls_to_arxiv(deployment_dataloaders, tokenizer):
    """
    Compute KL(deployment || clean ArXiv)
    using exactly the same token budget for every deployment
    condition.
    """

    print(
        "\n"
        "============================================================\n"
        "COMPUTING TOKEN-KL TO CLEAN ARXIV\n"
        "============================================================",
        flush=True,
    )
    vocab_size = tokenizer.vocab_size

    # --------------------------------------------------------
    # Reference distribution: clean held-out ArXiv
    # --------------------------------------------------------

    arxiv_distribution, n_tokens = (
        sm.compute_dataset_token_distribution_with_budget(
            dataloader=deployment_dataloaders["ArXiv"],
            vocab_size=vocab_size,
            max_tokens=EVAL_MAX_TOKENS,
            epsilon=1e-8,
        )
    )

    if n_tokens != EVAL_MAX_TOKENS:
        raise RuntimeError(
            f"ArXiv Token-KL reference used "
            f"{n_tokens} tokens instead of "
            f"{EVAL_MAX_TOKENS}."
        )
    token_kls = {}

    # --------------------------------------------------------
    # All deployment conditions
    # --------------------------------------------------------

    for dataset_name in DATASET_ORDER:
        distribution, n_tokens = (
            sm.compute_dataset_token_distribution_with_budget(
                dataloader=deployment_dataloaders[dataset_name],
                vocab_size=vocab_size,
                max_tokens=EVAL_MAX_TOKENS,
                epsilon=1e-8,
            )
        )
        kl = sm.compute_kl(distribution, arxiv_distribution)
        token_kls[dataset_name] = kl
        # Update metadata used everywhere else.
        DEPLOYMENT_DATASETS[dataset_name]["token_kl"] = kl
        print(
            f"{dataset_name:25s} | "
            f"KL(deployment || ArXiv) = "
            f"{kl:.6f}",
            flush=True,
        )

    return token_kls

# ============================================================
# HUGGING FACE MODEL LOADING
# ============================================================

def format_lambda_for_hf(lambd):
    """
    Format lambda exactly as used in the Hugging Face model bank.

    Examples
    --------
    0.0  -> "0.0"
    0.02 -> "0.02"
    0.1  -> "0.1"
    1.0  -> "1.0"
    """
    return str(float(lambd))


def get_hf_model_subfolder(dataset_name, lambd):
    lambda_str = format_lambda_for_hf(lambd)
    return f"{HF_MODEL_BANK_ROOT}/{dataset_name}/lambda_{lambda_str}/model"


def load_lambda_model_from_hf(lambd, device):
    """
    Download one ArXiv lambda model from Hugging Face,
    load it into memory, then delete the temporary local copy.

    Only one lambda model is downloaded at a time.

    Repository:
        Emma974/shiftlab-model-bank

    Structure:
        model_bank/
            ArXiv/
                lambda_0.0/
                lambda_0.02/
                ...
    """

    subfolder = get_hf_model_subfolder(dataset_name=SOURCE_NAME, lambd=lambd)

    print(
        "\n"
        "============================================================\n"
        f"Loading ArXiv λ={lambd} from Hugging Face\n"
        "============================================================\n"
        f"Repository: {HF_REPO_ID}\n"
        f"Subfolder:  {subfolder}",
        flush=True,
    )

    # TemporaryDirectory is created under /tmp so the model never occupies persistent space in HOME.
    with tempfile.TemporaryDirectory(
        prefix=f"shiftlab_arxiv_lambda_{format_lambda_for_hf(lambd)}_",
        dir="/tmp",
    ) as temp_dir:

        # Download only the files belonging to this lambda model.
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="model",
            allow_patterns=[f"{subfolder}/*"],
            local_dir=temp_dir,
            token=True,
        )
        local_model_dir = os.path.join(temp_dir, subfolder)
        config_path = os.path.join(local_model_dir, "config.json")
        weights_path = os.path.join(local_model_dir, "model.safetensors")

        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Missing downloaded config.json:\n"
                f"{config_path}"
            )

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Missing downloaded model.safetensors:\n"
                f"{weights_path}"
            )

        print(f"Loading λ={lambd} into memory...", flush=True)

        model = AutoModelForCausalLM.from_pretrained(local_model_dir, local_files_only=True)
        model.to(device)
        model.eval()

    # The TemporaryDirectory is deleted here.
    # The PyTorch model itself remains loaded in RAM/GPU.
    print(
        f"ArXiv λ={lambd} loaded successfully. "
        "Temporary model files removed.",
        flush=True,
    )
    return model

# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_lambda_model(model, lambd, deployment_dataloaders, device):
    """
    Evaluate one ArXiv lambda model on all deployment
    conditions.

    The reported loss is the standard token-level cross-entropy
    returned by utils.evaluation().
    """
    rows = []
    for dataset_name in DATASET_ORDER:

        dataset_info = DEPLOYMENT_DATASETS[dataset_name]
        dataloader = deployment_dataloaders[dataset_name]

        print(
            "\n"
            f"[λ={lambd}] "
            f"Evaluating on {dataset_name} "
            f"(KL={dataset_info['token_kl']:.4f})...",
            flush=True,
        )

        loss, ppl, accuracy, total_tokens = utils.evaluation(model=model, dataloader=dataloader, device=device)

        rows.append(
        {
            "source_dataset": SOURCE_NAME,
            "lambda": lambd,
            "deployment_dataset": dataset_name,
            "shift_level": dataset_info["shift_level"],
            "shift_type": dataset_info.get("shift_type", "natural_domain"),
            "noise_probability": dataset_info.get("noise_probability", None),
            "token_kl_deployment_to_arxiv": dataset_info["token_kl"],
            "eval_loss": loss,
            "eval_ppl": ppl,
            "eval_accuracy": accuracy,
            "eval_tokens": total_tokens,
        }
    )

        print(
            f"λ={lambd:g} | "
            f"dataset={dataset_name} | "
            f"KL={dataset_info['token_kl']:.4f} | "
            f"loss={loss:.6f} | "
            f"ppl={ppl:.4f} | "
            f"tokens={total_tokens:,}",
            flush=True,
        )

    return rows


# ============================================================
# LOSS TABLE
# ============================================================

def build_loss_table(results_df):
    """
    Build the main robustness table:

        rows    = lambda
        columns = deployment datasets
        values  = cross-entropy loss
    """

    loss_table = results_df.pivot(index="lambda", columns="deployment_dataset", values="eval_loss")
    loss_table = loss_table.reindex(index=LAMBDAS, columns=DATASET_ORDER)

    return loss_table


# ============================================================
# PERPLEXITY TABLE
# ============================================================

def build_ppl_table(results_df):
    """
    Same layout as the loss table, but with perplexity.
    """

    ppl_table = results_df.pivot(index="lambda", columns="deployment_dataset", values="eval_ppl")
    ppl_table = ppl_table.reindex(index=LAMBDAS, columns=DATASET_ORDER)

    return ppl_table


# ============================================================
# BEST LAMBDA SUMMARY
# ============================================================

def build_best_lambda_summary(loss_table):
    """
    For every deployment dataset, identify:

        lambda* = argmin_lambda loss(lambda)

    and quantify the improvement relative to ERM lambda=0.
    """

    rows = []
    for dataset_name in DATASET_ORDER:
        losses = loss_table[dataset_name]
        best_lambda = losses.idxmin()

        best_loss = losses.loc[best_lambda]
        erm_loss = losses.loc[0.0]

        absolute_gain = erm_loss - best_loss
        relative_gain_percent = 100.0 * absolute_gain / erm_loss

        info = DEPLOYMENT_DATASETS[dataset_name]    

        rows.append(
        {
            "deployment_dataset": dataset_name,
            "shift_level": info["shift_level"],
            "shift_type": info.get("shift_type", "natural_domain"),
            "noise_probability": info.get("noise_probability", None),
            "token_kl_deployment_to_arxiv": info["token_kl"],
            "best_lambda": best_lambda,
            "best_loss": best_loss,
            "erm_lambda": 0.0,
            "erm_loss": erm_loss,
            "absolute_gain_vs_erm": absolute_gain,
            "relative_gain_vs_erm_percent": relative_gain_percent,
        }
    )

    return pd.DataFrame(rows)


# ============================================================
# COLORED LOSS TABLE
# ============================================================

def plot_colored_loss_table(loss_table, output_path):
    """
    Save a PNG table.

    For each deployment dataset, the cell corresponding to the
    minimum loss across lambda is highlighted in light green and
    displayed in bold.

    The header row is enlarged to accommodate:
        - dataset name;
        - shift level;
        - Token-KL value.
    """
    column_labels = []

    for dataset_name in DATASET_ORDER:
        info = DEPLOYMENT_DATASETS[dataset_name]
        column_labels.append(
            f"{dataset_name}\n"
            f"{info['shift_level']} shift\n"
            f"KL={info['token_kl']:.2f}"
        )
    row_labels = [f"λ={lambd:g}" for lambd in loss_table.index]

    # --------------------------------------------------------
    # CELL CONTENT
    # --------------------------------------------------------

    cell_text = []
    for lambd in loss_table.index:
        row = []
        for dataset_name in DATASET_ORDER:
            value = loss_table.loc[lambd, dataset_name]
            row.append(f"{value:.4f}")

        cell_text.append(row)

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig_height = 0.55 * len(loss_table) + 3.8
    fig, ax = plt.subplots(figsize=(24, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=column_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Increase the general size of the cells.
    table.scale(1.0, 1.7)

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    for col_idx in range(len(DATASET_ORDER)):
        header_cell = table[(0, col_idx)]
        # The header contains three lines, so it needs substantially more vertical space than data rows.
        header_cell.set_height(header_cell.get_height() * 2.0)
        header_text = header_cell.get_text()
        header_text.set_weight("bold")
        header_text.set_fontsize(11)
        header_text.set_verticalalignment("center")
        header_text.set_horizontalalignment("center")

    # --------------------------------------------------------
    # ROW LABELS
    # --------------------------------------------------------

    for row_idx in range(1, len(loss_table) + 1):
        row_label_cell = table[(row_idx, -1)]
        row_label_cell.get_text().set_weight("bold")
        row_label_cell.get_text().set_fontsize(11)

    # --------------------------------------------------------
    # HIGHLIGHT BEST LOSS IN EACH COLUMN
    # --------------------------------------------------------

    for col_idx, dataset_name in enumerate(DATASET_ORDER):
        best_lambda = loss_table[dataset_name].idxmin()
        row_idx = list(loss_table.index).index(best_lambda)
        # +1 because row zero is the header.
        cell = table[(row_idx + 1, col_idx)]
        cell.set_facecolor("#d9ead3")
        cell.get_text().set_weight("bold")

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    ax.set_title(
        "Robustness efficiency under increasing distribution shift\n"
        f"Model: {MODEL_TAG.replace('_', ' ').title()} — Source: ArXiv",
        fontsize=16,
        fontweight="bold",
        pad=28,
    )
    # Leave enough room around the enlarged header.
    plt.subplots_adjust(
        top=0.88,
        bottom=0.05,
        left=0.08,
        right=0.98,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# LATEX TABLE
# ============================================================

def save_latex_loss_table(loss_table, output_path):
    """
    Save the publication-ready LaTeX version of the main table.

    Requires:
        \\usepackage[table]{xcolor}

    Minimum loss in every column:
        light green background + bold text.
    """

    with open(output_path, "w",) as f:

        f.write(
            "% Requires: "
            "\\usepackage[table]{xcolor}\n"
        )

        f.write("\\begin{table*}[t]\n")

        f.write("\\centering\n")

        f.write(
            f"\\caption{{Cross-entropy loss of {MODEL_DISPLAY_NAME} models "
            "fine-tuned on ArXiv under increasing "
            "distribution shift. "
            "The shift is measured as "
            "$D_{\\mathrm{KL}}"
            "(p_{\\mathrm{deployment}}"
            "\\|p_{\\mathrm{ArXiv}})$. "
            "Lower is better. "
            "The best result for each deployment "
            "dataset is highlighted.}}\n"
        )

        f.write(f"\\label{{tab:robustness-efficiency-{MODEL_TAG.replace('_', '-')}}}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write(
            "\\begin{tabular}{l"
            + "c" * len(DATASET_ORDER)
            + "}\n"
        )

        f.write("\\hline\n")
        headers = ["$\\lambda$"]

        for dataset_name in DATASET_ORDER:
            info = DEPLOYMENT_DATASETS[dataset_name]
            latex_dataset_name = dataset_name.replace("%", "\\%")

            headers.append(
                f"{latex_dataset_name} "
                f"(KL={info['token_kl']:.2f})"
            )

        f.write(
            " & ".join(headers)
            + " \\\\\n"
        )
        f.write(
            "\\hline\n"
        )

        for lambd in loss_table.index:
            entries = [f"{lambd:g}"]

            for dataset_name in DATASET_ORDER:
                value = loss_table.loc[lambd, dataset_name]
                best_value = loss_table[dataset_name].min()

                if math.isclose(
                    value,
                    best_value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):

                    entries.append(
                        "\\cellcolor{green!20}"
                        f"\\textbf{{{value:.4f}}}"
                    )

                else:
                    entries.append(f"{value:.4f}")

            f.write(
                " & ".join(entries)
                + " \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table*}\n")

# ============================================================
# MAIN EXPERIMENT
# ============================================================

def robustness_efficiency():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    utils.set_seed(SEED)
    device = utils.get_device()

    print(
        "\n"
        "============================================================\n"
        "ROBUSTNESS EFFICIENCY EXPERIMENT\n"
        "============================================================\n"
        f"Source model family: {SOURCE_NAME}\n"
        f"Model architecture: {MODEL_NAME}\n"
        f"Number of lambda models: {len(LAMBDAS)}\n"
        f"Number of deployment datasets: {len(DATASET_ORDER)}\n"
        f"Device: {device}\n",
        flush=True,
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}",
            flush=True,
        )

        torch.cuda.reset_peak_memory_stats()

    # ========================================================
    # TOKENIZER / COLLATOR
    # ========================================================
    tokenizer, data_collator = setup_tokenizer_and_collator()

    # ========================================================
    # PREPARE EVALUATION DATASETS ONCE
    # ========================================================

    deployment_dataloaders, dataset_stats_rows = prepare_all_evaluation_dataloaders(tokenizer=tokenizer, data_collator=data_collator)
    token_kls = compute_all_token_kls_to_arxiv(deployment_dataloaders=deployment_dataloaders, tokenizer=tokenizer)

    # --------------------------------------------------------
    # Save dataset/evaluation metadata
    # --------------------------------------------------------
    for row in dataset_stats_rows:
        dataset_name = row["deployment_dataset"]
        row["token_kl_deployment_to_arxiv"] = token_kls[dataset_name]
        
    stats_df = pd.DataFrame(dataset_stats_rows)

    stats_path = os.path.join(OUTPUT_DIR, f"{MODEL_TAG}_evaluation_dataset_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    print(
        f"\nEvaluation dataset statistics saved to: "
        f"{stats_path}",
        flush=True,
    )

    # ========================================================
    # EVALUATE ALL LAMBDA MODELS
    # ========================================================

    results_path = os.path.join(OUTPUT_DIR, f"{MODEL_TAG}_robustness_efficiency_results.csv")

    # --------------------------------------------------------
    # RESUME FROM EXISTING RESULTS
    # --------------------------------------------------------

    if os.path.isfile(results_path):
        print(
            "\n"
            "============================================================\n"
            "EXISTING RESULTS FOUND — RESUMING EXPERIMENT\n"
            "============================================================\n"
            f"Results file: {results_path}",
            flush=True,
        )
        existing_results_df = pd.read_csv(results_path)
        required_columns = {
            "source_dataset",
            "lambda",
            "deployment_dataset",
            "shift_level",
            "shift_type",
            "noise_probability",
            "token_kl_deployment_to_arxiv",
            "eval_loss",
            "eval_ppl",
            "eval_accuracy",
            "eval_tokens",
        }
        missing_columns = (required_columns - set(existing_results_df.columns))
        if missing_columns:
            raise ValueError(
                "Existing results file is incompatible with "
                "the current experiment.\n"
                f"Missing columns: {sorted(missing_columns)}"
            )

        # Keep only results belonging to the current experiment.
        existing_results_df = existing_results_df[existing_results_df["source_dataset"] == SOURCE_NAME].copy()
        # Avoid duplicate rows if an older interrupted run happened
        # to save the same dataset/lambda combination twice.
        existing_results_df = (existing_results_df.drop_duplicates(
                subset=["source_dataset", "lambda", "deployment_dataset"],
                keep="last",
            )
            .reset_index(drop=True)
        )
        all_results = existing_results_df.to_dict(orient="records")

    else:
        print(
            "\nNo existing results found. "
            "Starting experiment from scratch.",
            flush=True,
        )
        all_results = []


    # --------------------------------------------------------
    # DETERMINE WHICH LAMBDAS ARE ALREADY COMPLETE
    # --------------------------------------------------------

    completed_lambdas = set()
    if all_results:
        current_results_df = pd.DataFrame(all_results)
        for lambd in LAMBDAS:
            lambda_df = current_results_df[
                np.isclose(
                    current_results_df["lambda"].astype(float),
                    float(lambd),
                    rtol=0.0,
                    atol=1e-12,
                )
            ]
            completed_datasets = set(lambda_df["deployment_dataset"].tolist())
            if set(DATASET_ORDER).issubset(completed_datasets):
                completed_lambdas.add(float(lambd))

    print(
        "\n===== Resume status =====",
        flush=True,
    )
    if completed_lambdas:
        print(
            "Already completed lambdas: "
            + ", ".join(
                f"{lambd:g}"
                for lambd in sorted(completed_lambdas)
            ),
            flush=True,
        )
    else:
        print(
            "Already completed lambdas: none",
            flush=True,
        )

    remaining_lambdas = [lambd for lambd in LAMBDAS if float(lambd) not in completed_lambdas]
    print(
        f"Completed: "
        f"{len(completed_lambdas)}/{len(LAMBDAS)}\n"
        f"Remaining: "
        f"{len(remaining_lambdas)}/{len(LAMBDAS)}",
        flush=True,
    )

    # ========================================================
    # EVALUATE MISSING LAMBDA MODELS
    # ========================================================

    for model_idx, lambd in enumerate(LAMBDAS, start=1):
        # ----------------------------------------------------
        # Skip fully completed lambda
        # ----------------------------------------------------
        if float(lambd) in completed_lambdas:
            print(
                "\n"
                "############################################################\n"
                f"MODEL {model_idx}/{len(LAMBDAS)} "
                f"— ArXiv λ={lambd}\n"
                "Already complete — SKIPPING\n"
                "############################################################",
                flush=True,
            )
            continue
        print(
            "\n"
            "############################################################\n"
            f"MODEL {model_idx}/{len(LAMBDAS)} "
            f"— ArXiv λ={lambd}\n"
            "############################################################",
            flush=True,
        )

        # ----------------------------------------------------
        # Load existing trained model from Hugging Face
        # ----------------------------------------------------

        model = load_lambda_model_from_hf(lambd=lambd, device=device)

        # ----------------------------------------------------
        # Evaluate on all deployment datasets
        # ----------------------------------------------------

        lambda_rows = evaluate_lambda_model(
            model=model,
            lambd=lambd,
            deployment_dataloaders=deployment_dataloaders,
            device=device,
        )

        # ----------------------------------------------------
        # Remove possible incomplete results for this lambda
        #
        # If a previous run stopped halfway through this lambda,
        # we recompute the complete lambda on all five datasets.
        # ----------------------------------------------------

        all_results = [
            row
            for row in all_results
            if not math.isclose(
                float(row["lambda"]),
                float(lambd),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        all_results.extend(lambda_rows)

        # ----------------------------------------------------
        # Incremental save
        # ----------------------------------------------------

        current_results_df = pd.DataFrame(all_results)
        current_results_df = (
            current_results_df
            .drop_duplicates(
                subset=["source_dataset", "lambda", "deployment_dataset"],
                keep="last",
            )
            .sort_values(
                by=["lambda", "deployment_dataset"]
            )
            .reset_index(drop=True)
        )
        current_results_df.to_csv(results_path, index=False)
        # Synchronize in-memory representation with saved CSV.
        all_results = current_results_df.to_dict(orient="records")

        print(
            f"\nλ={lambd:g} complete.\n"
            f"Intermediate results saved to: "
            f"{results_path}",
            flush=True,
        )

        # ----------------------------------------------------
        # GPU cleanup
        # ----------------------------------------------------

        del model
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ========================================================
    # COMPLETE RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path, index=False)

    # ========================================================
    # LOSS TABLE
    # ========================================================
    loss_table = build_loss_table(results_df)
    loss_table_path = os.path.join(OUTPUT_DIR, f"{MODEL_TAG}_robustness_efficiency_loss_table.csv")
    loss_table.to_csv(loss_table_path)

    # ========================================================
    # PERPLEXITY TABLE
    # ========================================================

    ppl_table = build_ppl_table(results_df)
    ppl_table_path = os.path.join(OUTPUT_DIR, f"{MODEL_TAG}_robustness_efficiency_ppl_table.csv")
    ppl_table.to_csv(ppl_table_path)

    # ========================================================
    # BEST LAMBDA SUMMARY
    # ========================================================

    best_summary = build_best_lambda_summary(loss_table)
    best_summary_path = os.path.join(OUTPUT_DIR, f"{MODEL_TAG}_best_lambda_by_shift.csv")
    best_summary.to_csv(best_summary_path, index=False)

    # ========================================================
    # COLORED TABLE
    # ========================================================

    plot_colored_loss_table(
        loss_table=loss_table,
        output_path=os.path.join(
            OUTPUT_DIR,
            f"{MODEL_TAG}_robustness_efficiency_table.png",
        ),
    )

    # ========================================================
    # LATEX TABLE
    # ========================================================

    save_latex_loss_table(
        loss_table=loss_table,
        output_path=os.path.join(
            OUTPUT_DIR,
            f"{MODEL_TAG}_robustness_efficiency_table.tex",
        ),
    )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print(
        "\n"
        "============================================================\n"
        "BEST MODEL FOR EACH DEPLOYMENT DATASET\n"
        "============================================================",
        flush=True,
    )

    print(
        best_summary[
            [
                "deployment_dataset",
                "shift_level",
                "token_kl_deployment_to_arxiv",
                "best_lambda",
                "best_loss",
                "erm_loss",
                "relative_gain_vs_erm_percent",
            ]
        ].to_string(
            index=False
        ),
        flush=True,
    )

    # ========================================================
    # TIME / MEMORY
    # ========================================================

    elapsed = time.time() - start_time

    print(
        "\n"
        "============================================================\n"
        "ROBUSTNESS EFFICIENCY COMPLETE\n"
        "============================================================\n"
        f"Total time: "
        f"{elapsed / 60:.2f} minutes\n"
        f"Results directory: "
        f"{OUTPUT_DIR}\n",
        flush=True,
    )

    if device.type == "cuda":

        peak_memory = torch.cuda.max_memory_allocated() / 1024**3

        print(
            f"Peak GPU memory: "
            f"{peak_memory:.2f} GB",
            flush=True,
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    robustness_efficiency()