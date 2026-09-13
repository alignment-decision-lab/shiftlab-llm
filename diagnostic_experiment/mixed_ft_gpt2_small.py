"""
Mixed Fine-Tuning for GPT-2 Small.

This script trains the Mixed-FT baseline associated with an EXPERIMENT_CONFIG
defined in experimental_pipeline.py.

For a selected experiment, it:
1. reads the source datasets from EXPERIMENT_CONFIG["sources"];
2. retrieves their Hugging Face dataset definitions from DATASET_REGISTRY;
3. builds an exactly uniform source mixture with a fixed total token budget;
4. fine-tunes the pretrained GPT-2 Small model;
5. temporarily saves the trained checkpoint and training statistics locally;
6. uploads everything to the Mixed-FT path specified in
   EXPERIMENT_CONFIG["mixed_ft"]["subfolder"];
7. verifies that the model exists on Hugging Face;
8. deletes the temporary local files only after a successful verification.

The script currently supports GPT-2 Small only.

Examples:
    python diagnostic_experiment/mixed_ft_gpt2_small.py --experiment 1

    python diagnostic_experiment/mixed_ft_gpt2_small.py \
        --experiment exp2_github_freelaw_stackexchange_dm

The source list and Mixed-FT path must therefore be defined only once, inside
experimental_pipeline.py.
"""

import os
import gc
import math
import time
import random
import shutil
import argparse
import tempfile

# Create a run-specific temporary directory instead of using a user-specific path.
TEMP_ROOT = tempfile.mkdtemp(prefix="shiftlab_mixed_ft_")
os.environ["HF_DATASETS_CACHE"] = os.path.join(TEMP_ROOT, "hf_datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(TEMP_ROOT, "hf_transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(TEMP_ROOT, "hf_hub")

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

import experimental_pipeline as pipeline


# ============================================================
# CONFIG
# ============================================================

HF_REPO_ID = "alignment-decision-lab/robustness-model-bank"

TRAINING_CONFIG = {
    "seed": 42,
    "context_length": 512,
    "max_tokens": 5_120_000,
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 5e-5,
    "weight_decay": 0.0,
    "val_split_ratio": 0.1,
    "max_epochs": 2,
}


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

def get_available_experiments():
    """Automatically collect EXPERIMENT_X_CONFIG dictionaries from experimental_pipeline.py."""
    configs = {}
    numbered = []

    for name, value in vars(pipeline).items():
        if not (name.startswith("EXPERIMENT_") and name.endswith("_CONFIG") and isinstance(value, dict)):
            continue
        if "experiment_name" not in value:
            continue

        configs[value["experiment_name"]] = value

        middle = name[len("EXPERIMENT_"):-len("_CONFIG")]
        if middle.isdigit():
            numbered.append((int(middle), value))

    for number, config in sorted(numbered):
        configs[str(number)] = config

    return configs


def get_experiment_config(identifier):
    configs = get_available_experiments()

    if identifier not in configs:
        names = sorted({config["experiment_name"] for config in configs.values()})
        raise ValueError(
            f"Unknown experiment '{identifier}'.\n"
            f"Available experiments:\n  - " + "\n  - ".join(names)
        )

    return configs[identifier]


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def normalize_source_name(source):
    return source.replace(" ", "_").replace("/", "_")


def build_mixed_ft_name(sources):
    return "Mixed_FT_" + "_".join(normalize_source_name(source) for source in sources)


def expected_mixed_ft_subfolder(experiment_config):
    model_key = experiment_config["model"]

    if model_key not in pipeline.MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'.")

    if model_key != "small":
        raise ValueError(
            f"mixed_ft_gpt2_small.py supports GPT-2 Small only. "
            f"Received model='{model_key}'."
        )

    prefix = pipeline.MODEL_REGISTRY[model_key]["bank_prefix"]
    return f"{prefix}/{build_mixed_ft_name(experiment_config['sources'])}/model"


def validate_experiment_config(experiment_config):
    required = ["experiment_name", "model", "sources", "mixed_ft"]
    missing = [key for key in required if key not in experiment_config]

    if missing:
        raise ValueError(f"EXPERIMENT_CONFIG is missing required keys: {missing}")

    name = experiment_config["experiment_name"]
    sources = experiment_config["sources"]

    if experiment_config["model"] != "small":
        raise ValueError(
            f"{name}: this script currently supports only model='small'. "
            f"Got model='{experiment_config['model']}'."
        )

    if not sources:
        raise ValueError(f"{name}: sources list is empty.")

    unknown = [source for source in sources if source not in pipeline.DATASET_REGISTRY]
    if unknown:
        raise ValueError(f"{name}: unknown source datasets: {unknown}")

    if "subfolder" not in experiment_config["mixed_ft"]:
        raise ValueError(f"{name}: EXPERIMENT_CONFIG['mixed_ft']['subfolder'] is missing.")

    expected = expected_mixed_ft_subfolder(experiment_config)
    configured = experiment_config["mixed_ft"]["subfolder"].strip("/")

    if configured != expected:
        raise ValueError(
            f"{name}: Mixed-FT path/source mismatch.\n"
            f"Sources:  {sources}\n"
            f"Expected: {expected}\n"
            f"Got:      {configured}"
        )

    if pipeline.MODEL_REGISTRY["small"]["model_name"] != "gpt2":
        raise ValueError("MODEL_REGISTRY['small']['model_name'] must be 'gpt2'.")

    return expected


def get_source_datasets(experiment_config):
    return {
        source: pipeline.DATASET_REGISTRY[source]["dataset_config"]
        for source in experiment_config["sources"]
    }


def setup_model_and_tokenizer(device):
    model_name = pipeline.MODEL_REGISTRY["small"]["model_name"]
    print(f"\nLoading pretrained model: {model_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return model, tokenizer, data_collator


def load_source_dataset(dataset_config):
    print(
        f"Loading {dataset_config['name']} "
        f"(split={dataset_config['split']}, streaming={dataset_config.get('streaming', False)})",
        flush=True,
    )

    return load_dataset(
        dataset_config["name"],
        dataset_config.get("config"),
        split=dataset_config["split"],
        streaming=dataset_config.get("streaming", False),
    )


def extract_text(example, dataset_config):
    if dataset_config.get("type") == "translation":
        translations = example.get("translation")
        if not isinstance(translations, dict):
            return None
        text = translations.get(dataset_config["language"])
    else:
        text = example.get(dataset_config.get("text_column", "text"))

    if isinstance(text, list):
        text = " ".join(map(str, text))
    if text is None:
        return None

    text = str(text).strip()
    return text if len(text) > 5 else None


# ============================================================
# TOKEN-BUDGET CONSTRUCTION
# ============================================================

def tokenize_source_with_token_budget(raw_dataset, tokenizer, dataset_config, context_length, token_budget):
    max_sequences = token_budget // context_length
    if max_sequences <= 0:
        raise ValueError(f"Token budget {token_budget} is smaller than context length {context_length}.")

    usable_token_budget = max_sequences * context_length
    input_blocks, attention_blocks = [], []
    token_buffer = []
    buffer_start = 0
    num_documents = 0
    num_raw_tokens = 0
    num_sequences = 0
    eos_token_id = tokenizer.eos_token_id

    if eos_token_id is None:
        raise ValueError("Tokenizer does not define eos_token_id.")

    for example in raw_dataset:
        if num_sequences >= max_sequences:
            break

        text = extract_text(example, dataset_config)
        if text is None:
            continue

        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]

        if not token_ids:
            continue

        num_documents += 1
        num_raw_tokens += len(token_ids)
        token_ids.append(eos_token_id)
        token_buffer.extend(token_ids)

        while len(token_buffer) - buffer_start >= context_length and num_sequences < max_sequences:
            end = buffer_start + context_length
            input_blocks.append(token_buffer[buffer_start:end])
            attention_blocks.append([1] * context_length)
            buffer_start = end
            num_sequences += 1

        if buffer_start >= 100_000:
            token_buffer = token_buffer[buffer_start:]
            buffer_start = 0

    effective_tokens = num_sequences * context_length

    if num_sequences < max_sequences:
        print(
            f"WARNING: dataset ended after {num_sequences:,}/{max_sequences:,} requested sequences.",
            flush=True,
        )

    dataset = Dataset.from_dict({"input_ids": input_blocks, "attention_mask": attention_blocks})
    stats = {
        "num_documents_used": num_documents,
        "raw_tokens_seen": num_raw_tokens,
        "requested_token_budget": token_budget,
        "usable_token_budget": usable_token_budget,
        "num_sequences": num_sequences,
        "context_length": context_length,
        "effective_tokens": effective_tokens,
    }
    return dataset, stats


def resample_dataset_to_size(dataset, target_size, seed):
    current_size = len(dataset)

    if target_size <= 0:
        raise ValueError("target_size must be strictly positive.")
    if current_size == 0:
        raise ValueError("Cannot resample an empty dataset.")

    rng = np.random.default_rng(seed)

    if current_size >= target_size:
        indices = rng.choice(current_size, size=target_size, replace=False)
        return dataset.select(indices.tolist())

    base_indices = np.arange(current_size)
    extra_indices = rng.choice(current_size, size=target_size - current_size, replace=True)
    indices = np.concatenate([base_indices, extra_indices])
    rng.shuffle(indices)

    return dataset.select(indices.tolist())


# ============================================================
# UNIFORM MIXTURE
# ============================================================

def build_uniform_train_val_mixture(tokenizer, config, source_datasets):
    seed = int(config["seed"])
    context_length = int(config["context_length"])
    total_token_budget = int(config["max_tokens"])
    val_ratio = float(config["val_split_ratio"])
    num_sources = len(source_datasets)

    if num_sources <= 0:
        raise ValueError("source_datasets is empty.")

    total_sequences = total_token_budget // context_length

    if total_sequences < 2 * num_sources:
        raise ValueError("Token budget is too small for train/validation data for all sources.")

    effective_total_tokens = total_sequences * context_length
    base_sequences = total_sequences // num_sources
    remainder = total_sequences % num_sources
    source_names = list(source_datasets.keys())

    target_total_per_source = {
        source: base_sequences + (1 if i < remainder else 0)
        for i, source in enumerate(source_names)
    }

    total_val_sequences = int(round(total_sequences * val_ratio))
    total_train_sequences = total_sequences - total_val_sequences
    base_train = total_train_sequences // num_sources
    train_remainder = total_train_sequences % num_sources

    target_train_per_source, target_val_per_source = {}, {}

    for i, source in enumerate(source_names):
        target_train = base_train + (1 if i < train_remainder else 0)
        target_total = target_total_per_source[source]
        target_val = target_total - target_train

        if target_train <= 0 or target_val <= 0:
            raise ValueError(
                f"Invalid train/validation allocation for {source}: "
                f"train={target_train}, val={target_val}."
            )

        target_train_per_source[source] = target_train
        target_val_per_source[source] = target_val

    print(
        "\n============================================================\n"
        "UNIFORM MIXED-FT DATASET\n"
        "============================================================\n"
        f"Sources:                       {num_sources}\n"
        f"Requested total token budget:  {total_token_budget:,}\n"
        f"Context length:                {context_length:,}\n"
        f"Total sequences:               {total_sequences:,}\n"
        f"Effective total tokens:        {effective_total_tokens:,}\n"
        f"Train sequences:               {total_train_sequences:,}\n"
        f"Validation sequences:          {total_val_sequences:,}\n"
        f"Validation ratio:              {val_ratio:.3f}\n"
        "------------------------------------------------------------",
        flush=True,
    )

    for source in source_names:
        print(
            f"{source}: total={target_total_per_source[source]:,}, "
            f"train={target_train_per_source[source]:,}, "
            f"val={target_val_per_source[source]:,}",
            flush=True,
        )

    print("============================================================\n", flush=True)

    source_train_datasets, source_val_datasets, source_stats = [], [], []

    for source_idx, (source_name, dataset_config) in enumerate(source_datasets.items()):
        print(
            "\n------------------------------------------------------------\n"
            f"SOURCE: {source_name}\n"
            "------------------------------------------------------------",
            flush=True,
        )

        target_total = target_total_per_source[source_name]
        target_train = target_train_per_source[source_name]
        target_val = target_val_per_source[source_name]

        raw_dataset = load_source_dataset(dataset_config)
        tokenized_dataset, stats = tokenize_source_with_token_budget(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            dataset_config=dataset_config,
            context_length=context_length,
            token_budget=target_total * context_length,
        )

        available_sequences = len(tokenized_dataset)

        if available_sequences == 0:
            raise RuntimeError(f"{source_name}: no complete {context_length}-token sequence could be constructed.")

        if available_sequences < 2:
            raise RuntimeError(
                f"{source_name}: only {available_sequences} complete sequence is available. "
                "At least 2 are required to keep training and validation disjoint."
            )

        if available_sequences >= target_val + 1:
            unique_val_size = target_val
        else:
            unique_val_size = int(round(available_sequences * val_ratio))
            unique_val_size = max(1, unique_val_size)
            unique_val_size = min(unique_val_size, available_sequences - 1)

        rng = np.random.default_rng(seed + source_idx)
        shuffled_indices = rng.permutation(available_sequences)
        val_indices = shuffled_indices[:unique_val_size]
        train_indices = shuffled_indices[unique_val_size:]

        unique_train = tokenized_dataset.select(train_indices.tolist())
        unique_val = tokenized_dataset.select(val_indices.tolist())

        source_train = resample_dataset_to_size(
            unique_train, target_train, seed + 10_000 + source_idx
        )
        source_val = resample_dataset_to_size(
            unique_val, target_val, seed + 20_000 + source_idx
        )

        train_was_upsampled = len(unique_train) < target_train
        val_was_upsampled = len(unique_val) < target_val

        source_train_datasets.append(source_train)
        source_val_datasets.append(source_val)

        stats.update({
            "source": source_name,
            "hf_dataset": dataset_config["name"],
            "hf_split": dataset_config["split"],
            "target_total_sequences": target_total,
            "target_train_sequences": target_train,
            "target_val_sequences": target_val,
            "available_unique_sequences": available_sequences,
            "unique_train_sequences": len(unique_train),
            "unique_val_sequences": len(unique_val),
            "num_train_sequences": len(source_train),
            "num_val_sequences": len(source_val),
            "train_was_upsampled": train_was_upsampled,
            "val_was_upsampled": val_was_upsampled,
            "train_extra_resampled_sequences": max(0, target_train - len(unique_train)),
            "val_extra_resampled_sequences": max(0, target_val - len(unique_val)),
            "train_upsampling_factor": len(source_train) / len(unique_train),
            "val_upsampling_factor": len(source_val) / len(unique_val),
            "train_tokens": len(source_train) * context_length,
            "val_tokens": len(source_val) * context_length,
            "final_mixture_tokens": (len(source_train) + len(source_val)) * context_length,
        })
        source_stats.append(stats)

        print(
            f"{source_name}\n"
            f"  Available unique sequences:  {available_sequences:,}\n"
            f"  Unique train sequences:      {len(unique_train):,}\n"
            f"  Unique validation sequences: {len(unique_val):,}\n"
            f"  Final train sequences:       {len(source_train):,}\n"
            f"  Final validation sequences:  {len(source_val):,}\n"
            f"  Train upsampled:              {train_was_upsampled}\n"
            f"  Validation upsampled:         {val_was_upsampled}",
            flush=True,
        )

        del raw_dataset, tokenized_dataset, unique_train, unique_val

    train_dataset = concatenate_datasets(source_train_datasets).shuffle(seed=seed)
    val_dataset = concatenate_datasets(source_val_datasets).shuffle(seed=seed)

    if len(train_dataset) != total_train_sequences:
        raise RuntimeError(
            f"Unexpected final training size: {len(train_dataset):,} instead of {total_train_sequences:,}."
        )

    if len(val_dataset) != total_val_sequences:
        raise RuntimeError(
            f"Unexpected final validation size: {len(val_dataset):,} instead of {total_val_sequences:,}."
        )

    final_total_sequences = len(train_dataset) + len(val_dataset)
    final_total_tokens = final_total_sequences * context_length

    if final_total_sequences != total_sequences:
        raise RuntimeError("Final sequence count does not match requested total.")

    print(
        "\n============================================================\n"
        "FINAL MIXTURE\n"
        "============================================================\n"
        f"Train sequences:       {len(train_dataset):,}\n"
        f"Validation sequences:  {len(val_dataset):,}\n"
        f"Total sequences:       {final_total_sequences:,}\n"
        f"Total tokens:          {final_total_tokens:,}\n"
        "============================================================\n",
        flush=True,
    )

    return train_dataset, val_dataset, source_stats


# ============================================================
# DATALOADERS / EVALUATION / TRAINING
# ============================================================

def create_dataloaders(train_dataset, val_dataset, data_collator, config):
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=data_collator,
        generator=generator,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=data_collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=data_collator,
    )

    return train_loader, train_eval_loader, val_loader


def evaluation(model, dataloader, device):
    model.eval()
    total_loss, total_tokens, correct = 0.0, 0, 0
    loss_function = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)

            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            batch_loss = loss_function(flat_logits, flat_labels)
            mask = flat_labels != -100

            total_loss += batch_loss.item()
            total_tokens += mask.sum().item()
            predictions = flat_logits.argmax(dim=-1)
            correct += (predictions[mask] == flat_labels[mask]).sum().item()

    if total_tokens == 0:
        raise ValueError("No valid tokens found.")

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss), correct / total_tokens, total_tokens


def train_mixed_ft(model, train_loader, train_eval_loader, val_loader, optimizer, device, config):
    accumulation_steps = int(config["gradient_accumulation_steps"])
    max_epochs = int(config["max_epochs"])

    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")

    global_step = 0
    cumulative_tokens_seen = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        num_batches = len(train_loader)
        total_optimization_loss = 0.0

        for i, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            total_optimization_loss += loss.item()

            if "attention_mask" in batch:
                cumulative_tokens_seen += batch["attention_mask"].sum().item()
            else:
                cumulative_tokens_seen += batch["input_ids"].numel()

            group_start = (i // accumulation_steps) * accumulation_steps
            current_group_size = min(accumulation_steps, num_batches - group_start)
            (loss / current_group_size).backward()

            if (i + 1) % accumulation_steps == 0 or i == num_batches - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        train_loss, train_ppl, train_acc, train_tokens = evaluation(model, train_eval_loader, device)
        val_loss, val_ppl, val_acc, val_tokens = evaluation(model, val_loader, device)
        avg_optimization_loss = total_optimization_loss / num_batches

        history.append({
            "epoch": epoch + 1,
            "optimizer_step": global_step,
            "cumulative_tokens_seen": cumulative_tokens_seen,
            "optimization_loss": avg_optimization_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_ppl": train_ppl,
            "val_ppl": val_ppl,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_eval_tokens": train_tokens,
            "val_eval_tokens": val_tokens,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch + 1}/{max_epochs} | step={global_step} | "
            f"tokens_seen={cumulative_tokens_seen:,} | "
            f"optim_loss={avg_optimization_loss:.4f} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_ppl={val_ppl:.2f} | val_acc={val_acc:.4f}",
            flush=True,
        )

    return history


# ============================================================
# SAVE / UPLOAD / VERIFY
# ============================================================

def save_and_upload(model, tokenizer, history, source_stats, config, experiment_config):
    api = HfApi()
    experiment_name = experiment_config["experiment_name"]
    sources = experiment_config["sources"]
    hf_subfolder = experiment_config["mixed_ft"]["subfolder"].strip("/")
    mixed_ft_name = build_mixed_ft_name(sources)

    model_dir = os.path.join(TEMP_ROOT, mixed_ft_name, "model")
    os.makedirs(model_dir, exist_ok=True)

    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)

    pd.DataFrame(history).to_csv(os.path.join(model_dir, "training_history.csv"), index=False)
    pd.DataFrame(source_stats).to_csv(os.path.join(model_dir, "source_dataset_stats.csv"), index=False)

    last_epoch = history[-1]
    summary = {
        "method": "Mixed FT",
        "experiment_name": experiment_name,
        "model": "small",
        "model_name": "gpt2",
        "hf_repo_id": HF_REPO_ID,
        "hf_subfolder": hf_subfolder,
        "sources": " | ".join(sources),
        "num_sources": len(sources),
        "mixture_type": "uniform_source_mixture",
        "seed": config["seed"],
        "context_length": config["context_length"],
        "requested_total_max_tokens": config["max_tokens"],
        "effective_total_source_tokens": sum(stats["final_mixture_tokens"] for stats in source_stats),
        "batch_size": config["batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "effective_batch_size": config["batch_size"] * config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "val_split_ratio": config["val_split_ratio"],
        "max_epochs": config["max_epochs"],
        "final_optimizer_step": last_epoch["optimizer_step"],
        "final_tokens_seen": last_epoch["cumulative_tokens_seen"],
        "final_train_loss": last_epoch["train_loss"],
        "final_val_loss": last_epoch["val_loss"],
        "final_train_ppl": last_epoch["train_ppl"],
        "final_val_ppl": last_epoch["val_ppl"],
        "final_train_acc": last_epoch["train_acc"],
        "final_val_acc": last_epoch["val_acc"],
    }

    pd.DataFrame([summary]).to_csv(os.path.join(model_dir, "training_summary.csv"), index=False)

    print(
        "\n============================================================\n"
        "TEMPORARY MIXED-FT CHECKPOINT SAVED\n"
        "============================================================\n"
        f"Temporary path: {model_dir}\n"
        f"HF target:      {HF_REPO_ID}/{hf_subfolder}\n"
        "============================================================",
        flush=True,
    )

    api.upload_folder(
        repo_id=HF_REPO_ID,
        repo_type="model",
        folder_path=model_dir,
        path_in_repo=hf_subfolder,
        commit_message=f"Add GPT-2 Small Mixed FT for {experiment_name}",
    )

    required_files = [
        f"{hf_subfolder}/config.json",
        f"{hf_subfolder}/model.safetensors",
    ]

    missing = [
        filename
        for filename in required_files
        if not api.file_exists(repo_id=HF_REPO_ID, filename=filename, repo_type="model")
    ]

    if missing:
        raise RuntimeError(
            "Upload finished but Hugging Face verification failed.\n"
            f"Missing files: {missing}\n"
            f"Temporary checkpoint kept at: {model_dir}"
        )

    print(
        "\n============================================================\n"
        "HUGGING FACE UPLOAD VERIFIED\n"
        "============================================================\n"
        f"{required_files[0]}: OK\n"
        f"{required_files[1]}: OK\n"
        "The temporary checkpoint can now be safely deleted.\n"
        "============================================================",
        flush=True,
    )

    return model_dir


# ============================================================
# RUN
# ============================================================

def run_experiment(experiment_config):
    validate_experiment_config(experiment_config)

    device = get_device()
    training_config = dict(TRAINING_CONFIG)
    training_config["seed"] = int(experiment_config.get("seed", TRAINING_CONFIG["seed"]))

    name = experiment_config["experiment_name"]
    sources = experiment_config["sources"]
    hf_subfolder = experiment_config["mixed_ft"]["subfolder"]

    set_seed(training_config["seed"])

    print(
        "\n############################################################\n"
        "GPT-2 SMALL MIXED-FT TRAINING\n"
        "############################################################\n"
        f"Experiment: {name}\n"
        f"Device:     {device}\n"
        f"Model:      gpt2\n"
        f"Sources:    {', '.join(sources)}\n"
        f"HF target:  {HF_REPO_ID}/{hf_subfolder}\n"
        f"Temp root:  {TEMP_ROOT}\n"
        "------------------------------------------------------------\n"
        f"Token budget:          {training_config['max_tokens']:,}\n"
        f"Context length:        {training_config['context_length']}\n"
        f"Batch size:            {training_config['batch_size']}\n"
        f"Gradient accumulation: {training_config['gradient_accumulation_steps']}\n"
        f"Effective batch size:  "
        f"{training_config['batch_size'] * training_config['gradient_accumulation_steps']}\n"
        f"Learning rate:         {training_config['learning_rate']}\n"
        f"Validation split:      {training_config['val_split_ratio']}\n"
        f"Epochs:                {training_config['max_epochs']}\n"
        f"Seed:                  {training_config['seed']}\n"
        "############################################################",
        flush=True,
    )

    start_time = time.time()
    source_datasets = get_source_datasets(experiment_config)

    model, tokenizer, data_collator = setup_model_and_tokenizer(device)

    train_dataset, val_dataset, source_stats = build_uniform_train_val_mixture(
        tokenizer=tokenizer,
        config=training_config,
        source_datasets=source_datasets,
    )

    train_loader, train_eval_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        data_collator,
        training_config,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    history = train_mixed_ft(
        model=model,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=training_config,
    )

    model_dir = save_and_upload(
        model=model,
        tokenizer=tokenizer,
        history=history,
        source_stats=source_stats,
        config=training_config,
        experiment_config=experiment_config,
    )

    elapsed = time.time() - start_time

    del model, tokenizer, data_collator
    del train_dataset, val_dataset, train_loader, train_eval_loader, val_loader
    del optimizer, history, source_stats

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    shutil.rmtree(TEMP_ROOT)

    print(
        "\n============================================================\n"
        "MIXED-FT COMPLETE\n"
        "============================================================\n"
        f"Experiment: {name}\n"
        f"Total time: {elapsed / 60:.2f} minutes\n"
        f"HF model:   {HF_REPO_ID}/{hf_subfolder}\n"
        f"Temporary files deleted: {TEMP_ROOT}\n"
        "============================================================",
        flush=True,
    )

    return {
        "status": "completed",
        "experiment_name": name,
        "sources": sources,
        "hf_subfolder": hf_subfolder,
        "total_time_sec": elapsed,
    }


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the GPT-2 Small Mixed-FT model associated with an experiment configuration."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help=(
            "Experiment number or experiment_name defined in experimental_pipeline.py. "
            "Examples: 1, 2, 3, exp2_github_freelaw_stackexchange_dm."
        ),
    )
    return parser.parse_args()


def main(experiment_config=None):
    if experiment_config is None:
        args = parse_args()
        experiment_config = get_experiment_config(args.experiment)

    try:
        return run_experiment(experiment_config)

    except Exception:
        print(
            "\n============================================================\n"
            "MIXED-FT FAILED\n"
            "============================================================\n"
            "The temporary directory was NOT deleted so that cached data or\n"
            "a locally saved checkpoint can be recovered if available.\n"
            f"Temporary directory: {TEMP_ROOT}\n"
            "============================================================",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()