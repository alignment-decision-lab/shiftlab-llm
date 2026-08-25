# First experiment with DistilGPT-2(82M parameters) using Wikisource Dataset.

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset, Dataset, concatenate_datasets
from itertools import islice
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import time
import math
import matplotlib.pyplot as plt
import os
import yaml
import argparse
import re
import random
import plotting

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True) # in the terminal
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    return config

# --DEVICE--
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}
    
# --DATASETS--
def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config["models"]["name"])
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_dataset(dataset, tokenizer, max_length):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset

def clean_wikisource_text(example):
    text = example["text"]

    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE) # Remove markdown headers (lines starting with #)
    text = re.sub(r"<[^>]+>", "", text) # Remove HTML tags
    text = re.sub(r"Sommaire\s*:.*", "", text) # Remove "Sommaire" section and everything that follows (common in Wikisource)
    text = re.sub(r"\n{3,}", "\n\n", text) # Replace multiple consecutive newlines with just two (to avoid excessive blank lines)
    text = re.sub(r"[ \t]+", " ", text) # Replace multiple spaces/tabs with a single space

    example["text"] = text.strip()
    return example

_VALID_CLEANING_MODES = {"none", "wikisource", "basic"}


def _resolve_max_length(config):
    """Resolve the tokenization sequence length.

    Requires the canonical 'max_length' key, but tolerates the older
    'context_length' name (used by one legacy config) with a visible
    warning instead of a silent KeyError deep in tokenization.
    """
    training_cfg = config["training"]

    if "max_length" in training_cfg:
        return training_cfg["max_length"]

    if "context_length" in training_cfg:
        print(
            "[config] 'training.context_length' is deprecated, use "
            "'training.max_length' instead. Using it as max_length for now."
        )
        return training_cfg["context_length"]

    raise ValueError(
        "config['training'] must define 'max_length' (the tokenization "
        "sequence length in tokens). Found neither 'max_length' nor the "
        "deprecated 'context_length'."
    )


def _load_raw_dataset(dataset_cfg):
    """Load the raw HF dataset, supporting the two loading modes used across
    configs/diagnostic/{wikitext,wikisource,histtext}/:

    - hub name + config name (wikitext, wikisource): load_dataset(name, config)
    - hub name + data_files glob (histtext):         load_dataset(name, data_files=...)

    These two modes are mutually exclusive in the HF `datasets` API, so a
    config that sets both is almost certainly a mistake and is rejected
    rather than silently picking one.
    """
    name = dataset_cfg["name"]
    config_name = dataset_cfg.get("config")
    data_files = dataset_cfg.get("data_files")

    if config_name is not None and data_files is not None:
        raise ValueError(
            f"config['dataset'] for {name!r} sets both 'config' and "
            "'data_files' -- these are mutually exclusive loading modes, "
            "pick one."
        )

    if data_files is not None:
        return load_dataset(name, data_files=data_files)

    return load_dataset(name, config_name)


def load_difficulty_shift_dataset(config):
    """Load and prepare the training corpus for a difficulty-shift run.

    This is the single generic loader for every dataset family under
    configs/diagnostic/{wikitext,wikisource,histtext}/. Dataset-specific
    behavior (e.g. wikisource markup cleaning) is controlled by
    config['dataset']['cleaning'], not by a separate function per dataset
    despite what this function used to be named.
    """
    dataset_cfg = config["dataset"]

    cleaning = dataset_cfg.get("cleaning", "none")
    if cleaning not in _VALID_CLEANING_MODES:
        raise ValueError(
            f"config['dataset']['cleaning'] = {cleaning!r} is not one of "
            f"{sorted(_VALID_CLEANING_MODES)} (dataset: {dataset_cfg.get('name')!r})."
        )

    dataset = _load_raw_dataset(dataset_cfg)
    dataset = dataset[dataset_cfg.get("split", "train")]

    if cleaning == "wikisource":
        dataset = dataset.map(clean_wikisource_text)

    if dataset_cfg["text_column"] != "text":
        dataset = dataset.rename_column(dataset_cfg["text_column"], "text")
    dataset = dataset.filter(lambda x: len(x["text"]) > 5)

    if config["training"]["dataset_size"] is not None:
        dataset = dataset.select(range(min(config["training"]["dataset_size"], len(dataset))))

    return dataset

def compute_sample_losses(model, tokenized_dataset, data_collator, device, config):
    dataloader = DataLoader(tokenized_dataset, batch_size=config["training"]["batch_size"], collate_fn=data_collator, shuffle=False)

    model.eval()
    losses = []
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)

            logits = outputs.logits
            labels = batch["labels"]

            logits = logits[:, :-1, :] #shifted.
            labels = labels[:, 1:]

            mask = labels != -100

            logits = logits.reshape(-1, logits.size(-1))# logits:(batch, sequence_length, vocab_size) -> (batch*sequence_length, vocab_size)
            labels = labels.reshape(-1) # labels:(batch, sequence_length) -> (batch*sequence_length)

            token_losses = loss_function(logits, labels) # loss per token
            token_losses = token_losses.reshape(batch["input_ids"].size(0), -1) # reshape back to (batch, sequence_length)
            token_losses = token_losses * mask # puting the pad loss to 0.

            sample_loss = token_losses.sum(dim=1) / mask.sum(dim=1) # compute mean loss per sample without the loss of padding token.

            losses.extend(sample_loss.cpu().tolist()) # concatenate the loss per sample from each batch.
    return losses

def split_easy_hard(tokenized_dataset, losses, alpha):
    assert len(tokenized_dataset) == len(losses)
    assert 0 <= alpha <= 1

    # Sort dataset:
    indexed_losses = []
    for i in range(len(losses)):
        indexed_losses.append((i, losses[i]))
    indexed_losses.sort(key=lambda x : x[1]) # sort according to the loss from easy to hard.

    n_easy = int(len(losses) * (1 - alpha))
    easy_samples = indexed_losses[:n_easy]
    hard_samples = indexed_losses[n_easy:]

    easy_indices = []
    for sample in easy_samples:
        easy_indices.append(sample[0])
    hard_indices = []
    for sample in hard_samples:
        hard_indices.append(sample[0])
    
    easy_dataset = tokenized_dataset.select(easy_indices)
    hard_dataset = tokenized_dataset.select(hard_indices)

    return easy_dataset, hard_dataset

def create_mixture_dataset(easy_dataset, hard_dataset, beta, size):
    assert 0 <= beta <= 1

    n_hard = int(beta * size)
    n_easy = size - n_hard

    assert n_easy <= len(easy_dataset)
    assert n_hard <= len(hard_dataset)

    easy_indices = random.sample(range(len(easy_dataset)), n_easy) # choose randomly n_easy indices.
    hard_indices = random.sample(range(len(hard_dataset)), n_hard)

    selected_easy = easy_dataset.select(easy_indices)
    selected_hard = hard_dataset.select(hard_indices)

    mixture_dataset = concatenate_datasets([selected_easy, selected_hard]) # Concatenate easy+hard.
    mixture_dataset = mixture_dataset.shuffle(seed=42) # Mix the dataset.

    return mixture_dataset



# --EVALUATION--
def evaluation(model, dataloader, device):
    model.eval()
    sum_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)

            sum_loss += outputs.loss.item()

            logits = outputs.logits
            labels = batch["labels"]

            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            predictions = torch.argmax(shift_logits, dim=-1)
            mask = shift_labels != -100

            correct += (predictions[mask] == shift_labels[mask]).sum().item()
            total += mask.sum().item()

    loss = sum_loss / len(dataloader)
    ppl = math.exp(loss)
    acc = correct / total if total > 0 else 0

    return loss, ppl, acc

# --TAIL METRIC--

def compute_tail_loss(sample_losses, tail_ratio=0.1):
    sorted_losses = sorted(sample_losses)
    k = int(len(sorted_losses) * tail_ratio)

    if k == 0:
        k = 1

    tail_losses = sorted_losses[-k:]
    tail_loss = sum(tail_losses) / len(tail_losses)

    return tail_loss

# --TRAINING--

def run_training(config):
    os.makedirs("outputs", exist_ok=True) # Create a folder named "outputs" if it does not exist, to save the metrics curve plot.

    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(config["models"]["name"])
    model.to(device)
    print("Using device:", device, flush=True)

    if device.type == "cuda":
        gpu_id = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU id: {gpu_id}")
        print(f"GPU name: {torch.cuda.get_device_name(gpu_id)}")

    
    tokenizer = load_tokenizer(config)
    dataset = load_difficulty_shift_dataset(config)
    max_length = _resolve_max_length(config)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    losses = compute_sample_losses(model, tokenized_dataset, data_collator, device, config)
    easy_dataset, hard_dataset = split_easy_hard(tokenized_dataset, losses, config["diagnostic"]["alpha"])
    mixture_dataset = create_mixture_dataset(easy_dataset, hard_dataset, config["diagnostic"]["beta_train"], config["diagnostic"]["train_size"])

    split_dataset = mixture_dataset.train_test_split(test_size=config["training"]["val_split_ratio"], seed=config["training"]["seed"])
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config["training"]["batch_size"], collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=config["training"]["batch_size"], collate_fn=data_collator)

    print(f"Dataset size: train={len(train_dataset)}, val={len(val_dataset)}")

    initial_val_loss, initial_val_ppl, initial_val_acc = evaluation(model, val_dataloader, device)

    print(
        f"Initial Val Loss: {initial_val_loss:.4f} | "
        f"Initial PPL: {initial_val_ppl:.2f} | "
        f"Initial Acc: {initial_val_acc:.4f}",
        flush=True
)

    optimizer = optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))

    train_losses = []
    val_losses = []
    val_perplexities = []
    val_accuracies = []
    cumulative_steps = []
    cumulative_tokens = []
    total_steps = 0
    total_tokens = 0

    test_betas = config["diagnostic"]["betas_test"]
    test_losses_by_epoch = {}
    for beta in test_betas:
        test_losses_by_epoch[beta] = []

    start_time = time.time()
    for epoch in range(config["training"]["epochs"]):
       # training loop:
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad() # By default, gradients are accumulated in PyTorch, else gradients would be mixed up together accross batches. It does not suppress the learning.
            outputs = model(**batch) # compute the loss
            loss = outputs.loss
            total_loss += loss.item()
            total_tokens += batch["input_ids"].numel()
            loss.backward() # Backpropagate the loss to compute the gradients of the model parameters with respect to the loss.
            optimizer.step() # Update the model parameters based on the computed gradients.

        total_steps += len(train_dataloader)
        avg_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_loss)

        # validation loop:
        val_loss, perplexity, accuracy = evaluation(model, val_dataloader, device)
        val_losses.append(val_loss)
        val_perplexities.append(perplexity)
        val_accuracies.append(accuracy)
        cumulative_steps.append(total_steps)
        cumulative_tokens.append(total_tokens)

        print(
        f"Epoch {epoch+1}/{config['training']['epochs']} | "
        f"Train Loss: {avg_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val PPL: {perplexity:.2f} | "
        f"Val Acc: {accuracy:.4f} | ",
        flush=True
)

        for beta_test in test_betas:
            test_dataset = create_mixture_dataset(easy_dataset, hard_dataset, beta_test, config["diagnostic"]["test_size"])
            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config["training"]["batch_size"], collate_fn=data_collator)
            test_loss, test_ppl, test_acc = evaluation(model, test_dataloader, device)
            test_losses_by_epoch[beta_test].append(test_loss)

    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training time: {training_time:.2f} seconds", flush=True)
    print(f"Training time: {training_time/60:.2f} minutes", flush=True)
    if device.type == "cuda":
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Max GPU memory allocated: {max_memory:.2f} GB", flush=True)

    val_losses_initial = [initial_val_loss] * len(train_losses)
    val_perplexities_initial = [initial_val_ppl] * len(train_losses)

    # Metrics after fine-tuning:

    final_test_losses = []
    final_tail_losses = []
    hist_losses_by_beta = {}

    for beta_test in test_betas:
        test_dataset = create_mixture_dataset(easy_dataset, hard_dataset, beta_test, config["diagnostic"]["test_size"])
        sample_losses = compute_sample_losses(model, test_dataset, data_collator, device, config)

        mean_loss = sum(sample_losses) / len(sample_losses)
        tail_loss = compute_tail_loss(sample_losses, tail_ratio=0.1)

        final_test_losses.append(mean_loss)
        final_tail_losses.append(tail_loss)
        hist_losses_by_beta[beta_test] = sample_losses

    

    epochs = range(1, len(train_losses) + 1)

    plotting.plot_training_loss_curves(
        {config["models"]["name"]: {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "cumulative_steps": cumulative_steps,
            "cumulative_tokens": cumulative_tokens,
        }},
        output_dir="outputs",
        filename_prefix="training_curves",
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.plot(epochs, val_losses_initial, label='Initial Validation Loss', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_perplexities, label='Validation Perplexity after Training')
    plt.plot(epochs, val_perplexities_initial, label='Initial Validation Perplexity', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    
    plt.savefig("outputs/fine_tuning_mixture.png")
    plt.close()

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(test_betas, final_test_losses, marker="o")
    plt.xlabel("Hard ratio in test set")
    plt.ylabel("Mean test loss")
    plt.title("Mean Test Loss vs Test Hard Ratio")

    plt.subplot(2, 2, 2)
    plt.plot(test_betas, final_tail_losses, marker="o")
    plt.xlabel("Hard ratio in test set")
    plt.ylabel("Tail loss (top 10%)")
    plt.title("Tail Loss vs Test Hard Ratio")

    plt.subplot(2, 2, 3)
    for beta_test in test_betas:
        plt.hist(
            hist_losses_by_beta[beta_test],
            bins=30,
            alpha=0.35, # transparency.
            label=f"beta={beta_test}"
        )

    plt.xlabel("Sample loss")
    plt.ylabel("Frequency")
    plt.title("Loss Distribution for Different Test Shifts")
    plt.legend()

    plt.subplot(2, 2, 4)
    for beta_test in test_betas:
        plt.plot(
            epochs,
            test_losses_by_epoch[beta_test],
            marker="o",
            label=f"beta={beta_test}"
        )

    plt.xlabel("Epochs")
    plt.ylabel("Test loss")
    plt.title("Test Loss over Epochs for Different Shifts")
    plt.legend()

    plt.tight_layout()
    plt.savefig("outputs/diagnostic_curves.png")
    plt.close()

    return train_losses, val_losses, val_perplexities, val_accuracies



if __name__ == "__main__":
    config = load_config()
    device = get_device()

    train_losses, val_losses, val_perplexities, val_accuracies = run_training(config)
    
    
    min_val_ppl = min(val_perplexities)
    best_epoch = val_perplexities.index(min_val_ppl) + 1
    print(f"Best Epoch: {best_epoch} with Validation Perplexity: {min_val_ppl:.2f}")
