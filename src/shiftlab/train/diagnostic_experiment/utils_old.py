import argparse
import yaml
import torch
import math
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import matplotlib.pyplot as plt
import os

# --GENERAL SETTINGS--
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True) # in the terminal
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    return config

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}

# --MODEL, TOKENIZER AND DATALOADER SETTINGS--

def setup_model_and_tokenizer(config, device):
    model = AutoModelForCausalLM.from_pretrained(config["models"]["name"])
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config["models"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return model, tokenizer, data_collator


# --DATASETS--

def tokenize_dataset(dataset, tokenizer, config):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=config["training"]["context_length"])
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset

def split_train_test_dataset(tokenized_dataset, config):
    train_pool_size = config["diagnostic"]["train_pool_size"]
    test_pool_size = config["diagnostic"]["test_pool_size"]

    assert train_pool_size + test_pool_size <= len(tokenized_dataset)

    train_pool = tokenized_dataset.select(range(train_pool_size))
    test_pool = tokenized_dataset.select(range(train_pool_size, train_pool_size + test_pool_size))

    return train_pool, test_pool

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

def split_easy_medium_hard(tokenized_dataset, losses, alpha_easy, alpha_hard):
    assert len(tokenized_dataset) == len(losses)
    assert 0 <= alpha_easy <= 1
    assert 0 <= alpha_hard <= 1
    assert 0 <= alpha_easy + alpha_hard < 1

    # Sort dataset:
    indexed_losses = []
    for i in range(len(losses)):
        indexed_losses.append((i, losses[i]))
    indexed_losses.sort(key=lambda x : x[1]) # sort according to the loss from easy to hard.

    n_easy = int(len(losses) * alpha_easy)
    n_medium = int(len(losses) * (1 - alpha_hard))
    easy_samples = indexed_losses[:n_easy]
    medium_samples = indexed_losses[n_easy:n_medium]
    hard_samples = indexed_losses[n_medium:]

    easy_indices = []
    for sample in easy_samples:
        easy_indices.append(sample[0])
    medium_indices = []
    for sample in medium_samples:
        medium_indices.append(sample[0])
    hard_indices = []
    for sample in hard_samples:
        hard_indices.append(sample[0])
    
    easy_dataset = tokenized_dataset.select(easy_indices)
    medium_dataset = tokenized_dataset.select(medium_indices)
    hard_dataset = tokenized_dataset.select(hard_indices)

    return easy_dataset, medium_dataset, hard_dataset

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

def create_mixture2_dataset(easy_dataset, medium_dataset, hard_dataset, alpha, beta, size):
    assert 0 <= alpha <= 1
    assert 0 <= beta <= 1
    assert 0 <= alpha + beta < 1

    n_easy = int(alpha * size)
    n_medium = int(beta * size)
    n_hard = size - n_easy - n_medium

    assert n_easy <= len(easy_dataset)
    assert n_medium <= len(medium_dataset)
    assert n_hard <= len(hard_dataset)

    easy_indices = random.sample(range(len(easy_dataset)), n_easy) # choose randomly n_easy indices.
    medium_indices = random.sample(range(len(medium_dataset)), n_medium)
    hard_indices = random.sample(range(len(hard_dataset)), n_hard)

    selected_easy = easy_dataset.select(easy_indices)
    selected_medium = medium_dataset.select(medium_indices)
    selected_hard = hard_dataset.select(hard_indices)

    mixture_dataset = concatenate_datasets([selected_easy, selected_medium, selected_hard]) # Concatenate easy+medium+hard.
    mixture_dataset = mixture_dataset.shuffle(seed=42) # Mix the dataset.

    return mixture_dataset

# --DATALOADERS--

def create_mixture_train_val_loaders(mixture_dataset, data_collator, config):
    split_dataset = mixture_dataset.train_test_split(test_size=config["training"]["val_split_ratio"], seed=config["training"]["seed"])

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config["training"]["batch_size"], collate_fn=data_collator)

    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=config["training"]["batch_size"], collate_fn=data_collator)

    return train_dataset, val_dataset, train_dataloader, val_dataloader

# --METRICS--

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

def compute_tail_loss(sample_losses, tail_ratio=0.1):
    sorted_losses = sorted(sample_losses)
    k = int(len(sorted_losses) * tail_ratio)

    if k == 0:
        k = 1

    tail_losses = sorted_losses[-k:]
    tail_loss = sum(tail_losses) / len(tail_losses)

    return tail_loss

# --TRAINING--

def train_one_epoch(model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)

    return avg_loss

def KL_DRO_one_epoch(model, train_dataloader, optimizer, gamma, lambd, rho, device):
    model.train()
    total_mix_loss = 0
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")

    for batch in train_dataloader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        outputs = model(**batch)

        avg_loss = outputs.loss

        # Compute Kl-DRO Loss:
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

        sample_loss = token_losses.sum(dim=1) / mask.sum(dim=1)

        x = torch.exp(sample_loss / lambd).sum() / len(sample_loss)
        dro_loss = lambd * torch.log(x) + lambd * rho

        # Compute Mix_loss:
        mix_loss = (1 - gamma) * avg_loss + gamma * dro_loss

        total_mix_loss += mix_loss.item()

        mix_loss.backward()
        optimizer.step()
    
    avg_mix_loss = total_mix_loss / len(train_dataloader)
    return avg_mix_loss


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

# --TRAINING--

def train_mixture(model, train_dataloader, val_dataloader, easy_dataset, hard_dataset, data_collator, device, config):
    
    optimizer = optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"].get("weight_decay", 0.0)))

    train_losses = []
    val_losses = []
    val_perplexities = []
    val_accuracies = []

    test_betas = config["diagnostic"]["betas_test"]
    test_losses_by_epoch = {beta: [] for beta in test_betas}

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device)

        val_loss, val_ppl, val_acc = evaluation(model, val_dataloader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_perplexities.append(val_ppl)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch+1}/{config['training']['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f} | "
            f"Val Acc: {val_acc:.4f}",
            flush=True
        )

        for beta_test in test_betas:
            test_dataset = create_mixture_dataset(easy_dataset, hard_dataset, beta_test, config["diagnostic"]["test_size"])

            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config["training"]["batch_size"], collate_fn=data_collator)

            test_loss, _, _ = evaluation(model, test_dataloader, device)
            test_losses_by_epoch[beta_test].append(test_loss)

    return train_losses, val_losses, val_perplexities, val_accuracies, test_losses_by_epoch

# --EVALUATION--

def evaluate_mixture(model, easy_dataset, hard_dataset, data_collator, device, config):
    test_betas = config["diagnostic"]["betas_test"]

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

    return final_test_losses, final_tail_losses, hist_losses_by_beta


# --TRAINING--

def train_mixture_rho(train_dataloader, val_dataloader,
                    easy_test, medium_test, hard_test,
                   data_collator, device, config):

    results = {}

    methods = {
        "ERM": None,
        "rho_small": config["training"]["rhos"]["small"],
        "rho_medium": config["training"]["rhos"]["medium"],
        "rho_large": config["training"]["rhos"]["large"],
    }

    for method_name, rho in methods.items():

        print(f"\n===== Training {method_name} =====", flush=True)

        model, _, _ = setup_model_and_tokenizer(config, device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"].get("weight_decay", 0.0))
        )

        train_losses = []
        val_losses = []
        val_perplexities = []
        val_accuracies = []

        for epoch in range(config["training"]["epochs"]):

            if method_name == "ERM":
                train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
            else:
                train_loss = KL_DRO_one_epoch(
                    model,
                    train_dataloader,
                    optimizer,
                    gamma=config["training"]["gamma"],
                    lambd=config["training"]["lambda"],
                    rho=rho,
                    device=device
                )

            val_loss, val_ppl, val_acc = evaluation(model, val_dataloader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_perplexities.append(val_ppl)
            val_accuracies.append(val_acc)

            print(
                f"[{method_name}] Epoch {epoch+1}/{config['training']['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val PPL: {val_ppl:.2f} | "
                f"Val Acc: {val_acc:.4f}",
                flush=True
            )

        eval_results = evaluate_scenarios(
            model,
            easy_test,
            medium_test,
            hard_test,
            data_collator,
            device,
            config
        )

        results[method_name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_perplexities": val_perplexities,
            "val_accuracies": val_accuracies,
            "eval": eval_results
        }

    return results

def train_mixture_lambd(train_dataloader, val_dataloader,
                   easy_test, medium_test, hard_test,
                   data_collator, device, config):

    results = {}

    methods = {
    "ERM": None,
    "lambda_small": config["training"]["lambdas"]["small"],
    "lambda_medium": config["training"]["lambdas"]["medium"],
    "lambda_large": config["training"]["lambdas"]["large"],
    }

    for method_name, lambd in methods.items():

        print(f"\n===== Training {method_name} =====", flush=True)

        model, _, _ = setup_model_and_tokenizer(config, device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"].get("weight_decay", 0.0))
        )

        train_losses = []
        val_losses = []
        val_perplexities = []
        val_accuracies = []

        for epoch in range(config["training"]["epochs"]):

            if method_name == "ERM":
                train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
            else:
                train_loss = KL_DRO_one_epoch(
                    model,
                    train_dataloader,
                    optimizer,
                    gamma=config["training"]["gamma"],
                    lambd=lambd,
                    rho=config["training"]["rho"],
                    device=device
                )

            val_loss, val_ppl, val_acc = evaluation(model, val_dataloader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_perplexities.append(val_ppl)
            val_accuracies.append(val_acc)

            print(
                f"[{method_name}] Epoch {epoch+1}/{config['training']['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val PPL: {val_ppl:.2f} | "
                f"Val Acc: {val_acc:.4f}",
                flush=True
            )

        eval_results = evaluate_scenarios(
            model,
            easy_test,
            medium_test,
            hard_test,
            data_collator,
            device,
            config
        )

        results[method_name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_perplexities": val_perplexities,
            "val_accuracies": val_accuracies,
            "eval": eval_results
        }

    return results

def train_mixture_lambd2(train_dataloader, val_dataloader,
                   close_dataset, mid_dataset, far_dataset,
                   data_collator, device, config):

    results = {}

    methods = {
    "ERM": None,
    "lambda_small": config["training"]["lambdas"]["small"],
    "lambda_medium": config["training"]["lambdas"]["medium"],
    "lambda_large": config["training"]["lambdas"]["large"],
    }

    for method_name, lambd in methods.items():

        print(f"\n===== Training {method_name} =====", flush=True)

        model, _, _ = setup_model_and_tokenizer(config, device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"].get("weight_decay", 0.0))
        )

        train_losses = []
        val_losses = []
        val_perplexities = []
        val_accuracies = []

        for epoch in range(config["training"]["epochs"]):

            if method_name == "ERM":
                train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
            else:
                train_loss = KL_DRO_one_epoch(
                    model,
                    train_dataloader,
                    optimizer,
                    gamma=config["training"]["gamma"],
                    lambd=lambd,
                    rho=config["training"]["rho"],
                    device=device
                )

            val_loss, val_ppl, val_acc = evaluation(model, val_dataloader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_perplexities.append(val_ppl)
            val_accuracies.append(val_acc)

            print(
                f"[{method_name}] Epoch {epoch+1}/{config['training']['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val PPL: {val_ppl:.2f} | "
                f"Val Acc: {val_acc:.4f}",
                flush=True
            )

        eval_results = evaluate_scenarios(
            model,
            close_dataset,
            mid_dataset,
            far_dataset,
            data_collator,
            device,
            config
        )

        results[method_name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_perplexities": val_perplexities,
            "val_accuracies": val_accuracies,
            "eval": eval_results
        }

    return results

# --EVALUATION--

def evaluate_scenarios(model, easy_dataset, medium_dataset, hard_dataset,
                        data_collator, device, config):

    alphas = config["diagnostic"]["alpha_k"]
    betas = config["diagnostic"]["beta_k"]
    scenario_names = config["diagnostic"]["scenario_names"]

    eval_results = {}

    for i, scenario_name in enumerate(scenario_names):

        alpha = alphas[i]
        beta = betas[i]

        test_dataset = create_mixture2_dataset(
            easy_dataset,
            medium_dataset,
            hard_dataset,
            alpha,
            beta,
            config["diagnostic"]["test_size"]
        )

        sample_losses = compute_sample_losses(
            model,
            test_dataset,
            data_collator,
            device,
            config
        )

        mean_loss = sum(sample_losses) / len(sample_losses)
        tail_loss = compute_tail_loss(sample_losses, tail_ratio=0.1)

        eval_results[scenario_name] = {
            "mean_loss": mean_loss,
            "tail_loss": tail_loss,
            "sample_losses": sample_losses
        }

    return eval_results


# --GRAPHS--

def plot_training_curves(val_losses_initial, train_losses, val_losses, val_perplexities_initial, val_perplexities, val_accuracies, config):
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    AXIS_LIMITS = {
    "loss": (0.0, 6.0),
    "perplexity": (0, 300),
    "accuracy": (0, 1),
    }

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.plot(epochs, val_losses_initial, label='Initial Validation Loss', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(*AXIS_LIMITS["loss"])
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_perplexities, label='Validation Perplexity after Training')
    plt.plot(epochs, val_perplexities_initial, label='Initial Validation Perplexity', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.ylim(*AXIS_LIMITS["perplexity"])
    plt.title('Validation Perplexity')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(*AXIS_LIMITS["accuracy"])
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/performance.png")
    plt.close()


def plot_diagnostic_curves(severities, histogram_severities, final_test_losses, final_tail_losses, hist_losses_by_severity, test_losses_by_epoch, config):
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    severity_name = config["diagnostic"]["severity_name"]
    severity_symbol = config["diagnostic"]["severity_symbol"]
    epochs = range(1, len(test_losses_by_epoch[severities[0]]) + 1)

    AXIS_LIMITS = {
    "loss": (0.0, 6.0),
    "perplexity": (0, 300),
    "accuracy": (0, 1),
    }

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(severities, final_test_losses, marker="o")
    plt.xlabel(severity_name)
    plt.ylabel("Mean test loss")
    plt.ylim(*AXIS_LIMITS["loss"])
    plt.title(f"Mean Test Loss vs {severity_name}")

    plt.subplot(2, 2, 2)
    plt.plot(severities, final_tail_losses, marker="o")
    plt.xlabel(severity_name)
    plt.ylabel("Tail loss (top 10%)")
    plt.title(f"Tail Loss vs {severity_name}")

    plt.subplot(2, 2, 3)
    for severity in histogram_severities:
        plt.hist(
            hist_losses_by_severity[severity],
            bins=30,
            alpha=0.35, # transparency.
            label=f"{severity_symbol}={severity}"
        )

    plt.xlabel("Sample loss")
    plt.ylabel("Frequency")
    plt.title("Loss Distribution for Different Test Shifts")
    plt.legend()

    plt.subplot(2, 2, 4)
    for severity in severities:
        plt.plot(
            epochs,
            test_losses_by_epoch[severity],
            marker="o",
            label=f"{severity_symbol}={severity}"
        )

    plt.xlabel("Epochs")
    plt.ylabel("Test loss")
    plt.ylim(*AXIS_LIMITS["loss"])
    plt.title("Test Loss over Epochs for Different Shifts")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/robustness_controlled_shift.png")
    plt.close()


def plot_multi_model_performance(results, config):
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    losses = results["ERM"]["train_losses"]
    epochs = range(1, len(losses) + 1)
    AXIS_LIMITS = {
    "loss": (0.0, 6.0),
    "perplexity": (0, 300),
    "accuracy": (0, 1),
    }

    plt.figure(figsize=(14, 10))

    plt.subplot(1, 3, 1)
    for model_name, model_results in results.items():
        plt.plot(epochs, model_results["train_losses"], label=f"{model_name} Training Loss")
        plt.plot(epochs, model_results["val_losses"], label=f"{model_name} Validation Loss", linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(*AXIS_LIMITS["loss"])
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    for model_name, model_results in results.items():
        plt.plot(epochs, model_results["val_perplexities"], label=f"{model_name} Validation Perplexity")
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.ylim(*AXIS_LIMITS["perplexity"])
    plt.title('Validation Perplexity')
    plt.legend()

    plt.subplot(1, 3, 3)
    for model_name, model_results in results.items():
        plt.plot(epochs, model_results["val_accuracies"], label=f"{model_name} Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(*AXIS_LIMITS["accuracy"])
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/performance.png")
    plt.close()


def plot_multi_model_diagnostic_rho(results, config):
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    scenarios = list(results["ERM"]["eval"].keys()) # get the name of the scenario (e.g., "mostly easy", "balanced", "mostly hard").
    number_of_scenarios = len(results["ERM"]["eval"].keys())

    AXIS_LIMITS = {
    "loss": (0.0, 6.0)}

    space = {"ERM":-1.5,
             "rho_small": -0.5,
             "rho_medium": 0.5,
             "rho_large": 1.5}

    plt.figure(figsize=(16, 12))

    plt.subplot(1, 2, 1)
    x = np.arange(number_of_scenarios)
    for model_name, model_results in results.items():
        mean_losses = [model_results["eval"][scenario]["mean_loss"] for scenario in scenarios]
        plt.bar(x + space[model_name] * 0.2, mean_losses, width=0.2, label=f"{model_name} Mean Loss")
    plt.xticks(x, scenarios)
    plt.xlabel("Test scenario")
    plt.ylabel("Mean test loss")
    plt.ylim(*AXIS_LIMITS["loss"])
    plt.title("Mean Test Loss by Scenario")
    plt.legend()

    plt.subplot(1, 2, 2)
    x = np.arange(number_of_scenarios)
    for model_name, model_results in results.items():
        tail_losses = [model_results["eval"][scenario]["tail_loss"] for scenario in scenarios]
        plt.bar(x + space[model_name] * 0.2, tail_losses, width=0.2, label=f"{model_name} Tail Loss")
    plt.xticks(x, scenarios)
    plt.xlabel("Test scenario")
    plt.ylabel("Tail test loss")
    plt.ylim(*AXIS_LIMITS["loss"])
    plt.title("Tail Test Loss by Scenario")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/robustness_scenarios.png")
    plt.close()

def plot_multi_model_diagnostic_lambd(results, config):
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    scenarios = list(results["ERM"]["eval"].keys()) # get the name of the scenario (e.g., "mostly easy", "balanced", "mostly hard").
    number_of_scenarios = len(results["ERM"]["eval"].keys())

    AXIS_LIMITS = {
    "loss": (0.0, 10.0)}

    space = {
    "ERM": -1.5,
    "lambda_small": -0.5,
    "lambda_medium": 0.5,
    "lambda_large": 1.5
    }

    plt.figure(figsize=(16, 12))

    plt.subplot(1, 2, 1)
    x = np.arange(number_of_scenarios)
    for model_name, model_results in results.items():
        mean_losses = [model_results["eval"][scenario]["mean_loss"] for scenario in scenarios]
        plt.bar(x + space[model_name] * 0.2, mean_losses, width=0.2, label=f"{model_name} Mean Loss")
    plt.xticks(x, scenarios)
    plt.xlabel("Test scenario")
    plt.ylabel("Mean test loss")
    plt.ylim(*AXIS_LIMITS["loss"])
    plt.title("Mean Test Loss by Scenario")
    plt.legend()

    plt.subplot(1, 2, 2)
    x = np.arange(number_of_scenarios)
    for model_name, model_results in results.items():
        tail_losses = [model_results["eval"][scenario]["tail_loss"] for scenario in scenarios]
        plt.bar(x + space[model_name] * 0.2, tail_losses, width=0.2, label=f"{model_name} Tail Loss")
    plt.xticks(x, scenarios)
    plt.xlabel("Test scenario")
    plt.ylabel("Tail test loss")
    plt.ylim(*AXIS_LIMITS["loss"])
    plt.title("Tail Test Loss by Scenario")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/robustness_scenarios.png")
    plt.close()


def plot_multi_model_histograms(results, config):
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    scenarios = list(results["ERM"]["eval"].keys()) # get the name of the scenario (e.g., "mostly easy", "balanced", "mostly hard").
    all_losses = []
    for model_results in results.values():
        for scenario in scenarios:
            all_losses.extend(model_results["eval"][scenario]["sample_losses"])
    xmin = min(all_losses)
    xmax = max(all_losses)

    plt.figure(figsize=(16, 12))
    for i, scenario in enumerate(scenarios):
        plt.subplot(1, len(scenarios), i+1)
        for model_name, model_results in results.items():
            sample_losses = model_results["eval"][scenario]["sample_losses"]
            plt.hist(sample_losses, bins=30, alpha=0.35, label=f"{model_name} Sample Losses")
        plt.xlabel("Sample loss")
        plt.xlim(xmin, xmax)
        plt.ylabel("Frequency")
        plt.title(f"Loss Distribution for {scenario} Scenario")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/robustness_histograms.png")
    plt.close()






