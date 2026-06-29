import argparse
import yaml
import torch
import math
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from datasets import concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import matplotlib.pyplot as plt
import os


# ------- GENERAL SETTINGS -------

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


# ------- MODEL, TOKENIZER AND DATALOADER SETTINGS -------

def setup_model_and_tokenizer(config, device):
    model = AutoModelForCausalLM.from_pretrained(config["models"]["name"])
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config["models"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return model, tokenizer, data_collator


# ------- DATASETS UTILITIES -------

def split_train_test_dataset(dataset, config):
    train_pool_size = config["diagnostic"]["train_pool_size"]
    test_pool_size = config["diagnostic"]["test_pool_size"]

    assert train_pool_size + test_pool_size <= len(dataset)

    train_pool = dataset.select(range(train_pool_size))
    test_pool = dataset.select(range(train_pool_size, train_pool_size + test_pool_size))

    return train_pool, test_pool

def add_ocr_noise(example, p, seed=None):

    if seed is not None:
        random.seed(seed)

    replacements = {
        "o": "0",
        "i": "l",
        "e": "c",
        "a": "@",
        "s": "5",
        "œ": "oe",
        "l": "1",
        "O": "0"
    }
    text = example["text"]
    noisy_text = ""

    for char in text:
        if char in replacements and random.random() < p:
            noisy_text += replacements[char]
        else:
            noisy_text += char

    example["text"] = noisy_text
    return example

def create_shift_datasets(dataset, config, seed):
    total_size = len(dataset)
    dataset = dataset.select(range(total_size))

    quarter = total_size // 4

    clean_reference_dataset = dataset.select(range(0, quarter))
    close_clean_dataset = dataset.select(range(quarter, 2 * quarter))
    mid_clean_dataset = dataset.select(range(2 * quarter, 3 * quarter))
    far_clean_dataset = dataset.select(range(3 * quarter, total_size))

    close_dataset = close_clean_dataset.map(lambda x: add_ocr_noise(x, p=config["diagnostic"]["close_noise"], seed=seed))
    mid_dataset = mid_clean_dataset.map(lambda x: add_ocr_noise(x, p=config["diagnostic"]["mid_noise"], seed=seed))
    far_dataset = far_clean_dataset.map(lambda x: add_ocr_noise(x, p=config["diagnostic"]["far_noise"], seed=seed))

    return clean_reference_dataset, close_dataset, mid_dataset, far_dataset

def tokenize_dataset(dataset, tokenizer, config):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=config["training"]["context_length"])
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset

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

def create_three_way_mixture_dataset(easy_dataset, medium_dataset, hard_dataset, alpha, beta, size, seed):
    assert 0 <= alpha <= 1
    assert 0 <= beta <= 1
    assert 0 <= alpha + beta < 1

    n_easy = int(alpha * size)
    n_medium = int(beta * size)
    n_hard = size - n_easy - n_medium

    assert n_easy <= len(easy_dataset)
    assert n_medium <= len(medium_dataset)
    assert n_hard <= len(hard_dataset)

    random.seed(seed)
    easy_indices = random.sample(range(len(easy_dataset)), n_easy) # choose randomly n_easy indices.
    medium_indices = random.sample(range(len(medium_dataset)), n_medium)
    hard_indices = random.sample(range(len(hard_dataset)), n_hard)

    selected_easy = easy_dataset.select(easy_indices)
    selected_medium = medium_dataset.select(medium_indices)
    selected_hard = hard_dataset.select(hard_indices)

    mixture_dataset = concatenate_datasets([selected_easy, selected_medium, selected_hard]) # Concatenate easy+medium+hard.
    mixture_dataset = mixture_dataset.shuffle(seed=seed) # Mix the dataset.

    return mixture_dataset

def create_mixture_train_val_loaders(mixture_dataset, data_collator, config):
    split_dataset = mixture_dataset.train_test_split(test_size=config["training"]["val_split_ratio"], seed=config["training"]["seed"])

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config["training"]["batch_size"], collate_fn=data_collator)

    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=config["training"]["batch_size"], collate_fn=data_collator)

    return train_dataset, val_dataset, train_dataloader, val_dataloader

def create_probe_dataloader(val_dataset, data_collator, batch_size, probe_seed, probe_size=None):
    if probe_size is None:
        probe_size = len(val_dataset)
    probe_size = min(probe_size, len(val_dataset))

    generator = torch.Generator().manual_seed(probe_seed)
    indices = torch.randperm(len(val_dataset), generator=generator)[:probe_size]

    probe_dataset = Subset(val_dataset, indices.tolist())
    probe_dataloader = DataLoader(probe_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    return probe_dataloader

# ------- METRICS -------

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

def compute_shift_severity(config, clean_dataset, shifted_dataset, data_collator, device):
    model, _, _ = setup_model_and_tokenizer(config, device)

    clean_losses = compute_sample_losses(model, clean_dataset, data_collator, device, config)
    shifted_losses = compute_sample_losses(model, shifted_dataset, data_collator, device, config)

    clean_ppl = math.exp(np.mean(clean_losses))
    shifted_ppl = math.exp(np.mean(shifted_losses))

    return shifted_ppl - clean_ppl

# ------- TRAINING -------

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

        log_mean_exp = torch.logsumexp(sample_loss / lambd, dim=0) - torch.log(
        torch.tensor(len(sample_loss), device=device, dtype=sample_loss.dtype)
        )

        dro_loss = lambd * log_mean_exp + lambd * rho

        # Compute Mix_loss:
        mix_loss = (1 - gamma) * avg_loss + gamma * dro_loss

        total_mix_loss += mix_loss.item()

        mix_loss.backward()
        optimizer.step()
    
    avg_mix_loss = total_mix_loss / len(train_dataloader)
    return avg_mix_loss

def train_method(method_name, train_dataloader, val_dataloader, clean_reference_dataset, close_dataset,
                mid_dataset, far_dataset, data_collator, device, config):

    print(f"\n===== Training {method_name} =====", flush=True)

    model, _, _ = setup_model_and_tokenizer(config, device)
    method_config = config["methods"][method_name]

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

        if method_config["type"] == "ERM":
            train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        else:
            lambd = method_config["lambda"]
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
    # Because of the randomness of the dataset creation and batch sampling, we evaluate on multiple seeds.
    all_eval_results = []
    for seed in config["diagnostic"]["evaluation_seeds"]:

        eval_results = evaluate_scenarios(
            model,
            close_dataset,
            mid_dataset,
            far_dataset,
            clean_reference_dataset,
            data_collator,
            device,
            config,
            seed
        )
        all_eval_results.append(eval_results)
    
    scenario_names = config["diagnostic"]["scenario_names"]
    aggregated_results = {}
    for scenario_name in scenario_names:
        mean_losses = [eval_result[scenario_name]["mean_loss"] for eval_result in all_eval_results]
        tail_losses = [eval_result[scenario_name]["tail_loss"] for eval_result in all_eval_results]
        degradations = [eval_result[scenario_name]["degradation"] for eval_result in all_eval_results]
        sample_losses_all_seeds = [eval_result[scenario_name]["sample_losses"] for eval_result in all_eval_results]

        aggregated_results[scenario_name] = {
            "mean_losses": mean_losses,
            "tail_losses": tail_losses,
            "degradations": degradations,
            "sample_losses_all_seeds": sample_losses_all_seeds,
            # Mean:
            "mean_loss_mean": np.mean(mean_losses),
            "tail_loss_mean": np.mean(tail_losses),
            "degradation_mean": np.mean(degradations),
            # Ecart-type:
            "mean_loss_std": np.std(mean_losses),
            "tail_loss_std": np.std(tail_losses),
            "degradation_std": np.std(degradations)
        }


    results = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_perplexities": val_perplexities,
            "val_accuracies": val_accuracies,
            "eval": aggregated_results
        }

    return results



# ------- EVALUATION SCENARIOS -------

def evaluate_scenarios(model, close_dataset, mid_dataset, far_dataset, clean_reference_dataset,
                        data_collator, device, config, seed):

    alphas = config["diagnostic"]["alpha_k"]
    betas = config["diagnostic"]["beta_k"]
    scenario_names = config["diagnostic"]["scenario_names"]

    eval_results = {}

    ID_losses = compute_sample_losses(
            model,
            clean_reference_dataset,
            data_collator,
            device,
            config
        )

    for i, scenario_name in enumerate(scenario_names):

        alpha = alphas[i]
        beta = betas[i]

        test_dataset = create_three_way_mixture_dataset(
            close_dataset,
            mid_dataset,
            far_dataset,
            alpha,
            beta,
            config["diagnostic"]["test_size"],
            seed
        )

        OOD_losses = compute_sample_losses(
            model,
            test_dataset,
            data_collator,
            device,
            config
        )

        mean_loss = sum(OOD_losses) / len(OOD_losses)
        tail_loss = compute_tail_loss(OOD_losses, tail_ratio=0.1)
        delta = np.mean(OOD_losses) - np.mean(ID_losses)

        eval_results[scenario_name] = {
            "mean_loss": mean_loss,
            "tail_loss": tail_loss,
            "sample_losses": OOD_losses,
            "degradation": delta
        }

    return eval_results


# ------- GRAPHS -------

def plot_training_curves(results, config):
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    losses = results["ERM"]["train_losses"]
    epochs = range(1, len(losses) + 1)


    plt.figure(figsize=(16, 10))

    plt.subplot(1, 3, 1)
    for model_name, model_results in results.items():
        plt.plot(epochs, model_results["train_losses"], label=f"{model_name} Training Loss")
        plt.plot(epochs, model_results["val_losses"], label=f"{model_name} Validation Loss", linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    for model_name, model_results in results.items():
        plt.plot(epochs, model_results["val_perplexities"], label=f"{model_name} Validation Perplexity")
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity')
    plt.legend()

    plt.subplot(1, 3, 3)
    for model_name, model_results in results.items():
        plt.plot(epochs, model_results["val_accuracies"], label=f"{model_name} Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/performance_curves.png")
    plt.close()

def plot_shift_visual(results, config):
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    scenarios = list(results["ERM"]["eval"].keys())

    plt.figure(figsize=(18, 12))
    for i, scenario in enumerate(scenarios):
        plt.subplot(1, 4, i+1)
        all_sample_losses = []
        for losses in results["ERM"]["eval"][scenario]["sample_losses_all_seeds"]:
            all_sample_losses.extend(losses)
        plt.hist(all_sample_losses, bins=30, alpha=0.35, label="ERM Sample Losses")
        plt.xlabel("Sample loss")
        plt.ylabel("Frequency")
        plt.title(f"Loss Distribution for {scenario} Scenario")
        plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(np.arange(len(scenarios)), [results["ERM"]["eval"][scenario]["degradation_mean"] for scenario in scenarios], marker='o', label="ERM Degradation Δ")
    plt.xticks(np.arange(len(scenarios)), scenarios)
    plt.xlabel("Scenarios")
    plt.ylabel("Degradation Δ")
    plt.title("Degradation Δ by Scenario")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/shift_visualization.png")
    plt.close()

def plot_robustness_boxplots(results, config):
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    scenarios = list(results["ERM"]["eval"].keys())
    method_names = list(results.keys())
    metrics = {
        "Mean loss": ("mean_loss_mean", "mean_loss_std"),
        "Tail loss": ("tail_loss_mean", "tail_loss_std"),
        "Degradation Δ": ("degradation_mean", "degradation_std"),
    }

    plt.figure(figsize=(18, 8))
    bar_width = 0.18
    x = np.arange(len(scenarios))

    for plot_idx, (metric_title, (mean_key, std_key)) in enumerate(metrics.items()):
        plt.subplot(1, 3, plot_idx + 1)

        for method_idx, method in enumerate(method_names):
            means = [results[method]["eval"][scenario][mean_key] for scenario in scenarios]
            stds = [results[method]["eval"][scenario][std_key] for scenario in scenarios]

            positions = x + method_idx * bar_width
            if method == "ERM":
                label = "ERM"
            else:
                lambd = config["methods"][method]["lambda"]
                label = f"λ={lambd}"
            
            plt.bar(positions, means, width=bar_width, label=label,yerr=stds, capsize=4, ecolor="black", error_kw={"elinewidth": 1.5, "capthick": 1.5})

        plt.xticks(x + bar_width * (len(method_names)-1) / 2, scenarios, rotation=15)
        plt.ylabel(metric_title)
        plt.title(f"{metric_title} across seeds")
        if plot_idx == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/robustness_boxplots.png")
    plt.close()
