# In this file, we are going to explore the potential values of lambda for PG-19 dataset, this is in order to have a good range to use in our experiments.
import utils
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from shiftlab.data.load_datasets import load_dataset_from_config

# -- TRAINING --

def training_lambda(train_dataloader, val_dataloader, test_datasets,data_collator, device, config):
    results = {}

    for lambd in config["diagnostic"]["lambdas"]:
        print(f"\n===== Training KL-DRO-{lambd} =====", flush=True)

        model, _, _ = utils.setup_model_and_tokenizer(config, device)

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

            train_loss = utils.KL_DRO_one_epoch(
                    model,
                    train_dataloader,
                    optimizer,
                    gamma=config["training"]["gamma"],
                    lambd=lambd,
                    rho=config["training"]["rho"],
                    device=device
                )
            val_loss, val_ppl, val_acc = utils.evaluation(model, val_dataloader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_perplexities.append(val_ppl)
            val_accuracies.append(val_acc)

            print(
                f"[λ={lambd}] Epoch {epoch+1}/{config['training']['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val PPL: {val_ppl:.2f} | "
                f"Val Acc: {val_acc:.4f}",
                flush=True
            )
        results[lambd] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_perplexities": val_perplexities,
        "val_accuracies": val_accuracies,
        "test": {}
        }
        for noise_name, test_dataset in test_datasets.items():
            test_losses = utils.compute_sample_losses(model, test_dataset, data_collator, device, config)
            test_tail_loss = utils.compute_tail_loss(test_losses, tail_ratio=0.1)
            results[lambd]["test"][noise_name] = {
            "test_loss_mean": np.mean(test_losses),
            "test_tail_loss": test_tail_loss
            }
    return results

def make_noisy_test_datasets(test_raw, tokenizer, config):
    noise_levels = config["diagnostic"]["noise_levels"]

    noisy_test_datasets = {}

    for noise_name, p in noise_levels.items():
        print(f"Creating test dataset for noise={noise_name}, p={p}")

        if p == 0:
            noisy_raw = test_raw
        else:
            noisy_raw = test_raw.map(lambda example: utils.add_ocr_noise(example, p=p, seed=config["training"]["seed"]))

        noisy_test_datasets[noise_name] = utils.tokenize_dataset(noisy_raw, tokenizer, config)
    return noisy_test_datasets

# -- GRAPHS --

def calibration_graphs(results, config):
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    lambdas = config["diagnostic"]["lambdas"]
    noise_levels = config["diagnostic"]["noise_levels"].keys()

    plt.figure(figsize=(16, 8))
    
    
    # Mean loss
    plt.subplot(1, 2, 1)
    for noise_name in noise_levels:
        mean_losses = [results[lambd]["test"][noise_name]["test_loss_mean"] for lambd in lambdas]
        ERM_mean_losses = [results["ERM"]["test"][noise_name]["test_loss_mean"]] * len(lambdas)

        plt.plot(lambdas, mean_losses, marker="o", label=f"KL-DRO {noise_name}")
        plt.plot(lambdas, ERM_mean_losses, linestyle="--", label=f"ERM {noise_name}")
    plt.xscale("log")
    plt.xlabel("Lambdas")
    plt.ylabel(" Mean Loss")
    plt.title("Mean Loss VS Lambdas")
    plt.legend()

    # Tail loss
    plt.subplot(1, 2, 2)
    for noise_name in noise_levels:
        tail_losses = [results[lambd]["test"][noise_name]["test_tail_loss"] for lambd in lambdas]
        ERM_tail_losses = [results["ERM"]["test"][noise_name]["test_tail_loss"]] * len(lambdas)

        plt.plot(lambdas, tail_losses, marker="o", label=f"KL-DRO {noise_name}")
        plt.plot(lambdas, ERM_tail_losses, linestyle="--", label=f"ERM {noise_name}")
    plt.xscale("log")
    plt.xlabel("Lambdas")
    plt.ylabel("Tail Loss")
    plt.title("Tail Loss VS Lambdas")
    plt.legend()

    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/lambda_calibration_curves.png")
    plt.close()

# -- TABLE --

def lambda_shift_table(results, config, test_datasets, data_collator, device):
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    lambdas = config["diagnostic"]["lambdas"]
    noise_levels = config["diagnostic"]["noise_levels"]

    clean_dataset = test_datasets["clean"]

    rows = []

    for noise_name, p in noise_levels.items():
        shifted_dataset = test_datasets[noise_name]
        shift_severity = utils.compute_shift_severity(config, clean_dataset, shifted_dataset, data_collator, device)

        tail_losses = np.array([
            float(results[lambd]["test"][noise_name]["test_tail_loss"])
            for lambd in lambdas
        ], dtype=float)

        mean_losses = np.array([
            float(results[lambd]["test"][noise_name]["test_loss_mean"])
            for lambd in lambdas
        ], dtype=float)
        tol = 0.02
        min_tail = np.nanmin(tail_losses)
        min_mean = np.nanmin(mean_losses)

        best_lambdas_tail = [
            lambd for lambd, loss in zip(lambdas, tail_losses)
            if not np.isnan(loss) and loss <= min_tail + tol
        ]

        best_lambdas_mean = [
            lambd for lambd, loss in zip(lambdas, mean_losses)
            if not np.isnan(loss) and loss <= min_mean + tol
        ]
        best_tail_idx = int(np.nanargmin(tail_losses))
        best_mean_idx = int(np.nanargmin(mean_losses))
        best_lambda_tail = lambdas[best_tail_idx]
        best_lambda_mean = lambdas[best_mean_idx]
        print("DEBUG", noise_name)
        print("tail_losses:", tail_losses)
        print("min_tail:", min_tail)
        print("best_lambdas_tail:", best_lambdas_tail)
        print("mean_losses:", mean_losses)
        print("min_mean:", min_mean)
        print("best_lambdas_mean:", best_lambdas_mean)
        rows.append({
            "noise_name": noise_name,
            "p": p,
            "shift_severity": shift_severity,
            "λ*_tail": best_lambda_tail,
            "λ*_tail range": str(best_lambdas_tail),
            "λ*_mean": best_lambda_mean,
            "λ*_mean range": str(best_lambdas_mean)
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "lambda_shift_table.csv")
    df.to_csv(csv_path, index=False)
    print(df)
    print(f"Lambda shift table saved to {csv_path}")
    

def run_lambda_calibration(config):
    device = utils.get_device()
    _ , tokenizer, data_collator = utils.setup_model_and_tokenizer(config, device)

    if device.type == "cuda":
        gpu_id = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU id: {gpu_id}")
        print(f"GPU name: {torch.cuda.get_device_name(gpu_id)}")

    # Load dataset
    raw_dataset = load_dataset_from_config(config)
    train_raw, test_raw = utils.split_train_test_dataset(raw_dataset, config)
    
    train_pool = utils.tokenize_dataset(train_raw, tokenizer, config)
    test_datasets = make_noisy_test_datasets(test_raw, tokenizer, config)

    train_dataset, val_dataset, train_dataloader, val_dataloader = utils.create_mixture_train_val_loaders(train_pool, data_collator, config)

    # Training
    results = training_lambda(train_dataloader, val_dataloader, test_datasets, data_collator, device, config)
    model, _, _ = utils.setup_model_and_tokenizer(config, device)
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

        train_loss = utils.train_one_epoch(model, train_dataloader, optimizer, device)
        val_loss, val_ppl, val_acc = utils.evaluation(model, val_dataloader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_perplexities.append(val_ppl)
        val_accuracies.append(val_acc)

        print(
                f"[ERM] Epoch {epoch+1}/{config['training']['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val PPL: {val_ppl:.2f} | "
                f"Val Acc: {val_acc:.4f}",
                flush=True
            )
    results["ERM"] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_perplexities": val_perplexities,
        "val_accuracies": val_accuracies,
        "test": {}
    }
    for noise_name, test_dataset in test_datasets.items():
        test_losses = utils.compute_sample_losses(model, test_dataset, data_collator, device, config)
        test_tail_loss = utils.compute_tail_loss(test_losses, tail_ratio=0.1)
        results["ERM"]["test"][noise_name] = {
            "test_loss_mean": np.mean(test_losses),
            "test_tail_loss": test_tail_loss
        }
    
    calibration_graphs(results, config)
    lambda_shift_table(results, config, test_datasets, data_collator, device)
    print("Done.", flush=True)

if __name__ == "__main__":
    config = utils.load_config()
    run_lambda_calibration(config)


