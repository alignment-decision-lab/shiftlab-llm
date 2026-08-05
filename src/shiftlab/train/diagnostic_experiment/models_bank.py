import os
import copy
import torch
import utils
import pandas as pd
import numpy as np
import shift_measurement as sm
import lambda_window as lw
import torch.optim as optim
from shiftlab.data.load_datasets import load_dataset_from_subconfig
from torch.utils.data import DataLoader

datasets_config = {
     "FreeLaw": {       
        "dataset": {
        "type": "hf_text",
        "name": "timaeus/pile-freelaw",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
        "training": {
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.00002,
            "context_length": 512,
            "weight_decay": 0.0,
            "dataset_size": 10000,
            "seed": 42,
            "val_split_ratio": 0.1,
            "gamma": 0.7,
            "rho": 0.0
    }},

     "PubMed Central": {     
     "dataset": {
         "type": "hf_text",
         "name": "datajuicer/the-pile-pubmed-central-refined-by-data-juicer",
         "split": "train",
         "text_column": "text",
         "streaming": True,
     },
     "training": {
         "batch_size": 4,
         "gradient_accumulation_steps": 4,
         "learning_rate": 0.00002,
         "context_length": 512,
         "weight_decay": 0.0,
         "dataset_size": 10000,

         "seed": 42,
         "val_split_ratio": 0.1,
         "gamma": 0.7,
         "rho": 0.0
     }},

     "ArXiv": {     
     "dataset": {
         "type": "hf_text",
         "name": "timaeus/pile-arxiv",
         "split": "train",
         "text_column": "text",
         "streaming": True,
     },
     "training": {
         "batch_size": 4,
         "gradient_accumulation_steps": 4,
         "learning_rate": 0.00002,
         "context_length": 512,
         "weight_decay": 0.0,
         "dataset_size": 10000,

         "seed": 42,
         "val_split_ratio": 0.1,
         "gamma": 0.7,
         "rho": 0.0
     }},

}


def collect_model_bank_configs(config, datasets_config,):
    """Create one complete configuration per source and lambda."""

    lambdas = config["model_bank"]["lambdas"]
    model_cfg = config["models"]

    one_configs = []

    for dataset_name, dataset_config in datasets_config.items():
        for lambda_ in lambdas:
            cfg = copy.deepcopy(dataset_config)

            cfg["dataset_name"] = dataset_name
            cfg["models"] = copy.deepcopy(model_cfg)
            cfg["training"]["lambda"] = float(lambda_)

            one_configs.append(cfg)

    return one_configs

def prepare_one_source_dataset(config, dataset_name, dataset_config, device):
    """ Prepare the source distribution, initial sample losses, and lambda-rho curve for one source dataset. """

    print(
        f"\n===== Preparing source dataset "
        f"{dataset_name} =====",
        flush=True,
    )

    dataset_dir_name = dataset_name.replace(" ", "_")

    dataset_dir = os.path.join(config["model_bank"]["save_path"], dataset_dir_name)
    os.makedirs(dataset_dir, exist_ok=True)

    source_distribution_path = os.path.join(dataset_dir, "source_distribution.pt")
    source_losses_path = os.path.join(dataset_dir, "source_losses.pt")

    lambda_rho_curve_path = os.path.join(dataset_dir, "lambda_rho_curve.csv")

    source_config = copy.deepcopy(dataset_config)

    source_config["dataset_name"] = dataset_name
    source_config["models"] = copy.deepcopy(config["models"])
    seed = int(source_config["training"].get("seed", 42))
    utils.set_seed(seed)

    model, tokenizer, data_collator = utils.setup_model_and_tokenizer(source_config, device,)

    dataset = load_dataset_from_subconfig(source_config["dataset"], source_config["training"])

    tokenized_dataset = utils.tokenize_and_group_dataset(dataset, tokenizer, source_config)
    distribution_dataloader = DataLoader(tokenized_dataset, batch_size=source_config["training"]["batch_size"], shuffle=False, collate_fn=data_collator)

    distribution_max_tokens = int(config["model_bank"].get("distribution_max_tokens",500_000))
    distribution_epsilon = float(config["model_bank"].get("distribution_epsilon",1e-8))

    p_D, n_tokens_used = sm.compute_dataset_token_distribution_with_budget(
        dataloader=distribution_dataloader,
        vocab_size=tokenizer.vocab_size,
        max_tokens=distribution_max_tokens,
        epsilon=distribution_epsilon,
    )
    if n_tokens_used != distribution_max_tokens:
        raise RuntimeError(
            f"{dataset_name}: {n_tokens_used} tokens used "
            f"instead of {distribution_max_tokens}."
        )
    torch.save(p_D.cpu(), source_distribution_path)
    

    losses = utils.compute_sample_losses(
        model,
        tokenized_dataset,
        data_collator,
        device,
        source_config,
    )
    losses = torch.as_tensor(losses).detach().cpu()

    torch.save(losses, source_losses_path)

    lambda_grid = np.asarray(config["model_bank"]["lambda_grid"], dtype=float)

    lw.compute_lambda_rho_curve(
        losses={dataset_name: losses},
        lambda_grid=lambda_grid,
        output_csv_path=lambda_rho_curve_path,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "dataset_name": dataset_name,
        "dataset_dir": dataset_dir,
        "source_distribution_path": source_distribution_path,
        "source_losses_path": source_losses_path,
        "lambda_rho_curve_path": lambda_rho_curve_path,
    }

import matplotlib.pyplot as plt


def plot_training_curves(
    results,
    dataset_name,
    lambda_,
    output_dir,
):
    """Plot KL-DRO train and validation cross-entropy losses."""

    epochs = range(
        1,
        len(results["train_losses"]) + 1,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        results["train_losses"],
        marker="o",
        label="Train cross-entropy",
    )

    plt.plot(
        epochs,
        results["val_losses"],
        marker="o",
        label="Validation cross-entropy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(
        f"KL-DRO training curves for {dataset_name} "
        f"(lambda={lambda_:g})"
    )

    plt.xticks(list(epochs))
    plt.grid()
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "training_curves.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path

def train_one_bank_model(one_config, source_info, config, device):
    """Train one KL-DRO model with early stopping."""

    dataset_name = one_config["dataset_name"]
    dataset_cfg = one_config["dataset"]
    training_cfg = one_config["training"]

    lambda_ = float(training_cfg["lambda"])

    print(
        f"\n===== Training KL-DRO "
        f"lambda={lambda_} on {dataset_name} =====",
        flush=True,
    )

    seed = training_cfg.get("seed", 42)
    utils.set_seed(seed)

    model, tokenizer, data_collator = utils.setup_model_and_tokenizer(one_config, device,)
    model_name = (
        f"{one_config['models']['name']}"
        f"_KL_DRO_lambda_{lambda_:g}"
        f"_dataset_{dataset_name}"
    )

    dataset = load_dataset_from_subconfig(dataset_cfg, training_cfg)
    tokenized_dataset = utils.tokenize_and_group_dataset(dataset, tokenizer, one_config)
    train_dataset, val_dataset, train_dataloader, val_dataloader = utils.create_mixture_train_val_loaders(
                                                                        tokenized_dataset,
                                                                        data_collator,
                                                                        one_config,
                                                                    )

    train_eval_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=training_cfg["batch_size"],
        collate_fn=data_collator,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(
            training_cfg.get("weight_decay", 0.0)
        ),
    )

    accumulation_steps = training_cfg.get("gradient_accumulation_steps",1)
    early_stopping_cfg = config["model_bank"].get("early_stopping", {})
    max_epochs = int(early_stopping_cfg.get("max_epochs", 30))
    patience = int(early_stopping_cfg.get("patience", 3))
    min_delta = float(early_stopping_cfg.get("min_delta", 0.001))

    dataset_dir = source_info["dataset_dir"]
    lambda_dir_name = (f"lambda_{lambda_:g}")

    save_dir = os.path.join(dataset_dir,lambda_dir_name)
    analysis_dir = os.path.join(save_dir, "training_analysis")

    os.makedirs(save_dir,exist_ok=True,)
    os.makedirs(analysis_dir,exist_ok=True)

    optim_losses = []
    train_losses = []
    val_losses = []
    train_ppls = []
    val_ppls = []
    train_accs = []
    val_accs = []

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopping_reason = "max_epochs_reached"

    for epoch in range(max_epochs):
        if lambda_ == 0.0:
            optim_loss = utils.train_one_epoch_accumulated(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                device=device,
                accumulation_steps=accumulation_steps,
            )
        else:
            optim_loss = utils.KL_DRO_one_epoch_accumulated(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                gamma=float(training_cfg["gamma"]),
                lambd=lambda_,
                rho=float(training_cfg.get("rho", 0.0)),
                device=device,
                accumulation_steps=accumulation_steps,
            )

        train_loss, train_ppl, train_acc, _ = utils.evaluation(model, train_eval_dataloader, device=device)
        val_loss, val_ppl, val_acc, _ = utils.evaluation(model, val_dataloader, device=device)

        optim_losses.append(optim_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        improvement = best_val_loss - val_loss

        if improvement > min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            print(
                f"New best KL-DRO model saved at "
                f"epoch {epoch + 1}, "
                f"val_loss={val_loss:.4f}.",
                flush=True,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"dataset: {dataset_name} | "
            f"lambda: {lambda_:g} | "
            f"optim loss: {optim_loss:.4f} | "
            f"train loss: {train_loss:.4f} | "
            f"train ppl: {train_ppl:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val ppl: {val_ppl:.4f} | "
            f"best val loss: {best_val_loss:.4f}",
            flush=True,
        )

        if epochs_without_improvement >= patience:
            stopping_reason = "early_stopping"

            print(
                f"Early stopping at epoch "
                f"{epoch + 1}. Best epoch: "
                f"{best_epoch + 1}.",
                flush=True,
            )

            break
    epochs_run = len(val_losses)

    history_df = pd.DataFrame(
        {
            "epoch": range(1, epochs_run + 1),
            "optim_loss": optim_losses,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_ppl": train_ppls,
            "val_ppl": val_ppls,
            "train_acc": train_accs,
            "val_acc": val_accs,
        }
    )

    history_path = os.path.join(analysis_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    training_curves_path = plot_training_curves(
        results=results,
        dataset_name=dataset_name,
        lambda_=lambda_,
        output_dir=analysis_dir,
    )

    summary = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "lambda": lambda_,
        "gamma": training_cfg["gamma"],
        "rho": training_cfg.get("rho", 0.0),
        "dataset_size": training_cfg["dataset_size"],
        "batch_size": training_cfg["batch_size"],
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": training_cfg["batch_size"] * accumulation_steps,
        "context_length": training_cfg["context_length"],
        "learning_rate": training_cfg["learning_rate"],
        "weight_decay": training_cfg.get("weight_decay", 0.0),
        "seed": seed,
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val_loss,
        "best_val_ppl": val_ppls[best_epoch],
        "best_val_acc": val_accs[best_epoch],
        "stopping_reason": stopping_reason,
        "save_dir": save_dir,
        "history_path": history_path,
        "training_curves_path": training_curves_path,
        "source_distribution_path": source_info["source_distribution_path"],
        "source_losses_path": source_info["source_losses_path"],
        "lambda_rho_curve_path": source_info["lambda_rho_curve_path"],
    }

    summary_path = os.path.join(analysis_dir, "training_summary.csv")

    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    metadata_path = os.path.join(save_dir, "metadata.pth")

    torch.save(
        {
            **summary,
            "dataset_config": dataset_cfg,
            "training_config": training_cfg,
        },
        metadata_path,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        **summary,
        "metadata_path": metadata_path,
    }

def run_model_bank(config, datasets_config, device):
    """Train the complete KL-DRO model bank."""

    save_path = config["model_bank"]["save_path"]
    os.makedirs(save_path, exist_ok=True)

    source_infos = {}

    for dataset_name, dataset_config in datasets_config.items():
        source_infos[dataset_name] = prepare_one_source_dataset(
                                        config=config,
                                        dataset_name=dataset_name,
                                        dataset_config=dataset_config,
                                        device=device,
                                    )
                                

    one_configs = collect_model_bank_configs(config=config, datasets_config=datasets_config)

    results = []

    for one_config in one_configs:
        dataset_name = one_config["dataset_name"]

        result = train_one_bank_model(one_config=one_config, source_info=source_infos[dataset_name], config=config, device=device)
        results.append(result)

    metadata_df = pd.DataFrame(results)

    csv_path = os.path.join(save_path, "model_bank_metadata.csv")

    metadata_df.to_csv(csv_path, index=False)

    print(
        "KL-DRO model bank training completed.",
        flush=True,
    )

    print(
        f"Metadata saved at: {csv_path}",
        flush=True,
    )
if __name__ == "__main__":
    config = utils.load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_model_bank(config, datasets_config, device)
