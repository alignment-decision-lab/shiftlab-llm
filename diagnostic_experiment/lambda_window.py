from shiftlab.data.load_datasets import load_dataset_from_config
import utils
import shift_measurement as sm
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import matplotlib.pyplot as plt
import os

# -- DATASETS CONFIGURATIONS --
# Proxy
owt2_config = {
    "dataset": {
        "type": "hf_text",
        "name": "Skylion007/openwebtext",
        "split": "train",
        "text_column": "text",
        "streaming": True
    },
    "training": {
        "dataset_size": 50000,
        "batch_size": 16,
        "context_length": 64
    }
}
# Exact subset
pilecc_config = {
    "dataset": {
        "type": "hf_text",
        "name": "timaeus/pile-pile-cc",
        "split": "train",
        "text_column": "text",
        "streaming": True
    },
    "training": {
        "dataset_size": 50000,
        "batch_size": 16,
        "context_length": 64
    }
}
# Exact Subset
pubmed_config = {
    "dataset": {
        "type": "hf_text",
        "name": "timaeus/pile-pubmed_abstracts",
        "split": "train",
        "text_column": "text",
        "streaming": True
    },
    "training": {
        "dataset_size": 50000,
        "batch_size": 16,
        "context_length": 64
    }
}

# ---- METRICS ----

def compute_adversarial_weights(losses, lambda_val):
    """Compute adversarial weights for a list of losses."""
    losses = np.asarray(losses, dtype=np.float64)

    scaled = losses / lambda_val
    weights = np.exp(scaled - np.max(scaled)) # For Numerical stability.

    q = weights / np.sum(weights)

    return q

def compute_kl_to_uniform(q, eps=1e-12): 
    """ Compute KL divergence to uniform distribution for a given probability distribution q."""
    q = np.asarray(q, dtype=np.float64)
    n = len(q)

    return np.sum(q * (np.log(q + eps) + np.log(n)))

def compute_rho_for_lambda(losses, lambda_val):
    """Compute rho for a given lambda value."""
    q = compute_adversarial_weights(losses, lambda_val)
    rho = compute_kl_to_uniform(q)
    return rho

def compute_lambda_rho_curve(losses, lambda_grid, output_csv_path):
    """Compute the lambda-rho curve for a given list of losses and a grid of lambda values."""
    rows = []
    for dataset_name, dataset_losses in losses.items():

        for lambda_ in lambda_grid:
            rho = compute_rho_for_lambda(dataset_losses, lambda_)
            rows.append({
                "dataset": dataset_name,
                "lambda": lambda_,
                "rho": rho,
                })
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return df

def find_lambda_for_rho(curve_df, dataset_name, target_rho):
    """ Given a lambda-rho curve DataFrame, find the lambda value corresponding to a target rho for a specific dataset. """
    subset = curve_df[curve_df["dataset"] == dataset_name].sort_values("rho")

    rhos = subset["rho"].values
    lambdas = subset["lambda"].values

    if target_rho < rhos.min() or target_rho > rhos.max():
        return np.nan

    return np.interp(target_rho, rhos, lambdas)

def compute_rho_lambda_matrix(deploy_batches, reference_dataloaders, curve_df, vocab_size):
    """ Compute the rho-lambda matrix for given deploy batches and reference dataloaders. """
    rows = []

    for batch_name, batch in deploy_batches.items():
        for ref_name, ref_loader in reference_dataloaders.items():

            rho = sm.compute_token_kl_batch_vs_dataloader(
                batch,
                ref_loader,
                vocab_size=vocab_size
            )

            lambda_hat = find_lambda_for_rho(
                curve_df,
                dataset_name=ref_name,
                target_rho=rho
            )

            rows.append({
                "batch_dataset": batch_name,
                "reference_dataset": ref_name,
                "token_rho": rho,
                "lambda_hat": lambda_hat,
            })

    return pd.DataFrame(rows)

# ---- PLOTTING ----

def plot_single_reference_curves(curve_df, points_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    point_markers = {
        "owt2": "X",
        "pilecc": "P",
        "pubmed": "D",
    }

    point_colors = {
        "owt2": "black",
        "pilecc": "magenta",
        "pubmed": "cyan",
    }

    plot_paths = []

    for ref_name in curve_df["dataset"].unique():
        curve_subset = curve_df[curve_df["dataset"] == ref_name]
        points_subset = points_df[points_df["reference_dataset"] == ref_name]

        plt.figure(figsize=(7, 5))

        plt.plot(
            curve_subset["lambda"],
            curve_subset["rho"],
            marker="o",
            label=f"curve: {ref_name}"
        )

        for _, row in points_subset.iterrows():
            if np.isnan(row["lambda_hat"]):
                continue

            plt.scatter(
                row["lambda_hat"],
                row["token_rho"],
                s=140,
                marker=point_markers[row["batch_dataset"]],
                color=point_colors[row["batch_dataset"]],
                edgecolors="black",
                linewidths=1.0,
                label=f'batch: {row["batch_dataset"]}'
            )

            plt.annotate(
                row["batch_dataset"],
                (row["lambda_hat"], row["token_rho"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9
            )

        plt.xscale("log")
        plt.xlabel("lambda")
        plt.ylabel("rho")
        plt.title(f"Token-KL projection on {ref_name} lambda-rho curve")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        plot_path = os.path.join(
            output_dir,
            f"lambda_rho_projection_{ref_name}.png"
        )

        plt.savefig(plot_path, dpi=300)
        plt.close()

        plot_paths.append(plot_path)

    return plot_paths

# ---- MAIN ----

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ------- Load datasets configurations -------
    owt2_dataset = load_dataset_from_config(owt2_config)
    pilecc_dataset = load_dataset_from_config(pilecc_config)
    pubmed_dataset = load_dataset_from_config(pubmed_config)
    print("OWT2 loaded:", len(owt2_dataset))
    print("PILE-CC loaded:", len(pilecc_dataset))
    print("PUBMED loaded:", len(pubmed_dataset))

    # ------ Load tokenizer/model -------
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.to(device)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # ----- Tokenize datasets -----
    owt2_tokenized_dataset = utils.tokenize_dataset(
        owt2_dataset,
        tokenizer,
        owt2_config
    )
    pilecc_tokenized_dataset = utils.tokenize_dataset(
        pilecc_dataset,
        tokenizer,
        pilecc_config
    )
    pubmed_tokenized_dataset = utils.tokenize_dataset(
        pubmed_dataset,
        tokenizer,
        pubmed_config
    )
    # ----- Create dataloaders -----
    owt2_deploy_dataloader = sm.create_deploy_dataloader(
        owt2_tokenized_dataset,
        data_collator,
        owt2_config
    )

    pilecc_deploy_dataloader = sm.create_deploy_dataloader(
        pilecc_tokenized_dataset,
        data_collator,
        pilecc_config
    )

    pubmed_deploy_dataloader = sm.create_deploy_dataloader(
        pubmed_tokenized_dataset,
        data_collator,
        pubmed_config
    )

    owt2_reference_dataloader = sm.create_reference_dataloader(
        owt2_tokenized_dataset,
        data_collator,
        owt2_config,
        seed=42
    )

    pilecc_reference_dataloader = sm.create_reference_dataloader(
        pilecc_tokenized_dataset,
        data_collator,
        pilecc_config,
        seed=42
    )

    pubmed_reference_dataloader = sm.create_reference_dataloader(
        pubmed_tokenized_dataset,
        data_collator,
        pubmed_config,
        seed=42
    )
    
    # ----- Create batches and reference dataloaders dictionaries -----
    deploy_batches = {
        "owt2": sm.get_first_batch(owt2_deploy_dataloader),
        "pilecc": sm.get_first_batch(pilecc_deploy_dataloader),
        "pubmed": sm.get_first_batch(pubmed_deploy_dataloader),
    }

    reference_dataloaders = {
        "owt2": owt2_reference_dataloader,
        "pilecc": pilecc_reference_dataloader,
        "pubmed": pubmed_reference_dataloader,
    }

    # ----- Compute losses -----
    owt2_losses = utils.compute_sample_losses(model, owt2_tokenized_dataset, data_collator, device, owt2_config)
    pilecc_losses = utils.compute_sample_losses(model, pilecc_tokenized_dataset, data_collator, device, pilecc_config)
    pubmed_losses = utils.compute_sample_losses(model, pubmed_tokenized_dataset, data_collator, device, pubmed_config)

    losses = {
        "owt2": owt2_losses,
        "pilecc": pilecc_losses,
        "pubmed": pubmed_losses
    }

    # ----- Compute lambda-grid -----
    lambda_grid = np.logspace(-2, 2, 50)

    # ----- Compute lambda-rho curve -----
    output_dir = "outputs/diagnostic/lambda_rho_window"
    csv_path = os.path.join(output_dir, "lambda_rho_curve.csv")
    lambda_rho_curve_df = compute_lambda_rho_curve(losses, lambda_grid, csv_path)
    print("Lambda-rho curve CSV saved to:", csv_path)

    # ---- Compute rho-lambda matrix -----
    rho_lambda_matrix_df = compute_rho_lambda_matrix(
        deploy_batches=deploy_batches,
        reference_dataloaders=reference_dataloaders,
        curve_df=lambda_rho_curve_df,
        vocab_size=tokenizer.vocab_size
    )

    matrix_csv_path = os.path.join(output_dir, "rho_lambda_matrix.csv")
    rho_lambda_matrix_df.to_csv(matrix_csv_path, index=False)
    print("Rho-lambda matrix saved to:", matrix_csv_path)

    # ----- Plot lambda-rho curve -----
    plot_paths = plot_single_reference_curves(
        lambda_rho_curve_df,
        rho_lambda_matrix_df,
        output_dir
    )

    for path in plot_paths:
        print("Saved projection plot:", path)

    print("Done.")



