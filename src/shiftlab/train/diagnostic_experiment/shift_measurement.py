from shiftlab.data.load_datasets import load_dataset_from_config
from shiftlab.train.diagnostic_experiment import utils
import argparse
import yaml
import torch
import math
import random
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from datasets import concatenate_datasets
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
        "dataset_size": 1000,
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
        "dataset_size": 1000,
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
        "dataset_size": 1000,
        "batch_size": 16,
        "context_length": 64
    }
}

# -- BATCH PREPARATIONS --

def create_deploy_dataloader(tokenized_dataset, data_collator, config):
    deploy_dataloader = DataLoader(
        tokenized_dataset,
        shuffle=False,
        batch_size=config["training"]["batch_size"],
        collate_fn=data_collator
    )
    return deploy_dataloader


def create_reference_dataloader(tokenized_dataset, data_collator, config, seed):
    generator = torch.Generator().manual_seed(seed)

    reference_dataloader = DataLoader(
        tokenized_dataset,
        shuffle=True,
        batch_size=config["training"]["batch_size"],
        collate_fn=data_collator,
        generator=generator
    )
    return reference_dataloader

def get_first_batch(dataloader):
    return next(iter(dataloader))

def get_n_batches(dataloader, n_batches):
    batches = []
    iterator = iter(dataloader)

    for _ in range(n_batches):
        try:
            batches.append(next(iterator))
        except StopIteration:
            break

    return batches

# -- SHIFT MEASURES --

# - Token KL -

def compute_batch_token_distribution(batch, vocab_size, epsilon=1e-8):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    valid_tokens = input_ids[attention_mask == 1]
    counts = torch.bincount(valid_tokens, minlength=vocab_size) # Counts the number of times the token appears in the batch, there is vocab_size tokens in total.
    counts = counts.float()
    counts = counts + epsilon # Some tokens may never appear so their probability is 0, which can be bothersome when computing the KL divergence.

    p = counts / counts.sum() # Computing the frequency of each token.

    return p

def compute_dataset_token_distribution(dataloader, vocab_size, epsilon=1e-8):
    counts = torch.zeros(vocab_size)

    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        valid_tokens = input_ids[attention_mask == 1]

        counts += torch.bincount(
            valid_tokens.reshape(-1),
            minlength=vocab_size
        ).float()

    counts = counts + epsilon
    p = counts / counts.sum()

    return p

def compute_kl(p, q):
    return torch.sum(p * torch.log(p / q)).item()

def compute_token_kl_batch_vs_batch(batch_p, batch_q, vocab_size, epsilon=1e-8):
    p = compute_batch_token_distribution(
        batch_p,
        vocab_size=vocab_size,
        epsilon=epsilon
    )

    q = compute_batch_token_distribution(
        batch_q,
        vocab_size=vocab_size,
        epsilon=epsilon
    )

    kl = compute_kl(p, q)

    return kl

def compute_token_kl_batch_vs_dataloader(batch, dataloader, vocab_size, epsilon=1e-8):
    p_batch = compute_batch_token_distribution(
        batch,
        vocab_size=vocab_size,
        epsilon=epsilon
    )

    p_ref = compute_dataset_token_distribution(
        dataloader,
        vocab_size=vocab_size,
        epsilon=epsilon
    )

    kl = compute_kl(p_batch, p_ref)

    return kl


def compute_dataset_token_distribution_with_budget(
    dataloader,
    vocab_size,
    max_tokens,
    epsilon=1e-8,
):
    """
    Compute a token distribution using exactly max_tokens valid tokens.
    Padding tokens are ignored.
    """
    counts = torch.zeros(vocab_size, dtype=torch.float64)
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        valid_tokens = input_ids[attention_mask == 1].reshape(-1).cpu()

        remaining_tokens = max_tokens - total_tokens

        if remaining_tokens <= 0:
            break

        # Use only the number of tokens still needed.
        valid_tokens = valid_tokens[:remaining_tokens]

        counts += torch.bincount(
            valid_tokens,
            minlength=vocab_size,
        ).to(torch.float64)

        total_tokens += valid_tokens.numel()

    if total_tokens < max_tokens:
        raise ValueError(
            f"Only {total_tokens:,} valid tokens were available, "
            f"but {max_tokens:,} were requested."
        )

    counts += epsilon
    distribution = counts / counts.sum()

    return distribution, total_tokens

# - Embedding KL - 

def compute_valid_hidden_states(batch, model, device):
    batch = utils.move_batch_to_device(batch, device)

    model.eval()
    with torch.no_grad():
        outputs = model.transformer(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

    hidden_states = outputs.last_hidden_state
    attention_mask = batch["attention_mask"]

    valid_hidden_states = hidden_states[attention_mask == 1]

    return valid_hidden_states

def compute_dataloader_hidden_states(dataloader, model, device, max_tokens=None):
    all_embeddings = []
    total_tokens = 0

    for batch in dataloader:
        embeddings = compute_valid_hidden_states(batch, model, device)
        all_embeddings.append(embeddings.cpu())
        total_tokens += embeddings.shape[0]

        if max_tokens is not None and total_tokens >= max_tokens:
            break

    embeddings = torch.cat(all_embeddings, dim=0)

    if max_tokens is not None:
        embeddings = embeddings[:max_tokens]

    return embeddings


# Embedding Mean

def compute_embedding_mean(batch, model, device):
    embeddings = compute_valid_hidden_states(batch, model, device)
    mu = embeddings.mean(dim=0)
    return mu

def compute_dataloader_embedding_mean(dataloader, model, device, max_tokens=None):
    embeddings = compute_dataloader_hidden_states(
        dataloader,
        model,
        device,
        max_tokens=max_tokens
    )

    mu = embeddings.mean(dim=0)
    return mu

def compute_l2_distance(mu_p, mu_q):
    return torch.norm(mu_p - mu_q).item()

def compute_embedding_mean_batch_vs_batch(batch_p, batch_q, model, device):
    mu_p = compute_embedding_mean(batch_p, model, device).cpu()
    mu_q = compute_embedding_mean(batch_q, model, device).cpu()

    return compute_l2_distance(mu_p, mu_q)

def compute_embedding_mean_batch_vs_dataloader(
    batch_p,
    dataloader_q,
    model,
    device,
    max_tokens=None
):
    mu_p = compute_embedding_mean(batch_p, model, device).cpu()

    mu_q = compute_dataloader_embedding_mean(
        dataloader_q,
        model,
        device,
        max_tokens=max_tokens
    ).cpu()

    return compute_l2_distance(mu_p, mu_q)


# Diag Gaussian KL

def compute_diag_gaussian_stats(embeddings, epsilon=1e-5):
    mu = embeddings.mean(dim=0)
    var = embeddings.var(dim=0, unbiased=False)
    var = var + epsilon

    return mu, var

def compute_diag_gaussian_kl(mu_p, var_p, mu_q, var_q):
    kl_per_dim = (
        torch.log(var_q / var_p)
        + (var_p + (mu_p - mu_q) ** 2) / var_q
        - 1
    )

    kl = 0.5 * torch.sum(kl_per_dim)

    return kl.item()

def compute_diag_gaussian_kl_batch_vs_batch(batch_p, batch_q, model, device, epsilon=1e-5):
    embeddings_p = compute_valid_hidden_states(batch_p, model, device).cpu()
    embeddings_q = compute_valid_hidden_states(batch_q, model, device).cpu()

    mu_p, var_p = compute_diag_gaussian_stats(embeddings_p, epsilon=epsilon)
    mu_q, var_q = compute_diag_gaussian_stats(embeddings_q, epsilon=epsilon)

    return compute_diag_gaussian_kl(mu_p, var_p, mu_q, var_q)

def compute_diag_gaussian_kl_batch_vs_dataloader(
    batch_p,
    dataloader_q,
    model,
    device,
    max_tokens=None,
    epsilon=1e-5
):
    embeddings_p = compute_valid_hidden_states(batch_p, model, device).cpu()

    embeddings_q = compute_dataloader_hidden_states(
        dataloader_q,
        model,
        device,
        max_tokens=max_tokens
    ).cpu()

    mu_p, var_p = compute_diag_gaussian_stats(embeddings_p, epsilon=epsilon)
    mu_q, var_q = compute_diag_gaussian_stats(embeddings_q, epsilon=epsilon)

    return compute_diag_gaussian_kl(mu_p, var_p, mu_q, var_q)


# PCA Gaussian KL

def fit_pca(embeddings, n_components):
    max_components = min(
        embeddings.shape[0] - 1,
        embeddings.shape[1],
    )

    n_components = min(
        int(n_components),
        int(max_components),
    )

    if n_components < 1:
        raise ValueError(
            "Not enough embeddings to fit PCA."
        )

    pca_mean = embeddings.mean(dim=0)
    x_centered = embeddings - pca_mean

    _, _, components = torch.pca_lowrank(
        x_centered,
        q=n_components,
    )

    return pca_mean, components[:, :n_components]

def project_pca(embeddings, pca_mean, pca_components):
    return (embeddings - pca_mean) @ pca_components

def compute_full_gaussian_stats(embeddings, epsilon=1e-5):
    mu = embeddings.mean(dim=0)
    X = embeddings - mu
    cov = (X.T @ X) / X.shape[0]

    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    cov = cov + epsilon * eye # To regularize for later to compute the inverse.

    return mu, cov

def compute_full_gaussian_kl(mu_p, cov_p, mu_q, cov_q):
    d = mu_p.numel()
    sign_p, logdet_p = torch.linalg.slogdet(cov_p)
    sign_q, logdet_q = torch.linalg.slogdet(cov_q)

    if sign_p <= 0 or sign_q <= 0:
        raise ValueError("Covariance matrix is not positive definite.")

    trace_term = torch.trace(torch.linalg.solve(cov_q, cov_p))

    diff = (mu_q - mu_p).unsqueeze(1)
    mahalanobis = (diff.T @ torch.linalg.solve(cov_q, diff)).squeeze()

    kl = 0.5 * (logdet_q - logdet_p - d + trace_term + mahalanobis)
    return kl.item()

def compute_pca_gaussian_kl_from_embeddings(
    embeddings_p,
    embeddings_q,
    n_components=50,
    epsilon=1e-5
):
    n_components = min(
        n_components,
        embeddings_p.shape[0] - 1,
        embeddings_q.shape[0] - 1,
        embeddings_p.shape[1],
    )

    if n_components < 1:
        raise ValueError("Not enough embeddings to fit PCA.")

    combined_embeddings = torch.cat([embeddings_p, embeddings_q], dim=0)

    pca_mean, pca_components = fit_pca(
        combined_embeddings,
        n_components=n_components
    )

    z_p = project_pca(embeddings_p, pca_mean, pca_components)
    z_q = project_pca(embeddings_q, pca_mean, pca_components)

    mu_p, cov_p = compute_full_gaussian_stats(z_p, epsilon=epsilon)
    mu_q, cov_q = compute_full_gaussian_stats(z_q, epsilon=epsilon)

    return compute_full_gaussian_kl(mu_p, cov_p, mu_q, cov_q)

def compute_pca_gaussian_kl_batch_vs_batch(
    batch_p,
    batch_q,
    model,
    device,
    n_components=50,
    epsilon=1e-5
):
    embeddings_p = compute_valid_hidden_states(batch_p, model, device).cpu()
    embeddings_q = compute_valid_hidden_states(batch_q, model, device).cpu()

    return compute_pca_gaussian_kl_from_embeddings(
        embeddings_p,
        embeddings_q,
        n_components=n_components,
        epsilon=epsilon
    )

def compute_pca_gaussian_kl_batch_vs_dataloader(
    batch_p,
    dataloader_q,
    model,
    device,
    n_components=50,
    max_tokens=None,
    epsilon=1e-5
):
    embeddings_p = compute_valid_hidden_states(batch_p, model, device).cpu()

    embeddings_q = compute_dataloader_hidden_states(
        dataloader_q,
        model,
        device,
        max_tokens=max_tokens
    ).cpu()

    return compute_pca_gaussian_kl_from_embeddings(
        embeddings_p,
        embeddings_q,
        n_components=n_components,
        epsilon=epsilon
    )

def compute_dataset_shift_statistics(
    dataloader,
    model,
    device,
    vocab_size,
    max_embedding_tokens=20000,
    token_distribution_max_tokens=500_000,
    token_distribution=None,
    token_epsilon=1e-8,
    gaussian_epsilon=1e-5,
):
    """
    Compute reusable token and embedding statistics for one dataset.

    If token_distribution is provided, it is reused directly.
    This is used for source datasets whose official distributions
    were already computed by models_bank.py.

    Otherwise, a new distribution is computed from exactly
    token_distribution_max_tokens valid tokens.
    """

    if token_distribution is None:
        token_distribution, n_tokens_used = (
            compute_dataset_token_distribution_with_budget(
                dataloader=dataloader,
                vocab_size=vocab_size,
                max_tokens=token_distribution_max_tokens,
                epsilon=token_epsilon,
            )
        )

        if n_tokens_used != token_distribution_max_tokens:
            raise RuntimeError(
                f"{n_tokens_used} tokens used instead of "
                f"{token_distribution_max_tokens}."
            )

        token_distribution = token_distribution.cpu()

    else:
        token_distribution = (
            torch.as_tensor(token_distribution)
            .detach()
            .cpu()
            .to(torch.float64)
        )

        if token_distribution.numel() != vocab_size:
            raise ValueError(
                "The provided token distribution has "
                f"{token_distribution.numel()} entries, "
                f"but vocab_size={vocab_size}."
            )

        token_distribution = (
            token_distribution
            / token_distribution.sum()
        )

    embeddings = compute_dataloader_hidden_states(
        dataloader=dataloader,
        model=model,
        device=device,
        max_tokens=max_embedding_tokens,
    ).float().cpu()

    embedding_mean = embeddings.mean(dim=0)

    diag_mean, diag_var = compute_diag_gaussian_stats(
        embeddings,
        epsilon=gaussian_epsilon,
    )

    return {
        "token_distribution": token_distribution,
        "embeddings": embeddings,
        "embedding_mean": embedding_mean,
        "diag_mean": diag_mean,
        "diag_var": diag_var,
        "num_embedding_tokens": int(
            embeddings.shape[0]
        ),
    }

def precompute_all_dataset_statistics(
    dataset_dataloaders,
    model,
    device,
    vocab_size,
    source_token_distributions=None,
    token_distribution_max_tokens=500_000,
    max_embedding_tokens=20000,
    token_epsilon=1e-8,
    gaussian_epsilon=1e-5,
):
    """
    Precompute shift statistics for all source and deployment datasets.

    Source datasets reuse the official token distributions produced
    by models_bank.py. Deployment distributions are computed using
    a fixed token budget.
    """

    if source_token_distributions is None:
        source_token_distributions = {}

    all_stats = {}

    for dataset_name, dataloader in dataset_dataloaders.items():
        print(
            f"Computing shift statistics for "
            f"{dataset_name}...",
            flush=True,
        )

        official_source_distribution = (
            source_token_distributions.get(
                dataset_name
            )
        )

        all_stats[dataset_name] = (
            compute_dataset_shift_statistics(
                dataloader=dataloader,
                model=model,
                device=device,
                vocab_size=vocab_size,
                max_embedding_tokens=(
                    max_embedding_tokens
                ),
                token_distribution_max_tokens=(
                    token_distribution_max_tokens
                ),
                token_distribution=(
                    official_source_distribution
                ),
                token_epsilon=token_epsilon,
                gaussian_epsilon=gaussian_epsilon,
            )
        )

    return all_stats

def fit_common_pca_from_dataset_statistics(
    all_stats,
    n_components=20,
    max_tokens_per_dataset=5000,
):
    """
    Fit one common PCA basis shared by all datasets and batches.
    """
    pooled_embeddings = []

    for stats in all_stats.values():
        embeddings = stats["embeddings"]

        pooled_embeddings.append(
            embeddings[:max_tokens_per_dataset]
        )

    pooled_embeddings = torch.cat(
        pooled_embeddings,
        dim=0,
    )

    pca_mean, pca_components = fit_pca(
        pooled_embeddings,
        n_components=n_components,
    )

    return pca_mean, pca_components

def add_pca_statistics(
    all_stats,
    pca_mean,
    pca_components,
    components_list=(5, 10, 20),
    epsilon=1e-5,
):
    """
    Project all datasets into the same PCA space and compute
    Gaussian statistics for each requested dimension.
    """
    for dataset_name, stats in all_stats.items():
        embeddings = stats["embeddings"]

        full_projection = project_pca(
            embeddings,
            pca_mean,
            pca_components,
        )

        stats["pca"] = {}

        for n_components in components_list:
            projected = full_projection[:, :n_components]

            mean, covariance = compute_full_gaussian_stats(
                projected,
                epsilon=epsilon,
            )

            stats["pca"][n_components] = {
                "mean": mean,
                "covariance": covariance,
            }

    return all_stats

def compute_dataset_to_source_distances(
    all_stats,
    deployment_names,
    source_names,
    pca_components_list=(5, 10, 20),
):
    """
    Compute dataset-to-source distances for every deployment/source pair.
    """
    distances = {}

    for deployment_name in deployment_names:
        distances[deployment_name] = {}

        deployment_stats = all_stats[deployment_name]

        for source_name in source_names:
            source_stats = all_stats[source_name]

            pair_distances = {
                "token_kl": compute_kl(
                    deployment_stats["token_distribution"],
                    source_stats["token_distribution"],
                ),

                "embedding_mean_l2": compute_l2_distance(
                    deployment_stats["embedding_mean"],
                    source_stats["embedding_mean"],
                ),

                "diag_gaussian_kl": compute_diag_gaussian_kl(
                    deployment_stats["diag_mean"],
                    deployment_stats["diag_var"],
                    source_stats["diag_mean"],
                    source_stats["diag_var"],
                ),
            }

            for n_components in pca_components_list:
                deployment_pca = deployment_stats["pca"][
                    n_components
                ]
                source_pca = source_stats["pca"][
                    n_components
                ]

                pair_distances[
                    f"pca_gaussian_kl_{n_components}"
                ] = compute_full_gaussian_kl(
                    deployment_pca["mean"],
                    deployment_pca["covariance"],
                    source_pca["mean"],
                    source_pca["covariance"],
                )

            distances[deployment_name][
                source_name
            ] = pair_distances

    return distances

def compute_batch_to_source_distances(
    batch,
    source_statistics,
    reference_model,
    device,
    pca_mean,
    pca_components,
    vocab_size,
    pca_components_list=(5, 10, 20),
    token_epsilon=1e-8,
    gaussian_epsilon=1e-5,
):
    """
    Compute all distances from one arriving batch to each source dataset.
    """
    batch_token_distribution = compute_batch_token_distribution(
        batch,
        vocab_size=vocab_size,
        epsilon=token_epsilon,
    ).cpu()

    batch_embeddings = compute_valid_hidden_states(
        batch,
        reference_model,
        device,
    ).float().cpu()

    batch_mean = batch_embeddings.mean(dim=0)

    batch_diag_mean, batch_diag_var = compute_diag_gaussian_stats(
        batch_embeddings,
        epsilon=gaussian_epsilon,
    )

    full_batch_projection = project_pca(
        batch_embeddings,
        pca_mean,
        pca_components,
    )

    batch_pca_stats = {}

    for n_components in pca_components_list:
        projected = full_batch_projection[:, :n_components]

        mean, covariance = compute_full_gaussian_stats(
            projected,
            epsilon=gaussian_epsilon,
        )

        batch_pca_stats[n_components] = {
            "mean": mean,
            "covariance": covariance,
        }

    distances = {}

    for source_name, source_stats in source_statistics.items():
        source_distances = {
            "token_kl": compute_kl(
                batch_token_distribution,
                source_stats["token_distribution"],
            ),

            "embedding_mean_l2": compute_l2_distance(
                batch_mean,
                source_stats["embedding_mean"],
            ),

            "diag_gaussian_kl": compute_diag_gaussian_kl(
                batch_diag_mean,
                batch_diag_var,
                source_stats["diag_mean"],
                source_stats["diag_var"],
            ),
        }

        for n_components in pca_components_list:
            source_pca = source_stats["pca"][
                n_components
            ]

            source_distances[
                f"pca_gaussian_kl_{n_components}"
            ] = compute_full_gaussian_kl(
                batch_pca_stats[n_components]["mean"],
                batch_pca_stats[n_components]["covariance"],
                source_pca["mean"],
                source_pca["covariance"],
            )

        distances[source_name] = source_distances

    return distances

# -----------------------------------------
# ---- EXPERIMENT 1: NOISE CALIBRATION ----
# -----------------------------------------

def run_noise_calibration(
    fixed_batch,
    reference_dataloader,
    random_batches,
    model,
    device,
    vocab_size,
    output_csv_path,
    max_tokens=20000,
    pca_components_list=(5, 10, 20, 50),
    token_epsilon=1e-8,
    gaussian_epsilon=1e-5
):
    rows = []

    metric_fns = {
        "token_kl": {
            "batch_vs_dataloader": lambda: compute_token_kl_batch_vs_dataloader(
                fixed_batch,
                reference_dataloader,
                vocab_size=vocab_size,
                epsilon=token_epsilon
            ),
            "batch_vs_batch": lambda batch_q: compute_token_kl_batch_vs_batch(
                fixed_batch,
                batch_q,
                vocab_size=vocab_size,
                epsilon=token_epsilon
            ),
        },

        "embedding_mean_distance": {
            "batch_vs_dataloader": lambda: compute_embedding_mean_batch_vs_dataloader(
                fixed_batch,
                reference_dataloader,
                model,
                device,
                max_tokens=max_tokens
            ),
            "batch_vs_batch": lambda batch_q: compute_embedding_mean_batch_vs_batch(
                fixed_batch,
                batch_q,
                model,
                device
            ),
        },

        "diag_gaussian_kl": {
            "batch_vs_dataloader": lambda: compute_diag_gaussian_kl_batch_vs_dataloader(
                fixed_batch,
                reference_dataloader,
                model,
                device,
                max_tokens=max_tokens,
                epsilon=gaussian_epsilon
            ),
            "batch_vs_batch": lambda batch_q: compute_diag_gaussian_kl_batch_vs_batch(
                fixed_batch,
                batch_q,
                model,
                device,
                epsilon=gaussian_epsilon
            ),
        },
    }

    for metric_name, fns in metric_fns.items():
        print(f"\nRunning metric: {metric_name}")

        # 1. Fixed batch vs full/reference dataloader
        full_value = fns["batch_vs_dataloader"]()

        rows.append({
            "metric": metric_name,
            "comparison": "batch_vs_full_dataset",
            "mean": full_value,
            "std": 0.0,
            "n_random_batches": len(random_batches),
            "max_tokens": max_tokens,
            "n_components": n_components if metric_name == "pca_gaussian_kl" else None
        })

        print(f"  batch vs full dataset: {full_value:.6f}")

        # 2. Fixed batch vs random batches
        random_values = []

        for i, random_batch in enumerate(random_batches):
            value = fns["batch_vs_batch"](random_batch)
            random_values.append(value)
           #print(f"  random batch {i + 1}/{len(random_batches)}: {value:.6f}")

        random_values = np.array(random_values)

        rows.append({
            "metric": metric_name,
            "comparison": "batch_vs_random_batches",
            "mean": float(random_values.mean()),
            "std": float(random_values.std()),
            "n_random_batches": len(random_batches),
            "max_tokens": None,
            "n_components": n_components if metric_name == "pca_gaussian_kl" else None
        })

        print(
            f"  batch vs random batches: "
            f"{random_values.mean():.6f} ± {random_values.std():.6f}"
        )

    for n_components in pca_components_list:
        metric_name = f"pca_gaussian_kl_{n_components}"

        print(f"\nRunning metric: {metric_name}")

        full_value = compute_pca_gaussian_kl_batch_vs_dataloader(
            fixed_batch,
            reference_dataloader,
            model,
            device,
            n_components=n_components,
            max_tokens=max_tokens,
            epsilon=gaussian_epsilon
        )

        rows.append({
            "metric": metric_name,
            "comparison": "batch_vs_full_dataset",
            "mean": full_value,
            "std": 0.0,
            "n_random_batches": len(random_batches),
            "max_tokens": max_tokens,
            "n_components": n_components
        })

        print(f"  batch vs full dataset: {full_value:.6f}")

        random_values = []

        for random_batch in random_batches:
            value = compute_pca_gaussian_kl_batch_vs_batch(
                fixed_batch,
                random_batch,
                model,
                device,
                n_components=n_components,
                epsilon=gaussian_epsilon
            )
            random_values.append(value)

        random_values = np.array(random_values)

        rows.append({
            "metric": metric_name,
            "comparison": "batch_vs_random_batches",
            "mean": float(random_values.mean()),
            "std": float(random_values.std()),
            "n_random_batches": len(random_batches),
            "max_tokens": None,
            "n_components": n_components
        })

        print(
            f"  batch vs random batches: "
            f"{random_values.mean():.6f} ± {random_values.std():.6f}"
        )
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print(f"\nSaved noise calibration CSV to: {output_csv_path}")

    return df

def plot_noise_calibration(df, output_plot_path):
    metric_order = df["metric"].unique().tolist()

    comparison_order = [
        "batch_vs_full_dataset",
        "batch_vs_random_batches",
    ]

    x = np.arange(len(metric_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))

    for j, comparison in enumerate(comparison_order):
        means = []
        stds = []

        for metric in metric_order:
            row = df[
                (df["metric"] == metric)
                & (df["comparison"] == comparison)
            ].iloc[0]

            means.append(row["mean"])
            stds.append(row["std"])

        offset = (j - 0.5) * width

        label = (
            "fixed batch vs full dataset"
            if comparison == "batch_vs_full_dataset"
            else "fixed batch vs random batches"
        )

        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=label
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_order, rotation=20, ha="right")
    ax.set_ylabel("Distance")
    ax.set_title("Noise calibration on OWT2 vs OWT2")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"Saved noise calibration plot to: {output_plot_path}")

# ---------------------------------------------
# ---- EXPERIMENT 2: RAW SHIFT COMPARAISON ----
# ---------------------------------------------

def run_raw_shift_comparison(
    deploy_batches,
    owt2_reference_dataloader,
    model,
    device,
    vocab_size,
    output_csv_path,
    max_tokens=20000,
    pca_components_list=(5, 10, 20),
    token_epsilon=1e-8,
    gaussian_epsilon=1e-5
):
    rows = []

    for comparison_name, fixed_batch in deploy_batches.items():

        print(f"\n=== Comparison: {comparison_name} vs OWT2 reference ===")

        # Token KL
        rho = compute_token_kl_batch_vs_dataloader(
            fixed_batch,
            owt2_reference_dataloader,
            vocab_size=vocab_size,
            epsilon=token_epsilon
        )

        rows.append({
            "metric": "token_kl",
            "comparison": comparison_name,
            "rho": rho,
            "max_tokens": None,
            "n_components": None
        })

        print(f"token_kl: {rho:.6f}")

        # Embedding mean distance
        rho = compute_embedding_mean_batch_vs_dataloader(
            fixed_batch,
            owt2_reference_dataloader,
            model,
            device,
            max_tokens=max_tokens
        )

        rows.append({
            "metric": "embedding_mean_distance",
            "comparison": comparison_name,
            "rho": rho,
            "max_tokens": max_tokens,
            "n_components": None
        })

        print(f"embedding_mean_distance: {rho:.6f}")

        # Diag Gaussian KL
        rho = compute_diag_gaussian_kl_batch_vs_dataloader(
            fixed_batch,
            owt2_reference_dataloader,
            model,
            device,
            max_tokens=max_tokens,
            epsilon=gaussian_epsilon
        )

        rows.append({
            "metric": "diag_gaussian_kl",
            "comparison": comparison_name,
            "rho": rho,
            "max_tokens": max_tokens,
            "n_components": None
        })

        print(f"diag_gaussian_kl: {rho:.6f}")

        # PCA Gaussian KL for several dimensions
        for n_components in pca_components_list:
            metric_name = f"pca_gaussian_kl_{n_components}"

            rho = compute_pca_gaussian_kl_batch_vs_dataloader(
                fixed_batch,
                owt2_reference_dataloader,
                model,
                device,
                n_components=n_components,
                max_tokens=max_tokens,
                epsilon=gaussian_epsilon
            )

            rows.append({
                "metric": metric_name,
                "comparison": comparison_name,
                "rho": rho,
                "max_tokens": max_tokens,
                "n_components": n_components
            })

            print(f"{metric_name}: {rho:.6f}")

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print(f"\nSaved raw shift comparison CSV to: {output_csv_path}")

    return df

def plot_raw_shift_comparison(df_raw, output_plot_path):
    comparisons = ["intra_owt2", "near_pilecc", "far_pubmed"]
    metrics = df_raw["metric"].unique().tolist()

    x = np.arange(len(comparisons))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, metric in enumerate(metrics):
        values = []

        for comparison in comparisons:
            row = df_raw[
                (df_raw["metric"] == metric)
                & (df_raw["comparison"] == comparison)
            ].iloc[0]
            values.append(row["rho"])

        offset = (i - (len(metrics) - 1) / 2) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=metric
        )

    ax.set_xticks(x)
    ax.set_xticklabels(comparisons, rotation=15, ha="right")
    ax.set_ylabel("rho")
    ax.set_title("Raw shift distances to OWT2 reference")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"Saved raw shift comparison plot to: {output_plot_path}")



# ---------------------------------------------
# ---- EXPERIMENT 3: THE PILE COMPARAISON -----
# ---------------------------------------------

def compute_pile_relative_error(df_raw, output_csv_path):
    pile_delta_near = 0.0777
    pile_delta_far = 0.2584
    pile_ratio_far_near = pile_delta_far / pile_delta_near

    rows = []

    for metric in df_raw["metric"].unique():
        df_m = df_raw[df_raw["metric"] == metric]

        rho_intra = df_m[df_m["comparison"] == "intra_owt2"]["rho"].iloc[0]
        rho_near = df_m[df_m["comparison"] == "near_pilecc"]["rho"].iloc[0]
        rho_far = df_m[df_m["comparison"] == "far_pubmed"]["rho"].iloc[0]

        delta_near = rho_near - rho_intra
        delta_far = rho_far - rho_intra

        if abs(delta_near) < 1e-12:
            metric_ratio_far_near = np.nan
            error = np.nan
        else:
            metric_ratio_far_near = delta_far / delta_near
            error = abs(metric_ratio_far_near - pile_ratio_far_near)


        rows.append({
            "metric": metric,
            "rho_intra": rho_intra,
            "rho_near": rho_near,
            "rho_far": rho_far,
            "delta_near": delta_near,
            "delta_far": delta_far,
            "metric_ratio_far_near": metric_ratio_far_near,
            "pile_delta_near": pile_delta_near,
            "pile_delta_far": pile_delta_far,
            "pile_ratio_far_near": pile_ratio_far_near,
            "error": error,
        })

    df_error = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_error.to_csv(output_csv_path, index=False)

    print(f"Saved Pile relative error CSV to: {output_csv_path}")

    return df_error

def plot_pile_relative_error(df_error, output_plot_path):
    df_plot = df_error.sort_values("error")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        df_plot["metric"],
        df_plot["error"]
    )
    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot["metric"], rotation=25, ha="right")
    ax.set_ylabel("Absolute error")
    ax.set_title("Error to The Pile relative shift signal")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"Saved Pile relative error plot to: {output_plot_path}")


# ---------------
# ---- MAIN -----
# ---------------

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ------------------
    # 1. Load dataset
    # ------------------
    dataset = load_dataset_from_config(owt2_config)
    print("OWT2 loaded:", len(dataset))

    # -------------------------
    # 2. Load tokenizer/model
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.to(device)
    model.eval()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # ---------------------
    # 3. Tokenize dataset
    # ---------------------
    tokenized_dataset = utils.tokenize_dataset(
        dataset,
        tokenizer,
        owt2_config
    )

    # ------------------
    # 4. Dataloaders
    # ------------------
    deploy_dataloader = create_deploy_dataloader(
        tokenized_dataset,
        data_collator,
        owt2_config
    )

    reference_dataloader = create_reference_dataloader(
        tokenized_dataset,
        data_collator,
        owt2_config,
        seed=42
    )

    # ------------------
    # 5. Batches
    # ------------------
    fixed_batch = get_first_batch(deploy_dataloader)
    random_batches = get_n_batches(reference_dataloader, n_batches=50)

    vocab_size = tokenizer.vocab_size


    # -----------------
    # 6. Experiment 1
    # -----------------

    df = run_noise_calibration(
        fixed_batch=fixed_batch,
        reference_dataloader=reference_dataloader,
        random_batches=random_batches,
        model=model,
        device=device,
        vocab_size=tokenizer.vocab_size,
        output_csv_path="outputs/diagnostic/shift_measurement/distilgpt2/noise_calibration.csv",
        max_tokens=50000,
        pca_components_list=(5, 10, 20, 50)
    )

    plot_noise_calibration(
        df,
        output_plot_path="outputs/diagnostic/shift_measurement/distilgpt2/noise_calibration.png"
    )

    # -----------------------
    # 7. Load extra datasets
    # -----------------------
    pilecc_dataset = load_dataset_from_config(pilecc_config)
    pubmed_dataset = load_dataset_from_config(pubmed_config)

    print("PileCC loaded:", len(pilecc_dataset))
    print("PubMed loaded:", len(pubmed_dataset))

    # ---------------------------
    # 8. Tokenize extra datasets
    # ---------------------------
    pilecc_tokenized = utils.tokenize_dataset(
        pilecc_dataset,
        tokenizer,
        pilecc_config
    )

    pubmed_tokenized = utils.tokenize_dataset(
        pubmed_dataset,
        tokenizer,
        pubmed_config
    )

    # ----------------------
    # 9. Deploy dataloaders
    # ----------------------
    pilecc_deploy_dataloader = create_deploy_dataloader(
        pilecc_tokenized,
        data_collator,
        pilecc_config
    )

    pubmed_deploy_dataloader = create_deploy_dataloader(
        pubmed_tokenized,
        data_collator,
        pubmed_config
    )

    # ------------------
    # 10. Deploy batches
    # ------------------
    owt2_fixed_batch = fixed_batch

    pilecc_fixed_batch = get_first_batch(pilecc_deploy_dataloader)
    pubmed_fixed_batch = get_first_batch(pubmed_deploy_dataloader)

    deploy_batches = {
        "intra_owt2": owt2_fixed_batch,
        "near_pilecc": pilecc_fixed_batch,
        "far_pubmed": pubmed_fixed_batch,
    }

    # ------------------
    # 11. Experiment 2
    # ------------------
    df_raw = run_raw_shift_comparison(
        deploy_batches=deploy_batches,
        owt2_reference_dataloader=reference_dataloader,
        model=model,
        device=device,
        vocab_size=tokenizer.vocab_size,
        output_csv_path="outputs/diagnostic/shift_measurement/distilgpt2/raw_shift_comparison.csv",
        max_tokens=50000,
        pca_components_list=(5, 10, 20)
    )

    plot_raw_shift_comparison(
        df_raw,
        "outputs/diagnostic/shift_measurement/distilgpt2/raw_shift_comparison.png"
    )


    # ------------------
    # 12. Experiment 3
    # ------------------
    df_error = compute_pile_relative_error(
        df_raw,
        "outputs/diagnostic/shift_measurement/distilgpt2/pile_relative_error.csv"
    )

    plot_pile_relative_error(
        df_error,
        "outputs/diagnostic/shift_measurement/distilgpt2/pile_relative_error.png"
    )

    print("Done")