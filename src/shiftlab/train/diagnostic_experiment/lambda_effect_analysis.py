import os
import time
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import ExitStack
from safetensors import safe_open
from transformers import AutoModelForCausalLM

import utils
from shiftlab.data.load_datasets import load_dataset_from_subconfig


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "gpt2-medium"

SOURCE_NAME = "ArXiv"

SOURCE_CONFIG = {
        "type": "hf_text",
        "name": "timaeus/pile-arxiv",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    }

TRAINING_CONFIG = {
    "seed": 42,

    "context_length": 512,
    "max_tokens": 5_120_000,  # example: 10k sequences × 512 tokens

    "batch_size": 4,
    "gradient_accumulation_steps": 4,

    "learning_rate": 2e-5,
    "weight_decay": 0.0,

    "val_split_ratio": 0.1,

    # Same training budget for every lambda
    "max_epochs": 10,

    # KL-DRO
    "gamma": 0.7,
    "rho": 0.0,

    # Step logging
    "eval_every_optimizer_steps": 200,
    "eval_first_epoch_only": False,
}

LAMBDAS = [
    0.0,
    0.02,
    0.05,
    0.07,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.70,
    1.00,
    1.50,
    2.00,
]

INTERPOLATION_CONFIGS = [
    {
        "lambda_left": 0.10,
        "lambda_beta": 0.20,
        "lambda_right": 0.30,
        "beta": 0.5,
    },
    {
        "lambda_left": 0.30,
        "lambda_beta": 0.40,
        "lambda_right": 0.50,
        "beta": 0.5,
    },
    {
        "lambda_left": 0.40,
        "lambda_beta": 0.70,
        "lambda_right": 1.00,
        "beta": 0.5,
    },
    {
        "lambda_left": 1.00,
        "lambda_beta": 1.50,
        "lambda_right": 2.00,
        "beta": 0.5,
    },
]

OUTPUT_CONFIG = {
    "model_bank_dir": "outputs/model_bank",
    "analysis_dir": "outputs/lambda_effect_analysis",
}

CONFIG = {
    "models": {
        "name": MODEL_NAME,
    },
    "dataset": SOURCE_CONFIG,
    "training": TRAINING_CONFIG,
}

# ============================================================
# MODEL BANK
# ============================================================

def get_model_path(dataset_name, lambd):
    return os.path.join(
        OUTPUT_CONFIG["model_bank_dir"],
        dataset_name,
        f"lambda_{lambd}",
        "model",
    )

def get_training_output_dir(dataset_name, lambd):
    return os.path.join(
        OUTPUT_CONFIG["analysis_dir"],
        dataset_name,
        "training",
        f"lambda_{lambd}",
    )

def get_analysis_output_dir(dataset_name):
    return os.path.join(
        OUTPUT_CONFIG["analysis_dir"],
        dataset_name,
    )

def get_safetensors_path(lambd):
    """Return the safetensors weight file associated with one lambda."""

    model_path = get_model_path(SOURCE_NAME, lambd)

    weights_path = os.path.join(model_path, "model.safetensors")

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"No model.safetensors found for λ={lambd}: "
            f"{weights_path}"
        )

    return weights_path

def load_lambda_model(lambd, device):
    """Load one trained lambda model from the reusable model bank."""

    model_path = get_model_path(SOURCE_NAME, lambd)

    if not os.path.isfile(
        os.path.join(model_path, "config.json")
    ):
        raise FileNotFoundError(
            f"No model found for λ={lambd}: "
            f"{model_path}"
        )

    model = AutoModelForCausalLM.from_pretrained(model_path)

    model.to(device)
    model.eval()

    return model

def check_complete_model_bank():
    """Verify that every requested lambda model exists."""

    missing = []

    for lambd in LAMBDAS:
        model_path = get_model_path(SOURCE_NAME, lambd)

        model_exists = (
            os.path.isfile(
                os.path.join(
                    model_path,
                    "config.json",
                )
            )
            and (
                os.path.isfile(
                    os.path.join(
                        model_path,
                        "model.safetensors",
                    )
                )
                or os.path.isfile(
                    os.path.join(
                        model_path,
                        "pytorch_model.bin",
                    )
                )
            )
        )

        if not model_exists:
            missing.append(lambd)

    if missing:
        raise RuntimeError(
            f"Incomplete model bank. "
            f"Missing λ values: {missing}"
        )

    print(
        f"Model bank complete: "
        f"{len(LAMBDAS)} models found.",
        flush=True,
    )

def plot_model_training_curves(
    step_history,
    epoch_history,
    lambd,
    dataset_name,
    output_dir,
):
    """
    Plot training and validation CE losses for one lambda model.

    Three plots are generated:
        1. Loss vs epochs
        2. Loss vs optimizer steps
        3. Loss vs cumulative tokens seen
    """

    os.makedirs(output_dir, exist_ok=True)

    step_df = pd.DataFrame(step_history)
    epoch_df = pd.DataFrame(epoch_history)

    # 1. LOSS VS EPOCHS

    plt.figure(figsize=(9, 6))

    plt.plot(
        epoch_df["epoch"],
        epoch_df["train_loss"],
        marker="o",
        label="Train CE loss",
    )

    plt.plot(
        epoch_df["epoch"],
        epoch_df["val_loss"],
        marker="o",
        label="Validation CE loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(
        f"{dataset_name} — λ={lambd}\n"
        "Training and validation loss vs epochs"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "loss_vs_epochs.png",
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. LOSS VS OPTIMIZER STEPS
    
    plt.figure(figsize=(9, 6))

    plt.plot(
        step_df["optimizer_step"],
        step_df["train_loss"],
        marker="o",
        label="Train CE loss",
    )

    plt.plot(
        step_df["optimizer_step"],
        step_df["val_loss"],
        marker="o",
        label="Validation CE loss",
    )

    plt.xlabel("Optimizer steps")
    plt.ylabel("Cross-entropy loss")
    plt.title(
        f"{dataset_name} — λ={lambd}\n"
        "Training and validation loss vs optimizer steps"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "loss_vs_optimizer_steps.png",
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 3. LOSS VS CUMULATIVE TOKENS

    plt.figure(figsize=(9, 6))

    plt.plot(
        step_df["cumulative_tokens_seen"],
        step_df["train_loss"],
        marker="o",
        label="Train CE loss",
    )

    plt.plot(
        step_df["cumulative_tokens_seen"],
        step_df["val_loss"],
        marker="o",
        label="Validation CE loss",
    )

    plt.xlabel("Cumulative training tokens seen")
    plt.ylabel("Cross-entropy loss")
    plt.title(
        f"{dataset_name} — λ={lambd}\n"
        "Training and validation loss vs tokens"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "loss_vs_cumulative_tokens.png",
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"[λ={lambd}] Training curves saved to "
        f"{output_dir}",
        flush=True,
    )


def train_model_bank(
    train_optim_dataloader,
    train_dataloader,
    val_dataloader,
    train_step_eval_dataloader,
    device,
):
    saved_paths = {}
    metadata_rows = []

    for lambd in LAMBDAS:

        model_path = get_model_path(
            SOURCE_NAME,
            lambd,
        )

        training_output_dir = get_training_output_dir(
            SOURCE_NAME,
            lambd,
        )

        os.makedirs(training_output_dir, exist_ok=True)

        # ----------------------------------------------------
        # REUSE EXISTING MODEL
        # ----------------------------------------------------
        model_exists = (
            os.path.isfile(os.path.join(model_path, "config.json"))
            and (
                os.path.isfile(os.path.join(model_path, "model.safetensors"))
                or os.path.isfile(os.path.join(model_path, "pytorch_model.bin"))
            )
        )


        if model_exists:
            print(
                f"\n[λ={lambd}] Existing model found: "
                f"{model_path}"
            )
            print("Skipping training.", flush=True)

            saved_paths[lambd] = model_path
            continue

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        print(
            f"\n{'=' * 60}\n"
            f"Training {SOURCE_NAME} | λ={lambd}\n"
            f"{'=' * 60}",
            flush=True,
        )

        utils.set_seed(TRAINING_CONFIG["seed"])

        if train_optim_dataloader.generator is not None:
            train_optim_dataloader.generator.manual_seed(
                TRAINING_CONFIG["seed"]
            )

        model, _, _ = utils.setup_model_and_tokenizer(
            CONFIG,
            device,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        step_history, epoch_history = (
                utils.KL_DRO_train_with_step_logging(
                    model=model,
                    train_optim_dataloader=train_optim_dataloader,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    train_step_eval_dataloader=train_step_eval_dataloader,
                    optimizer=optimizer,
                    gamma=TRAINING_CONFIG["gamma"],
                    lambd=lambd,
                    rho=TRAINING_CONFIG["rho"],
                    device=device,
                    accumulation_steps=TRAINING_CONFIG[
                        "gradient_accumulation_steps"
                    ],
                    eval_every_optimizer_steps=TRAINING_CONFIG[
                        "eval_every_optimizer_steps"
                    ],
                    eval_first_epoch_only=TRAINING_CONFIG[
                        "eval_first_epoch_only"
                    ],
                    max_epochs=TRAINING_CONFIG[
                        "max_epochs"
                    ],
                )
            )

        # ----------------------------------------------------
        # SAVE MODEL FIRST
        # ----------------------------------------------------

        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)

        saved_paths[lambd] = model_path

        print(
            f"[λ={lambd}] Model saved to {model_path}",
            flush=True,
        )

        # ----------------------------------------------------
        # SAVE TRAINING HISTORIES
        # ----------------------------------------------------

        step_df = pd.DataFrame(step_history)
        epoch_df = pd.DataFrame(epoch_history)

        step_df.to_csv(
            os.path.join(training_output_dir, "step_history.csv"),
            index=False,
        )

        epoch_df.to_csv(
            os.path.join(training_output_dir, "epoch_history.csv"),
            index=False,
        )

        plot_model_training_curves(
            step_history=step_history,
            epoch_history=epoch_history,
            lambd=lambd,
            dataset_name=SOURCE_NAME,
            output_dir=training_output_dir,
        )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        last_epoch = epoch_history[-1]

        metadata_rows.append(
            {
                "dataset": SOURCE_NAME,
                "lambda": lambd,
                "model_name": MODEL_NAME,
                "model_path": model_path,

                "seed": TRAINING_CONFIG["seed"],

                "context_length": TRAINING_CONFIG[
                    "context_length"
                ],

                "max_tokens": TRAINING_CONFIG[
                    "max_tokens"
                ],

                "batch_size": TRAINING_CONFIG[
                    "batch_size"
                ],

                "gradient_accumulation_steps":
                    TRAINING_CONFIG[
                        "gradient_accumulation_steps"
                    ],

                "effective_batch_size":
                    TRAINING_CONFIG["batch_size"]
                    * TRAINING_CONFIG[
                        "gradient_accumulation_steps"
                    ],

                "max_epochs": TRAINING_CONFIG[
                    "max_epochs"
                ],

                "final_optimizer_step":
                    last_epoch["optimizer_step"],

                "final_tokens_seen":
                    last_epoch["cumulative_tokens_seen"],

                "final_train_loss":
                    last_epoch["train_loss"],

                "final_val_loss":
                    last_epoch["val_loss"],

                "final_val_ppl":
                    last_epoch["val_ppl"],
            }
        )

        # ----------------------------------------------------
        # CLEAN GPU
        # ----------------------------------------------------

        del model
        del optimizer

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return saved_paths, metadata_rows

#===========================================================
# METRICS
#===========================================================

def compute_update_gram_matrix(lambdas):
    """
    Compute the Gram matrix of fine-tuning updates

        Delta_lambda = theta_lambda - theta_pretrained

    without concatenating all model parameters.

    The computation is performed on CPU, parameter tensor by
    parameter tensor. Therefore, the PCA does not require GPU memory.

    Returns
    -------
    G : torch.Tensor
        Gram matrix of shape (n_lambdas, n_lambdas), where

            G[i, j] = <Delta_i, Delta_j>.
    """

    print(
        "\n===== Computing update Gram matrix =====",
        flush=True,
    )

    n = len(lambdas)

    # --------------------------------------------------------
    # PRETRAINED REFERENCE MODEL -- CPU ONLY
    # --------------------------------------------------------

    print(
        f"Loading pretrained reference model "
        f"{MODEL_NAME} on CPU...",
        flush=True,
    )

    pretrained_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    pretrained_model.to("cpu")
    pretrained_model.eval()
    pretrained_state = pretrained_model.state_dict()

    # Gram matrix is very small: only n_lambdas x n_lambdas.
    G = torch.zeros((n, n), dtype=torch.float64)

    weight_paths = [get_safetensors_path(lambd) for lambd in lambdas]

    # --------------------------------------------------------
    # OPEN ALL SAFETENSORS FILES
    # --------------------------------------------------------

    with ExitStack() as stack:
        readers = [
            stack.enter_context(
                safe_open(
                    path,
                    framework="pt",
                    device="cpu",
                )
            )
            for path in weight_paths
        ]

        # All lambda models should contain the same tensors.
        common_keys = set(readers[0].keys())

        for reader in readers[1:]:
            common_keys &= set(reader.keys())

        # Keep only floating-point parameters also present
        # in the pretrained model.
        parameter_names = []

        for name in sorted(common_keys):
            if name not in pretrained_state:
                continue
            if not torch.is_floating_point(pretrained_state[name]):
                continue

            parameter_names.append(name)

        print(
            f"Number of parameter tensors used: "
            f"{len(parameter_names)}",
            flush=True,
        )

        # ----------------------------------------------------
        # PROCESS ONE PARAMETER TENSOR AT A TIME
        # ----------------------------------------------------

        for tensor_idx, name in enumerate(parameter_names):
            p_pre = pretrained_state[name].detach().cpu().float()
            updates = []

            for reader in readers:
                p_lambda = reader.get_tensor(name).detach().cpu().float()
                delta = (p_lambda - p_pre).reshape(-1)
                updates.append(delta)

            # Shape:
            #
            #   n_lambdas x number_of_parameters_in_this_tensor
            #
            # This exists only for the current layer/tensor.
            D = torch.stack(updates, dim=0)

            # Add the contribution of this tensor:
            #
            #   D @ D^T
            #
            # gives all pairwise dot products for this layer.
            G += (D @ D.T).double()

            del D
            del updates
            del p_pre

            if (
                tensor_idx % 20 == 0
                or tensor_idx == len(parameter_names) - 1
            ):
                print(
                    f"Processed parameter tensor "
                    f"{tensor_idx + 1}/"
                    f"{len(parameter_names)}",
                    flush=True,
                )

    del pretrained_model
    del pretrained_state

    print(
        "Gram matrix computed.",
        flush=True,
    )
    return G

def save_gram_matrix(G, lambdas):
    """Save the Gram matrix as a CSV file."""

    output_dir = get_analysis_output_dir(SOURCE_NAME)
    os.makedirs(output_dir, exist_ok=True)

    labels = [f"lambda_{lambd}" for lambd in lambdas]

    df = pd.DataFrame(G.numpy(), index=labels, columns=labels)

    output_path = os.path.join(output_dir, "update_gram_matrix.csv")

    df.to_csv(output_path)
    print(
        f"Gram matrix saved to {output_path}",
        flush=True,
    )

def load_or_compute_gram(lambdas):
    """Load the Gram matrix if already computed, otherwise compute and save it."""
    output_dir = get_analysis_output_dir(SOURCE_NAME)
    os.makedirs(output_dir, exist_ok=True)
    gram_path = os.path.join(output_dir, "update_gram_matrix.csv")

    if os.path.isfile(gram_path):
        print(f"Loading existing Gram matrix: {gram_path}", flush=True)
        df = pd.read_csv(gram_path, index_col=0)
        G = torch.tensor(df.values, dtype=torch.float64)

        if G.shape != (len(lambdas), len(lambdas)):
            raise ValueError(
                f"Existing Gram matrix has shape {G.shape}, "
                f"expected {(len(lambdas), len(lambdas))}."
            )
        return G

    G = compute_update_gram_matrix(lambdas)
    save_gram_matrix(G, lambdas)
    return G

def compute_geometry_from_gram(G, lambdas):
    """
    Compute geometric diagnostics directly from the Gram matrix.

    Definitions
    -----------
    Delta_lambda = theta_lambda - theta_pretrained

    d_init(lambda):
        L2 distance between theta_lambda and the ERM model theta_0:

            d_init(lambda)
            = ||theta_lambda - theta_0||
            = ||Delta_lambda - Delta_0||

    d_step(lambda_i):
        L2 distance between two consecutive lambda models:

            ||theta_lambda_i - theta_lambda_{i-1}||

    cosine_with_ERM_update:
        Cosine similarity between the fine-tuning update Delta_lambda
        and the ERM fine-tuning update Delta_0.
    """
    rows = []
    erm_index = lambdas.index(0.0)
    G_00 = G[erm_index, erm_index].item()

    for i, lambd in enumerate(lambdas):

        # ----------------------------------------------------
        # DISTANCE TO ERM MODEL theta_0
        #
        # ||Delta_i - Delta_0||²
        # = <Delta_i, Delta_i>
        # + <Delta_0, Delta_0>
        # - 2 <Delta_i, Delta_0>
        # ----------------------------------------------------

        squared_d_init = G[i, i].item() + G_00 - 2.0 * G[i, erm_index].item()
        d_init_value = math.sqrt(max(squared_d_init, 0.0))

        # ----------------------------------------------------
        # DISTANCE TO PREVIOUS LAMBDA MODEL
        # ----------------------------------------------------

        if i == 0:
            d_step_value = float("nan")

        else:
            squared_d_step = G[i, i].item() + G[i - 1, i - 1].item() - 2.0 * G[i, i - 1].item()
            d_step_value = math.sqrt(max(squared_d_step, 0.0))

        # ----------------------------------------------------
        # COSINE SIMILARITY WITH ERM FINE-TUNING UPDATE
        #
        # cos(Delta_lambda, Delta_0)
        # ----------------------------------------------------

        denominator = math.sqrt(max(G[i, i].item(), 0.0) * max(G_00, 0.0))

        if denominator > 0.0:
            cosine_erm = G[i, erm_index].item() / denominator
        else:
            cosine_erm = float("nan")

        rows.append(
            {
                "lambda": lambd,
                "d_init": d_init_value,
                "d_step_from_previous": d_step_value,
                "cosine_with_ERM_update": cosine_erm,
            }
        )

    return pd.DataFrame(rows)

def compute_pca_from_gram(G, n_components=2):
    """
    Compute exact PCA coordinates from an uncentered Gram matrix.

    Parameters
    ----------
    G:
        Gram matrix G_ij = <Delta_i, Delta_j>.

    n_components:
        Number of PCA dimensions to return.

    Returns
    -------
    coordinates:
        PCA coordinates of shape
        (n_lambdas, n_components).

    explained_variance_ratio:
        Fraction of total PCA variance explained by each
        returned component.
    """
    G = G.to(dtype=torch.float64)
    n = G.shape[0]

    # --------------------------------------------------------
    # CENTER THE GRAM MATRIX
    # --------------------------------------------------------
    identity = torch.eye(n, dtype=torch.float64)
    ones = torch.ones((n, n), dtype=torch.float64) / n
    H = identity - ones
    G_centered = H @ G @ H

    # Force exact numerical symmetry.
    G_centered = (G_centered + G_centered.T) / 2.0

    # --------------------------------------------------------
    # EIGENDECOMPOSITION
    # --------------------------------------------------------

    eigenvalues, eigenvectors = torch.linalg.eigh(G_centered)

    # torch.linalg.eigh returns eigenvalues
    # from smallest to largest.
    order = torch.argsort(eigenvalues, descending=True)

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Very small negative eigenvalues can appear because
    # of floating-point numerical errors.
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    total_variance = eigenvalues.sum()

    if total_variance <= 0:
        raise ValueError(
            "PCA variance is zero."
        )

    explained_variance_ratio = eigenvalues / total_variance

    # --------------------------------------------------------
    # PCA COORDINATES
    # --------------------------------------------------------

    selected_eigenvalues = eigenvalues[:n_components]

    selected_eigenvectors = eigenvectors[:, :n_components]
    
    coordinates = (
        selected_eigenvectors
        * torch.sqrt(
            selected_eigenvalues
        ).unsqueeze(0)
    )

    return (
        coordinates.numpy(),
        explained_variance_ratio[
            :n_components
        ].numpy(),
    )

def gram_model_distance(G, lambdas, lambda_a, lambda_b):
    """
    Compute ||theta_lambda_a - theta_lambda_b|| from the Gram matrix.
    """

    i = lambdas.index(lambda_a)
    j = lambdas.index(lambda_b)

    squared_distance = (
        G[i, i].item()
        + G[j, j].item()
        - 2.0 * G[i, j].item()
    )

    return math.sqrt(max(squared_distance, 0.0))

def gram_curvature_ratio(G, lambdas, lambda_left, lambda_beta, lambda_right):
    """
    Compute the curvature ratio

        ||theta_L - 2 theta_beta + theta_R||
        -------------------------------------
               ||theta_L - theta_R||

    directly from the Gram matrix.

    This definition is appropriate when lambda_beta is the midpoint
    between lambda_left and lambda_right (beta = 0.5).
    """

    i_left = lambdas.index(lambda_left)
    i_beta = lambdas.index(lambda_beta)
    i_right = lambdas.index(lambda_right)

    # Numerator squared:
    # ||Delta_L - 2 Delta_beta + Delta_R||^2
    numerator_sq = (
        G[i_left, i_left].item()
        + 4.0 * G[i_beta, i_beta].item()
        + G[i_right, i_right].item()
        - 4.0 * G[i_left, i_beta].item()
        + 2.0 * G[i_left, i_right].item()
        - 4.0 * G[i_beta, i_right].item()
    )

    # Denominator squared:
    # ||Delta_L - Delta_R||^2
    denominator_sq = (
        G[i_left, i_left].item()
        + G[i_right, i_right].item()
        - 2.0 * G[i_left, i_right].item()
    )

    numerator = math.sqrt(max(numerator_sq, 0.0))
    denominator = math.sqrt(max(denominator_sq, 0.0))

    return numerator / (denominator + 1e-12)

def interpolate_models(model_left, model_right, beta, config, device):
    """ Create the interpolated model. """
    model_interp, _, _ = utils.setup_model_and_tokenizer(config, device)
    state_interp = {}

    state_left = model_left.state_dict()
    state_right = model_right.state_dict()
    for name in state_left.keys():
        if name in state_right:
            p_left = state_left[name]
            p_right = state_right[name]
            if torch.is_floating_point(p_left):
                p_interp = (1 - beta) * p_left + beta * p_right
                
            else:
                p_interp = p_left
            state_interp[name] = p_interp
            
    model_interp.load_state_dict(state_interp)
    model_interp.to(device)
    return model_interp

def D_pred(model_i, model_j, probe_dataloader, device):
    """Compute predictive KL divergence between two models."""
    model_i.eval()
    model_j.eval()
    total_divergence = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in probe_dataloader:
            batch = utils.move_batch_to_device(batch, device)
            logits_i = model_i(**batch).logits
            logits_j = model_j(**batch).logits

            probs_i = torch.nn.functional.softmax(logits_i, dim=-1)
            log_probs_j = torch.nn.functional.log_softmax(logits_j, dim=-1)

            kl_ij = torch.nn.functional.kl_div(
                log_probs_j, probs_i, reduction="batchmean"
            )

            batch_size = batch["input_ids"].size(0)
            total_divergence += kl_ij.item() * batch_size
            total_examples += batch_size

    return total_divergence / total_examples

def compute_interpolation_metrics(
    G,
    lambdas,
    geometry_df,
    lambda_left,
    lambda_beta,
    lambda_right,
    beta,
    model_left,
    model_trained,
    model_right,
    model_interp,
    probe_dataloader,
    device,
):
    """
    Compute geometric and predictive diagnostics for one interpolation.

    Geometry is computed from the Gram matrix whenever possible.

    theta_trained = model explicitly trained at lambda_beta
    theta_interp  = (1-beta) theta_left + beta theta_right
    """
    # 1. GEOMETRY FROM GRAM MATRIX

    # Distance between trained lambda_beta model and ERM lambda=0.
    d_init_value = geometry_df.loc[
        geometry_df["lambda"] == lambda_beta,
        "d_init",
    ].iloc[0]

    # Distances between trained models.
    d_endpoints = gram_model_distance(G, lambdas, lambda_left, lambda_right)
    d_left_trained = gram_model_distance(G, lambdas, lambda_left, lambda_beta)
    d_right_trained = gram_model_distance(G, lambdas, lambda_beta, lambda_right)

    # --------------------------------------------------------
    # Distance trained model <-> interpolated model
    #
    # theta_interp =
    #     (1-beta) theta_left + beta theta_right
    #
    # Everything can be obtained from Gram.
    # --------------------------------------------------------

    i_left = lambdas.index(lambda_left)
    i_beta = lambdas.index(lambda_beta)
    i_right = lambdas.index(lambda_right)

    a = 1.0 - beta
    b = beta

    # ||a Delta_L + b Delta_R - Delta_beta||^2
    d_interp_sq = (
        a**2 * G[i_left, i_left].item()
        + b**2 * G[i_right, i_right].item()
        + G[i_beta, i_beta].item()
        + 2.0 * a * b * G[i_left, i_right].item()
        - 2.0 * a * G[i_left, i_beta].item()
        - 2.0 * b * G[i_right, i_beta].item()
    )

    d_interp = math.sqrt(max(d_interp_sq, 0.0))
    d_rel = d_interp / (d_init_value + 1e-12)

    endpoint_ratio = d_interp / (d_endpoints + 1e-12)

    # --------------------------------------------------------
    # Curvature
    # --------------------------------------------------------

    if abs(beta - 0.5) < 1e-12:
        c_curvature = gram_curvature_ratio(
            G,
            lambdas,
            lambda_left,
            lambda_beta,
            lambda_right,
        )
    else:
        c_curvature = float("nan")

    # --------------------------------------------------------
    # Triangle ratio
    # --------------------------------------------------------

    r_triangle = (d_left_trained + d_right_trained) / (d_endpoints + 1e-12)

    # ========================================================
    # 2. PREDICTIVE DISTANCE
    # ========================================================

    D_pred_value = D_pred(
        model_trained,
        model_interp,
        probe_dataloader,
        device,
    )

    # ========================================================
    # 3. PERFORMANCE
    # ========================================================

    trained_loss, trained_ppl, _, _ = utils.evaluation(
        model_trained,
        probe_dataloader,
        device,
    )

    interp_loss, interp_ppl, _, _ = utils.evaluation(
        model_interp,
        probe_dataloader,
        device,
    )

    loss_left, left_ppl, _, _ = utils.evaluation(
        model_left,
        probe_dataloader,
        device,
    )

    loss_right, right_ppl, _, _ = utils.evaluation(
        model_right,
        probe_dataloader,
        device,
    )

    # Difference trained vs interpolation.
    delta_loss = abs(
        trained_loss - interp_loss
    )

    delta_ppl = abs(
        trained_ppl - interp_ppl
    )

    # Compare interpolation with best endpoint.
    best_endpoint_loss = min(
        loss_left,
        loss_right,
    )

    delta_interp_best_endpoint = (
        interp_loss - best_endpoint_loss
    )
    interp_minus_trained = interp_loss - trained_loss

    # ========================================================
    # RESULTS
    # ========================================================

    return {
        "lambda_left": lambda_left,
        "lambda_beta": lambda_beta,
        "lambda_right": lambda_right,
        "beta": beta,

        # Geometry
        "d_interp": d_interp,
        "d_init": d_init_value,
        "d_rel": d_rel,

        "d_endpoints": d_endpoints,
        "d_left_trained": d_left_trained,
        "d_right_trained": d_right_trained,

        "endpoint_ratio": endpoint_ratio,
        "c_curvature": c_curvature,
        "r_triangle": r_triangle,

        # Predictive divergence
        "D_pred": D_pred_value,

        # Loss
        "loss_left": loss_left,
        "loss_trained": trained_loss,
        "loss_interp": interp_loss,
        "loss_right": loss_right,

        # Perplexity
        "ppl_left": left_ppl,
        "ppl_trained": trained_ppl,
        "ppl_interp": interp_ppl,
        "ppl_right": right_ppl,

        # Performance differences
        "delta_loss": delta_loss,
        "delta_ppl": delta_ppl,

        "best_endpoint_loss": best_endpoint_loss,
        "delta_interp_best_endpoint": delta_interp_best_endpoint,
        "interp_minus_trained": interp_minus_trained,
    }

#===========================================================
# PLOTS
#==========================================================

def plot_pca_updates(X_2D, lambdas, explained_var):
    """ Plot the PCA projection of updates. """
    output_dir = get_analysis_output_dir(SOURCE_NAME)  
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 7))

    plt.plot(X_2D[:,0], X_2D[:,1], marker='o')
    for i, lambd in enumerate(lambdas):
        plt.annotate(f"λ={lambd}", (X_2D[i, 0], X_2D[i, 1]))
    plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
    plt.title("PCA of updates vectors")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/PCA_updates.png")
    plt.close()

def plot_lambda_geometry(geometry_df):
    """Plot d_init, d_step and cosine similarity as separate figures."""
    output_dir = get_analysis_output_dir(SOURCE_NAME)
    os.makedirs(output_dir, exist_ok=True)

    # 1. d_init vs lambda
    plt.figure(figsize=(8, 6))
    plt.plot(geometry_df["lambda"], geometry_df["d_init"], marker="o")
    plt.xlabel("λ")
    plt.ylabel(r"$d_{\mathrm{init}}=\|\theta_\lambda-\theta_0\|_2$")
    plt.title("Distance from ERM vs λ")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "d_init_vs_lambda.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. d_step vs lambda
    step_df = geometry_df.dropna(subset=["d_step_from_previous"])
    plt.figure(figsize=(8, 6))
    plt.plot(step_df["lambda"], step_df["d_step_from_previous"], marker="o")
    plt.xlabel("λ")
    plt.ylabel(r"$d_{\mathrm{step}}$")
    plt.title("Distance between consecutive λ models")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "d_step_vs_lambda.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. cosine similarity vs lambda
    plt.figure(figsize=(8, 6))
    plt.plot(geometry_df["lambda"], geometry_df["cosine_with_ERM_update"], marker="o")
    plt.xlabel("λ")
    plt.ylabel("Cosine similarity")
    plt.title("Cosine similarity with ERM update vs λ")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cosine_similarity_vs_lambda.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Global geometry plots saved to {output_dir}", flush=True)

def plot_interpolation_geometry(df):
    output_dir = get_analysis_output_dir(SOURCE_NAME)
    os.makedirs(output_dir, exist_ok=True)

    labels = [f"{r.lambda_left}-{r.lambda_beta}-{r.lambda_right}" for r in df.itertuples()]
    x = range(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = [
        ("d_rel", r"$d_{\mathrm{interp}}/d_{\mathrm{init}}$", "Relative interpolation error"),
        ("endpoint_ratio", r"$d_{\mathrm{interp}}/d_{\mathrm{endpoints}}$", "Relative to endpoint distance"),
        ("c_curvature", r"$c_{\mathrm{curvature}}$", "Curvature ratio"),
        ("r_triangle", r"$r_\triangle$", "Triangle ratio"),
    ]

    for ax, (column, ylabel, title) in zip(axes.flat, metrics):
        ax.bar(x, df[column])
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_xlabel(r"$\lambda_L-\lambda_\beta-\lambda_R$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y")

    axes[1, 1].axhline(1.0, linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "interpolation_geometry.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_interpolation_performance(df):
    output_dir = get_analysis_output_dir(SOURCE_NAME)
    os.makedirs(output_dir, exist_ok=True)

    labels = [f"{r.lambda_left}-{r.lambda_beta}-{r.lambda_right}" for r in df.itertuples()]
    x = list(range(len(df)))
    width = 0.2

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar([i - 1.5*width for i in x], df["loss_left"], width, label="Left")
    axes[0].bar([i - 0.5*width for i in x], df["loss_trained"], width, label="Trained midpoint")
    axes[0].bar([i + 0.5*width for i in x], df["loss_interp"], width, label="Interpolated")
    axes[0].bar([i + 1.5*width for i in x], df["loss_right"], width, label="Right")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Model performance")
    axes[0].legend()

    axes[1].bar(x, df["interp_minus_trained"])
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].set_ylabel(r"$L_{\mathrm{interp}}-L_{\mathrm{trained}}$")
    axes[1].set_title("Interpolation vs trained midpoint")

    axes[2].bar(x, df["delta_interp_best_endpoint"])
    axes[2].axhline(0.0, linestyle="--", linewidth=1)
    axes[2].set_ylabel(r"$L_{\mathrm{interp}}-\min(L_L,L_R)$")
    axes[2].set_title("Interpolation vs best endpoint")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_xlabel(r"$\lambda_L-\lambda_\beta-\lambda_R$")
        ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "interpolation_performance.png"), dpi=300, bbox_inches="tight")
    plt.close()

#===========================================================
# MAIN
#===========================================================


def lambda_bank():

    start_time = time.time()

    utils.set_seed(TRAINING_CONFIG["seed"])

    device = utils.get_device()

    print(f"Device: {device}", flush=True)

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}",
            flush=True,
        )

        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # TOKENIZER / COLLATOR
    # --------------------------------------------------------

    base_model, tokenizer, data_collator = (
        utils.setup_model_and_tokenizer(
            CONFIG,
            device,
        )
    )

    del base_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # LOAD SOURCE DATASET
    # --------------------------------------------------------

    print(
        f"\nLoading source dataset: {SOURCE_NAME}",
        flush=True,
    )

    raw_dataset = load_dataset_from_subconfig(
        SOURCE_CONFIG, TRAINING_CONFIG
    )

    # --------------------------------------------------------
    # TOKEN BUDGET
    # --------------------------------------------------------

    tokenized_dataset, dataset_stats = (
        utils.tokenize_and_group_with_token_budget(
            dataset=raw_dataset,
            tokenizer=tokenizer,
            config=CONFIG,
        )
    )

    # --------------------------------------------------------
    # TRAIN / VAL DATALOADERS
    # --------------------------------------------------------

    (
        train_optim_dataloader,
        train_dataloader,
        val_dataloader,
        train_step_eval_dataloader,
    ) = utils.create_training_dataloaders(
        tokenized_dataset=tokenized_dataset,
        data_collator=data_collator,
        training_config=TRAINING_CONFIG,
    )

    print(
        "\n===== Dataset split =====\n"
        f"Train sequences: {len(train_dataloader.dataset):,}\n"
        f"Val sequences:   {len(val_dataloader.dataset):,}\n"
        f"Train tokens:    "
        f"{len(train_dataloader.dataset) * TRAINING_CONFIG['context_length']:,}\n"
        f"Val tokens:      "
        f"{len(val_dataloader.dataset) * TRAINING_CONFIG['context_length']:,}\n",
        flush=True,
    )

    # --------------------------------------------------------
    # MODEL BANK
    # --------------------------------------------------------

    saved_paths, metadata_rows = train_model_bank(
        train_optim_dataloader=train_optim_dataloader,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_step_eval_dataloader=train_step_eval_dataloader,
        device=device,
    )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata_path = os.path.join(
        OUTPUT_CONFIG["model_bank_dir"],
        "model_bank_metadata.csv",
    )

    os.makedirs(
        OUTPUT_CONFIG["model_bank_dir"],
        exist_ok=True,
    )

    if metadata_rows:
        metadata_df = pd.DataFrame(metadata_rows)

        metadata_df.to_csv(
            metadata_path,
            index=False,
        )

        print(
            f"Model bank metadata saved to: "
            f"{metadata_path}",
            flush=True,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n===== Model bank =====")

    for lambd, path in saved_paths.items():
        print(
            f"λ={lambd}: {path}"
        )

    elapsed = time.time() - start_time

    print(
        f"\nTotal time: {elapsed / 60:.2f} minutes",
        flush=True,
    )

    if device.type == "cuda":

        peak = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"Peak GPU memory: {peak:.2f} GB",
            flush=True,
        )

def run_interpolation_analysis(G, geometry_df, probe_dataloader, device):
    """Run all interpolation configurations."""
    rows = []
    for config_interp in INTERPOLATION_CONFIGS:

        lambda_left = config_interp["lambda_left"]
        lambda_beta = config_interp["lambda_beta"]
        lambda_right = config_interp["lambda_right"]
        beta = config_interp["beta"]

        print(
            "\n"
            "============================================================\n"
            f"INTERPOLATION: "
            f"{lambda_left} -> {lambda_beta} -> {lambda_right}\n"
            "============================================================",
            flush=True,
        )

        # ----------------------------------------------------
        # LOAD TRAINED MODELS
        # ----------------------------------------------------

        model_left = load_lambda_model(lambda_left, device)
        model_trained = load_lambda_model(lambda_beta, device)
        model_right = load_lambda_model(lambda_right, device)
        
        # ----------------------------------------------------
        # BUILD INTERPOLATED MODEL
        # ----------------------------------------------------

        model_interp = interpolate_models(
            model_left,
            model_right,
            beta,
            CONFIG,
            device,
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        results = compute_interpolation_metrics(
            G=G,
            lambdas=LAMBDAS,
            geometry_df=geometry_df,
            lambda_left=lambda_left,
            lambda_beta=lambda_beta,
            lambda_right=lambda_right,
            beta=beta,
            model_left=model_left,
            model_trained=model_trained,
            model_right=model_right,
            model_interp=model_interp,
            probe_dataloader=probe_dataloader,
            device=device,
        )

        rows.append(results)

        # ----------------------------------------------------
        # GPU CLEANUP
        # ----------------------------------------------------

        del model_left
        del model_trained
        del model_right
        del model_interp

        if device.type == "cuda":
            torch.cuda.empty_cache()

    interpolation_df = pd.DataFrame(rows)

    output_dir = os.path.join(get_analysis_output_dir(SOURCE_NAME), "interpolation")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "interpolation_results.csv")
    interpolation_df.to_csv(output_path, index=False)

    print(
        f"\nInterpolation results saved to: "
        f"{output_path}",
        flush=True,
    )

    return interpolation_df

def lambda_analysis():
    start_time = time.time()
    utils.set_seed(TRAINING_CONFIG["seed"])
    device = utils.get_device()

    print(
        f"Analysis device: {device}",
        flush=True,
    )

    # ========================================================
    # CHECK MODEL BANK
    # ========================================================

    check_complete_model_bank()

    # ========================================================
    # GRAM MATRIX
    # ========================================================

    G = load_or_compute_gram(LAMBDAS)

    # ========================================================
    # GLOBAL GEOMETRY
    # ========================================================

    geometry_df = compute_geometry_from_gram(G, LAMBDAS)
    geometry_dir = os.path.join(get_analysis_output_dir(SOURCE_NAME), "geometry")
    os.makedirs(geometry_dir, exist_ok=True)

    geometry_df.to_csv(
        os.path.join(
            geometry_dir,
            "lambda_geometry.csv",
        ),
        index=False,
    )
    plot_lambda_geometry(geometry_df)

    # ========================================================
    # PCA
    # ========================================================

    X_2D, explained_var = compute_pca_from_gram(G, n_components=2)
    plot_pca_updates(X_2D, LAMBDAS, explained_var)

    print(
        "\n===== PCA explained variance =====\n"
        f"PC1: "
        f"{explained_var[0] * 100:.2f}%\n"
        f"PC2: "
        f"{explained_var[1] * 100:.2f}%\n"
        f"PC1 + PC2: "
        f"{explained_var.sum() * 100:.2f}%\n",
        flush=True,
    )

    # ========================================================
    # RECREATE EXACT VALIDATION SPLIT
    # ========================================================

    print(
        "\nReconstructing held-out validation data...",
        flush=True,
    )

    # We only need tokenizer + collator.
    base_model, tokenizer, data_collator = utils.setup_model_and_tokenizer(CONFIG, device)
    del base_model

    if device.type == "cuda":
        torch.cuda.empty_cache()
    raw_dataset = load_dataset_from_subconfig(SOURCE_CONFIG, TRAINING_CONFIG)

    tokenized_dataset, _ = (
utils.tokenize_and_group_with_token_budget(
            dataset=raw_dataset,
            tokenizer=tokenizer,
            config=CONFIG,
        )
    )
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

    # Use the complete held-out validation split as probe data.
    probe_dataloader = val_dataloader

    # ========================================================
    # INTERPOLATION ANALYSIS
    # ========================================================

    interpolation_df = run_interpolation_analysis(
        G=G,
        geometry_df=geometry_df,
        probe_dataloader=probe_dataloader,
        device=device,
    )

    # ========================================================
    # INTERPOLATION PLOTS
    # ========================================================

    plot_interpolation_geometry(interpolation_df)
    plot_interpolation_performance(interpolation_df)

    # ========================================================
    # SUMMARY
    # ========================================================

    elapsed = time.time() - start_time

    print(
        "\n"
        "============================================================\n"
        "LAMBDA EFFECT ANALYSIS COMPLETE\n"
        "============================================================\n"
        f"Dataset: {SOURCE_NAME}\n"
        f"Models: {len(LAMBDAS)}\n"
        f"Interpolations: "
        f"{len(INTERPOLATION_CONFIGS)}\n"
        f"Total analysis time: "
        f"{elapsed / 60:.2f} minutes\n",
        flush=True,
    )

if __name__ == "__main__":
    #lambda_bank()
    lambda_analysis()