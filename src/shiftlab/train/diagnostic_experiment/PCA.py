import os
import math
from contextlib import ExitStack

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from matplotlib.patches import Circle
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from lambda_effect_analysis import (
    MODEL_NAME,
    get_model_path,
    compute_pca_from_gram,
)


# ============================================================
# CONFIG
# ============================================================

# Lambdas available for FreeLaw and PubMed Central.
COMMON_LAMBDAS = [
    0.0,
    0.02,
    0.05,
    0.10,
    0.20,
    0.50,
    0.70,
    1.00,
    1.50,
    2.00,
]

# ArXiv contains the common grid + additional lambdas trained
ARXIV_LAMBDAS = [
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

DATASET_LAMBDAS = {
    "ArXiv": ARXIV_LAMBDAS,
    "FreeLaw": COMMON_LAMBDAS,
    "PubMed_Central": COMMON_LAMBDAS,
}

ANALYSIS_DIR = "outputs/PCA"

GRAM_PATH = os.path.join(ANALYSIS_DIR, "global_update_gram_matrix.csv")

COORDINATES_PATH = os.path.join(ANALYSIS_DIR, "pca_coordinates.csv")

FONT_SIZE_TITLE = 22
FONT_SIZE_AXES = 18
FONT_SIZE_TICKS = 14
FONT_SIZE_LEGEND = 15
FONT_SIZE_ANNOTATION = 12

DATASET_COLORS = {
    "ArXiv": "#1f77b4",          # Matplotlib C0
    "FreeLaw": "#ff7f0e",        # Matplotlib C1
    "PubMed_Central": "#2ca02c", # Matplotlib C2
}

# ============================================================
# 3D VISUALIZATION CONFIG
# ============================================================

DISPLAY_NAMES = {
    "ArXiv": "ArXiv",
    "FreeLaw": "FreeLaw",
    "PubMed_Central": "PubMed Central",
}

# ============================================================
# MODEL INVENTORY
# ============================================================

def get_safetensors_path(dataset_name, lambd):
    """
    Return safetensors path for a dataset/lambda model.

    Reuses get_model_path() from lambda_effect_analysis.py.
    """
    model_path = get_model_path(dataset_name, lambd)
    weights_path = os.path.join(model_path, "model.safetensors")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"No model.safetensors found for "
            f"{dataset_name}, lambda={lambd}:\n"
            f"{weights_path}"
        )
    return weights_path


def build_model_entries():
    """
    Build the complete list of fine-tuned models included
    in the global PCA.

    The pretrained theta_0 is NOT included here because its
    update vector is exactly zero. It will be added explicitly
    to the Gram matrix later.
    """
    entries = []

    for dataset_name, lambdas in DATASET_LAMBDAS.items():
        for lambd in lambdas:
            entries.append(
                {
                    "dataset": dataset_name,
                    "lambda": float(lambd),
                    "model_path": get_model_path(dataset_name, lambd),
                    "weights_path": get_safetensors_path(dataset_name, lambd),
                }
            )

    return entries


def check_model_bank(entries):
    """
    Check that every model requested for the PCA exists.
    """
    print(
        "\n===== Checking model bank =====",
        flush=True,
    )
    for entry in entries:
        model_path = entry["model_path"]
        config_path = os.path.join(model_path, "config.json")
        weights_path = entry["weights_path"]

        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Missing config.json:\n"
                f"{config_path}"
            )

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Missing model weights:\n"
                f"{weights_path}"
            )

    print(
        f"All {len(entries)} fine-tuned models found.",
        flush=True,
    )

    for dataset_name, lambdas in DATASET_LAMBDAS.items():
        print(
            f"{dataset_name}: "
            f"{len(lambdas)} models "
            f"-> {len(lambdas) - 1} robustness circles",
            flush=True,
        )


# ============================================================
# GLOBAL GRAM MATRIX
# ============================================================

def compute_global_update_gram_matrix(entries):
    """
    Compute one global Gram matrix containing all datasets.

    Fine-tuning update:

        Delta_{j,lambda}
        =
        theta_{j,lambda} - theta_pretrained

    Gram matrix:

        G[i,j]
        =
        <Delta_i, Delta_j>

    The computation follows exactly the same idea as
    compute_update_gram_matrix() in lambda_effect_analysis.py,
    but is generalized to multiple datasets.

    We never concatenate all GPT-2 Medium parameters into one
    enormous vector. Parameters are processed tensor by tensor
    on CPU.
    """

    print(
        "\n"
        "============================================================\n"
        "COMPUTING GLOBAL UPDATE GRAM MATRIX\n"
        "============================================================",
        flush=True,
    )
    n_models = len(entries)

    # --------------------------------------------------------
    # PRETRAINED REFERENCE MODEL theta_0
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

    # Small matrix: n_models x n_models.
    G = torch.zeros((n_models, n_models), dtype=torch.float64)

    weight_paths = [entry["weights_path"] for entry in entries]

    # --------------------------------------------------------
    # OPEN ALL SAFETENSORS
    # --------------------------------------------------------

    with ExitStack() as stack: # ExistStack() ensures all safetensors are closed after use.
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

        # Models should contain the same tensors.
        common_keys = set(readers[0].keys())

        for reader in readers[1:]:
            common_keys &= set(reader.keys()) 

        # Keep floating-point tensors present in pretrained model.
        parameter_names = []

        for name in sorted(common_keys):
            if name not in pretrained_state:
                continue
            if not torch.is_floating_point(pretrained_state[name]):
                continue

            parameter_names.append(name)

        print(
            f"Parameter tensors used: "
            f"{len(parameter_names)}",
            flush=True,
        )

        # ----------------------------------------------------
        # PROCESS ONE PARAMETER TENSOR AT A TIME
        # ----------------------------------------------------

        for tensor_idx, name in enumerate(parameter_names):

            p_pretrained = pretrained_state[name].detach().cpu().float()
            updates = []

            for reader in readers:
                p_model = reader.get_tensor(name).detach().cpu().float()

                delta = (p_model - p_pretrained).reshape(-1)
                updates.append(delta)

            # Shape:[number of models, number of parameters in this tensor]
            D = torch.stack(updates, dim=0)

            # Add this tensor's contribution to all
            # pairwise inner products.
            G += (D @ D.T).double()
            del D
            del updates
            del p_pretrained

            if (
                tensor_idx % 20 == 0
                or tensor_idx
                == len(parameter_names) - 1
            ):

                print(
                    f"Processed tensor "
                    f"{tensor_idx + 1}/"
                    f"{len(parameter_names)}",
                    flush=True,
                )

    del pretrained_model
    del pretrained_state

    print(
        "Global Gram matrix computed.",
        flush=True,
    )
    return G


# ============================================================
# SAVE / LOAD GRAM MATRIX
# ============================================================

def get_entry_labels(entries):

    return [
        (
            f"{entry['dataset']}"
            f"__lambda_{entry['lambda']}"
        )
        for entry in entries
    ]


def save_global_gram_matrix(G, entries):

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    labels = get_entry_labels(entries)

    df = pd.DataFrame(G.numpy(), index=labels, columns=labels)
    df.to_csv(GRAM_PATH)

    print(
        f"Global Gram matrix saved to:\n"
        f"{GRAM_PATH}",
        flush=True,
    )


def load_or_compute_global_gram(entries):
    """
    Reuse the Gram matrix when the exact same model list
    has already been processed.
    """

    expected_labels = get_entry_labels(entries)

    if os.path.isfile(GRAM_PATH):
        print(
            f"\nExisting Gram matrix found:\n"
            f"{GRAM_PATH}",
            flush=True,
        )

        df = pd.read_csv(GRAM_PATH, index_col=0)
        same_rows = (list(df.index) == expected_labels)
        same_columns = (list(df.columns) == expected_labels)

        if same_rows and same_columns:

            print(
                "Configuration matches. "
                "Reusing existing Gram matrix.",
                flush=True,
            )

            return torch.tensor(df.values, dtype=torch.float64,)

        print(
            "Existing Gram matrix does not match "
            "current model configuration.\n"
            "Recomputing...",
            flush=True,
        )

    G = compute_global_update_gram_matrix(entries)
    save_global_gram_matrix(G, entries)
    return G


# ============================================================
# ADD PRETRAINED theta_0 TO PCA
# ============================================================

def augment_gram_with_pretrained(G):
    """
    Explicitly add pretrained theta_0 to the Gram matrix.

    The fine-tuning update of the pretrained model is:

        Delta_pretrained
        =
        theta_pretrained - theta_pretrained
        =
        0

    Therefore:

        <Delta_pretrained, Delta_i> = 0

    for every fine-tuned model.

    The augmented Gram matrix is thus

                    pretrained   fine-tuned
        pretrained      0            0
        fine-tuned      0            G

    This means theta_0 is genuinely included in the PCA,
    instead of being added to the figure afterwards.
    """

    n_models = G.shape[0]
    G_augmented = torch.zeros((n_models + 1, n_models + 1), dtype=torch.float64)
    G_augmented[1:, 1: ] = G

    return G_augmented


# ============================================================
# PCA
# ============================================================

def run_common_pca(G, entries):
    """
    Perform ONE PCA shared by pretrained theta_0 and every
    fine-tuned model from every source dataset.

    Reuses compute_pca_from_gram() from
    lambda_effect_analysis.py.
    """

    # Add pretrained theta_0.
    G_augmented = augment_gram_with_pretrained(G)

    # Reuse existing PCA implementation.
    coordinates, explained_var = compute_pca_from_gram(G_augmented, n_components=3)
    
    # First coordinate corresponds to pretrained theta_0.
    pretrained_coordinates = coordinates[0]

    # Remaining coordinates correspond to entries.
    model_coordinates = coordinates[1:]
    
    return (
        pretrained_coordinates,
        model_coordinates,
        explained_var,
    )


# ============================================================
# SAVE PCA COORDINATES
# ============================================================

def save_pca_coordinates(pretrained_coordinates, model_coordinates, entries):

    rows = []

    # Pretrained model theta_0.
    rows.append(
        {
            "model_type": "pretrained",
            "dataset": "Pretrained GPT-2 Medium",
            "lambda": np.nan,
            "PC1": pretrained_coordinates[0],
            "PC2": pretrained_coordinates[1],
            "PC3": pretrained_coordinates[2],
        }
    )

    # Fine-tuned models.
    for entry, coordinate in zip(entries, model_coordinates): # zip() ensures the order of entries matches the order of model_coordinates.

        rows.append(
            {
                "model_type": (
                    "ERM"
                    if entry["lambda"] == 0.0
                    else "KL-DRO"
                ),
                "dataset": entry["dataset"],
                "lambda": entry["lambda"],
                "PC1": coordinate[0],
                "PC2": coordinate[1],
                "PC3": coordinate[2],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(COORDINATES_PATH, index=False)

    print(
        f"PCA coordinates saved to:\n"
        f"{COORDINATES_PATH}",
        flush=True,
    )
    return df


# ============================================================
# DATASET COORDINATE HELPERS
# ============================================================

def get_dataset_indices(entries, dataset_name):
    """
    Return indices for one dataset, sorted by lambda.
    """

    indices = [
        i
        for i, entry in enumerate(entries)
        if entry["dataset"] == dataset_name
    ]
    indices.sort(key=lambda i: entries[i]["lambda"])

    return indices


def get_erm_coordinate(entries, model_coordinates, dataset_name):
    """
    Return PCA coordinate of lambda=0 ERM for one dataset.
    """
    for i, entry in enumerate(entries):

        if (
            entry["dataset"] == dataset_name
            and entry["lambda"] == 0.0
        ):
            return model_coordinates[i]

    raise ValueError(
        f"No lambda=0 ERM found for "
        f"{dataset_name}."
    )


# ============================================================
# 2D CIRCLES
# ============================================================

def add_2d_robustness_circles(ax, dataset_name, entries, model_coordinates, color):
    """
    Draw EXACTLY one circle for every robust model lambda > 0.

    For each lambda:

        center = PCA(theta_dataset, lambda=0)

        radius =
            ||PCA(theta_dataset, lambda)
              - PCA(theta_dataset, lambda=0)||_2

    Therefore the circle passes exactly through the
    corresponding lambda model.

    If a dataset contains N models including lambda=0,
    this function draws exactly N - 1 circles.
    """

    indices = get_dataset_indices(entries, dataset_name)
    erm_coordinate = get_erm_coordinate(entries, model_coordinates, dataset_name)

    center = (erm_coordinate[0], erm_coordinate[1])
    circle_count = 0

    for i in indices:
        lambd = entries[i]["lambda"]

        # lambda=0 is the CENTER, not a circle.
        if lambd == 0.0:
            continue
        point = model_coordinates[i]

        radius = math.sqrt((point[0] - erm_coordinate[0]) ** 2 + (point[1] - erm_coordinate[1]) ** 2)
        circle = Circle(
            center,
            radius=radius,
            fill=False,
            edgecolor=color,
            linestyle="--",
            linewidth=2.0,
            alpha=0.35,
            zorder=1,
        )
        ax.add_patch(circle)
        circle_count += 1

    expected = (len(indices) - 1)

    if circle_count != expected:
        raise RuntimeError(
            f"{dataset_name}: expected "
            f"{expected} circles but drew "
            f"{circle_count}."
        )
    print(
        f"{dataset_name}: "
        f"{circle_count} circles drawn in 2D.",
        flush=True,
    )


# ============================================================
# 2D PCA FIGURE
# ============================================================

def plot_pca_2d(pretrained_coordinates, model_coordinates, entries, explained_var):

    fig, ax = plt.subplots(figsize=(15, 11))

    # --------------------------------------------------------
    # PRETRAINED theta_0
    # --------------------------------------------------------

    ax.scatter(
        pretrained_coordinates[0],
        pretrained_coordinates[1],
        marker="*",
        s=450,
        color="black",
        edgecolor="black",
        linewidth=1.5,
        zorder=10,
        label=r"Pretrained $\theta_0$",
    )

    ax.annotate(
        r"$\theta_0$",
        (pretrained_coordinates[0], pretrained_coordinates[1]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=FONT_SIZE_ANNOTATION + 3,
        fontweight="bold",
        color="black",
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    for dataset_name in DATASET_LAMBDAS:

        color = DATASET_COLORS[dataset_name]
        indices = get_dataset_indices(entries, dataset_name)
        coords = model_coordinates[indices]
        lambdas = [entries[i]["lambda"] for i in indices]

        # ----------------------------------------------------
        # CIRCLES FIRST
        # ----------------------------------------------------

        add_2d_robustness_circles(
            ax=ax,
            dataset_name=dataset_name,
            entries=entries,
            model_coordinates=model_coordinates,
            color=color,
        )

        # ----------------------------------------------------
        # LAMBDA TRAJECTORY
        # ----------------------------------------------------

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=60,
            color=color,
            label=DISPLAY_NAMES[dataset_name],
            zorder=5,
        )

        # ----------------------------------------------------
        # ERM lambda=0
        # ----------------------------------------------------

        erm_coordinate = get_erm_coordinate(entries, model_coordinates, dataset_name)

        ax.scatter(
            erm_coordinate[0],
            erm_coordinate[1],
            s=190,
            facecolor="white",
            edgecolor=color,
            linewidth=3.0,
            zorder=8,
        )

        ax.annotate(
            r"$\lambda=0$",
            (
                erm_coordinate[0],
                erm_coordinate[1],
            ),
            xytext=(8, -23),
            textcoords="offset points",
            fontsize=FONT_SIZE_ANNOTATION + 1,
            fontweight="bold",
            color=color,
        )

        # ----------------------------------------------------
        # theta_0 -> dataset ERM
        # ----------------------------------------------------

        ax.plot(
            [pretrained_coordinates[0], erm_coordinate[0]],
            [pretrained_coordinates[1], erm_coordinate[1]],
            linestyle=":",
            color=color,
            linewidth=1.4,
            alpha=0.55,
            zorder=2,
        )

        # ----------------------------------------------------
        # LAMBDA LABELS
        # ----------------------------------------------------

        for coord, lambd in zip(coords, lambdas):
            if lambd == 0.0:
                continue
            ax.annotate(
                rf"$\lambda={lambd:g}$",
                (coord[0], coord[1]),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=FONT_SIZE_ANNOTATION,
                color=color,
                zorder=9,
            )

    # --------------------------------------------------------
    # FIGURE STYLE
    # --------------------------------------------------------

    ax.set_xlabel(
        f"PC1 "
        f"({explained_var[0] * 100:.1f}% explained variance)",
        fontsize=FONT_SIZE_AXES,
    )

    ax.set_ylabel(
        f"PC2 "
        f"({explained_var[1] * 100:.1f}% explained variance)",
        fontsize=FONT_SIZE_AXES,
    )

    ax.set_title(
        "PCA of fine-tuned models across datasets "
        "and robustness levels",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        pad=18,
    )

    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25,
    )

    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="best")

    # Very important for true circles:
    # same geometric scaling on PC1 and PC2.
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    png_path = os.path.join(ANALYSIS_DIR, "PCA_2D.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"\n2D PCA saved to:\n"
        f"{png_path}\n",
        flush=True,
    )


# ============================================================
# 3D CIRCLE GEOMETRY
# ============================================================

def choose_circle_second_direction(radial_direction, reference_direction):
    """
    Build a unit vector orthogonal to radial_direction.

    The 3D circle needs a plane.

    We use the direction from pretrained theta_0 to the dataset
    ERM as a natural reference direction, projected orthogonally
    to the radial direction.

    If these vectors are nearly collinear, a Cartesian axis is
    used as fallback.
    """

    radial_direction = (radial_direction / np.linalg.norm(radial_direction))

    # Remove component parallel to radial vector.
    v = (
        reference_direction
        - np.dot(
            reference_direction,
            radial_direction,
        )
        * radial_direction
    )

    norm_v = np.linalg.norm(v)

    # Fallback if reference vector is parallel
    # to radial direction.
    if norm_v < 1e-10:
        candidates = [
            np.array(
                [1.0, 0.0, 0.0]
            ),
            np.array(
                [0.0, 1.0, 0.0]
            ),
            np.array(
                [0.0, 0.0, 1.0]
            ),
        ]

        # Pick Cartesian axis least aligned
        # with radial direction.
        fallback = min(
            candidates,
            key=lambda axis: abs(
                np.dot(
                    axis,
                    radial_direction,
                )
            ),
        )
        v = (
            fallback
            - np.dot(
                fallback,
                radial_direction,
            )
            * radial_direction
        )

        norm_v = np.linalg.norm(v)

    return v / norm_v

def get_3d_circle_points(center, target, pretrained_coordinate, n_points=240):
    """
    Return the actual points of one robustness circle.
    Uses exactly the same geometry as add_3d_circle().
    """

    center = np.asarray(center, dtype=float)
    target = np.asarray(target, dtype=float)
    pretrained_coordinate = np.asarray(pretrained_coordinate, dtype=float)
    radial_vector = target - center
    radius = np.linalg.norm(radial_vector)

    if radius < 1e-12:
        return center[None, :]
    radial_direction = radial_vector / radius
    reference_direction = center - pretrained_coordinate

    if np.linalg.norm(reference_direction) < 1e-12:
        reference_direction = np.array([0.0, 0.0, 1.0])

    second_direction = choose_circle_second_direction(radial_direction, reference_direction)

    angles = np.linspace(0.0, 2.0 * np.pi, n_points)

    circle_points = (center[None, :] + radius * (
            np.cos(angles)[:, None]
            * radial_direction[None, :]
            + np.sin(angles)[:, None]
            * second_direction[None, :]
        )
    )

    return circle_points

# ============================================================
# INTERACTIVE 3D PCA
# ============================================================

def add_dataset_to_interactive_3d(fig, dataset_name, pretrained_coordinates, model_coordinates, entries, show_pretrained_connection=True):
    """
    Add one dataset to an interactive Plotly 3D PCA figure.

    Contains:
        - all robustness circles;
        - robust models lambda > 0;
        - ERM lambda = 0;
        - optional theta_0 -> ERM connection.

    Robust models are shown as individual points only:
    there are NO lines connecting different lambda values.
    """

    display_name = DISPLAY_NAMES[dataset_name]
    color = DATASET_COLORS[dataset_name]

    indices = get_dataset_indices(entries, dataset_name)
    coords = np.asarray(model_coordinates[indices], dtype=float)
    lambdas = [entries[i]["lambda"] for i in indices]

    erm_coordinate = np.asarray(
        get_erm_coordinate(
            entries,
            model_coordinates,
            dataset_name,
        ),
        dtype=float,
    )
    pretrained_coordinate = np.asarray(pretrained_coordinates, dtype=float)

    # --------------------------------------------------------
    # ROBUSTNESS CIRCLES
    # --------------------------------------------------------

    for i in indices:
        lambd = entries[i]["lambda"]
        if lambd == 0.0:
            continue
        circle_points = get_3d_circle_points(
            center=erm_coordinate,
            target=model_coordinates[i],
            pretrained_coordinate=pretrained_coordinate,
            n_points=300,
        )
        fig.add_trace(
            go.Scatter3d(
                x=circle_points[:, 0],
                y=circle_points[:, 1],
                z=circle_points[:, 2],
                mode="lines",
                line=dict(color=color, width=5, dash="dash"),
                opacity=0.40,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # --------------------------------------------------------
    # theta_0 -> ERM
    # --------------------------------------------------------

    if show_pretrained_connection:

        fig.add_trace(
            go.Scatter3d(
                x=[pretrained_coordinate[0], erm_coordinate[0]],
                y=[pretrained_coordinate[1], erm_coordinate[1]],
                z=[pretrained_coordinate[2], erm_coordinate[2]],
                mode="lines",
                line=dict(color=color, width=4, dash="dot"),
                opacity=0.65,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # --------------------------------------------------------
    # ROBUST MODELS lambda > 0
    # --------------------------------------------------------

    robust_mask = np.array([lambd > 0.0 for lambd in lambdas])
    robust_coords = coords[robust_mask]
    robust_lambdas = [lambd for lambd in lambdas if lambd > 0.0]

    hover_text = [(f"<b>{display_name}</b><br>" f"λ = {lambd:g}") for lambd in robust_lambdas]

    lambda_labels = [
        f"λ={lambd:g}"
        for lambd in robust_lambdas
    ]

    fig.add_trace(
        go.Scatter3d(
            x=robust_coords[:, 0],
            y=robust_coords[:, 1],
            z=robust_coords[:, 2],

            mode="markers+text",
            marker=dict(size=7, color=color),
            text=lambda_labels,
            textposition="top center",
            textfont=dict(size=13, color=color),
            name=display_name,
            customdata=robust_lambdas,

            hovertemplate=(
                f"<b>{display_name}</b><br>"
                "λ = %{customdata:g}<br>"
                "PC1 = %{x:.3f}<br>"
                "PC2 = %{y:.3f}<br>"
                "PC3 = %{z:.3f}"
                "<extra></extra>"
            ),
        )
    ) 

    # --------------------------------------------------------
    # ERM lambda = 0
    # --------------------------------------------------------

    fig.add_trace(
    go.Scatter3d(
        x=[erm_coordinate[0]],
        y=[erm_coordinate[1]],
        z=[erm_coordinate[2]],

        mode="markers+text",
        marker=dict(size=10, color="white", line=dict(color=color, width=5)),
        text=["λ=0"],
        textposition="top center",
        textfont=dict(size=14, color=color),

        name=f"{display_name} — λ=0",

        hovertemplate=(
            f"<b>{display_name}</b><br>"
            "λ = 0 (ERM)<br>"
            "PC1 = %{x:.3f}<br>"
            "PC2 = %{y:.3f}<br>"
            "PC3 = %{z:.3f}"
            "<extra></extra>"
        ),
    )
)

def add_pretrained_to_interactive_3d(fig, pretrained_coordinates):
    """
    Add pretrained theta_0 to a Plotly 3D figure.

    The pretrained model is displayed as a black diamond
    with its theta_0 label permanently visible.
    """

    pretrained_coordinate = np.asarray(pretrained_coordinates, dtype=float)

    fig.add_trace(
        go.Scatter3d(
            x=[pretrained_coordinate[0]],
            y=[pretrained_coordinate[1]],
            z=[pretrained_coordinate[2]],

            # Show both the point and its permanent label.
            mode="markers+text",

            marker=dict(size=11, color="black", symbol="diamond"),

            # Permanent label displayed on the 3D figure.
            text=["θ₀"],
            textposition="top center",
            textfont=dict(size=15, color="black"),

            name="Pretrained θ₀",

            # Information displayed when hovering over the point.
            hovertemplate=(
                "<b>Pretrained θ₀</b><br>"
                "PC1 = %{x:.3f}<br>"
                "PC2 = %{y:.3f}<br>"
                "PC3 = %{z:.3f}"
                "<extra></extra>"
            ),
        )
    )

def style_interactive_3d(fig, explained_var, title):
    """
    Common style for interactive 3D PCA figures.
    """

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=25),
        ),

        scene=dict(
            xaxis=dict(
                title=(
                    f"PC1 "
                    f"({explained_var[0] * 100:.1f}%)"
                ),
                showbackground=True,
            ),
            yaxis=dict(
                title=(
                    f"PC2 "
                    f"({explained_var[1] * 100:.1f}%)"
                ),
                showbackground=True,
            ),
            zaxis=dict(
                title=(
                    f"PC3 "
                    f"({explained_var[2] * 100:.1f}%)"
                ),
                showbackground=True,
            ),
            # Keep the true relative scale of the PCA axes.
            aspectmode="data",
        ),

        legend=dict(
            x=0.01,
            y=0.99,
            font=dict(size=14),
        ),
        width=1300,
        height=950,

        margin=dict(l=0, r=0, b=0, t=80),
    )


def plot_pca_3d_global_interactive(pretrained_coordinates, model_coordinates, entries, explained_var):
    """
    Interactive global 3D PCA containing:
        - pretrained theta_0;
        - ArXiv;
        - FreeLaw;
        - PubMed Central;
        - all robustness circles.
    """

    print(
        "\nGenerating interactive global 3D PCA...",
        flush=True,
    )

    fig = go.Figure()
    # --------------------------------------------------------
    # PRETRAINED theta_0
    # --------------------------------------------------------

    add_pretrained_to_interactive_3d(fig=fig, pretrained_coordinates=pretrained_coordinates)

    # --------------------------------------------------------
    # ALL DATASETS
    # --------------------------------------------------------

    for dataset_name in DATASET_LAMBDAS:
        add_dataset_to_interactive_3d(
            fig=fig,
            dataset_name=dataset_name,
            pretrained_coordinates=pretrained_coordinates,
            model_coordinates=model_coordinates,
            entries=entries,
            show_pretrained_connection=True,
        )

    # --------------------------------------------------------
    # STYLE
    # --------------------------------------------------------

    style_interactive_3d(
        fig=fig,
        explained_var=explained_var,
        title=(
            "3D PCA of fine-tuned models across datasets "
            "and robustness levels"
        ),
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    html_path = os.path.join(ANALYSIS_DIR, "PCA_3D_global_interactive.html")
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    print(
        f"Interactive global 3D PCA saved to:\n"
        f"{html_path}",
        flush=True,
    )


def plot_pca_3d_dataset_interactive(dataset_name, pretrained_coordinates, model_coordinates, entries, explained_var):
    """
    Interactive 3D PCA for one dataset in the SAME common
    PCA coordinate system.
    """
    display_name = DISPLAY_NAMES[dataset_name]
    print(
        f"\nGenerating interactive 3D PCA for "
        f"{display_name}...",
        flush=True,
    )
    fig = go.Figure()
    # --------------------------------------------------------
    # PRETRAINED theta_0
    # --------------------------------------------------------

    add_pretrained_to_interactive_3d(fig=fig, pretrained_coordinates=pretrained_coordinates)

    # --------------------------------------------------------
    # SELECTED DATASET
    # --------------------------------------------------------

    add_dataset_to_interactive_3d(
        fig=fig,
        dataset_name=dataset_name,
        pretrained_coordinates=pretrained_coordinates,
        model_coordinates=model_coordinates,
        entries=entries,
        show_pretrained_connection=True,
    )

    # --------------------------------------------------------
    # STYLE
    # --------------------------------------------------------

    style_interactive_3d(
        fig=fig,
        explained_var=explained_var,
        title=(
            f"3D PCA — {display_name} robustness "
            f"relative to θ₀"
        ),
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    html_path = os.path.join(ANALYSIS_DIR, f"PCA_3D_{dataset_name}_interactive.html")
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    print(
        f"Interactive 3D PCA saved to:\n"
        f"{html_path}",
        flush=True,
    )

# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    print(
        "\n"
        "============================================================\n"
        "GLOBAL PCA ANALYSIS\n"
        "============================================================",
        flush=True,
    )

    print(
        f"Pretrained reference: {MODEL_NAME}",
        flush=True,
    )

    # ========================================================
    # 1. MODEL INVENTORY
    # ========================================================

    entries = build_model_entries()
    check_model_bank(entries)

    # ========================================================
    # 2. GLOBAL GRAM MATRIX
    # ========================================================

    G = load_or_compute_global_gram(entries)

    # ========================================================
    # 3. COMMON PCA INCLUDING PRETRAINED theta_0
    # ========================================================

    (   pretrained_coordinates,
        model_coordinates,
        explained_var,
    ) = run_common_pca(
        G=G,
        entries=entries,
    )

    # ========================================================
    # 4. EXPLAINED VARIANCE
    # ========================================================

    print(
        "\n===== PCA explained variance =====\n"
        f"PC1: "
        f"{explained_var[0] * 100:.2f}%\n"
        f"PC2: "
        f"{explained_var[1] * 100:.2f}%\n"
        f"PC3: "
        f"{explained_var[2] * 100:.2f}%\n"
        f"PC1 + PC2: "
        f"{explained_var[:2].sum() * 100:.2f}%\n"
        f"PC1 + PC2 + PC3: "
        f"{explained_var.sum() * 100:.2f}%\n",
        flush=True,
    )

    # ========================================================
    # 5. SAVE COORDINATES
    # ========================================================

    save_pca_coordinates(
        pretrained_coordinates=pretrained_coordinates,
        model_coordinates=model_coordinates,
        entries=entries,
    )

    # ========================================================
    # 6. 2D FIGURE
    # ========================================================

    plot_pca_2d(
        pretrained_coordinates=pretrained_coordinates,
        model_coordinates=model_coordinates,
        entries=entries,
        explained_var=explained_var,
    )

    # ========================================================
    # 7. GLOBAL INTERACTIVE 3D FIGURE
    # ========================================================

    plot_pca_3d_global_interactive(
        pretrained_coordinates=pretrained_coordinates,
        model_coordinates=model_coordinates,
        entries=entries,
        explained_var=explained_var,
    )

    # ========================================================
    # 8. DATASET-SPECIFIC INTERACTIVE 3D FIGURES
    # ========================================================

    for dataset_name in DATASET_LAMBDAS:

        plot_pca_3d_dataset_interactive(
            dataset_name=dataset_name,
            pretrained_coordinates=pretrained_coordinates,
            model_coordinates=model_coordinates,
            entries=entries,
            explained_var=explained_var,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n"
        "============================================================\n"
        "GLOBAL PCA ANALYSIS COMPLETE\n"
        "============================================================",
        flush=True,
    )

    print(
        f"Pretrained models: 1\n"
        f"Fine-tuned models: {len(entries)}\n"
        f"Total points in PCA: {len(entries) + 1}",
        flush=True,
    )

    total_circles = sum(
        len(lambdas) - 1
        for lambdas
        in DATASET_LAMBDAS.values()
    )

    print(
        f"Total robustness circles per figure: "
        f"{total_circles}",
        flush=True,
    )

    for dataset_name, lambdas in (
        DATASET_LAMBDAS.items()
    ):

        print(
            f"  {dataset_name}: "
            f"{len(lambdas)} models -> "
            f"{len(lambdas) - 1} circles",
            flush=True,
        )

    print(
        f"\nOutputs: {ANALYSIS_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()