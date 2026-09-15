import os
import time
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
import torch

from lambda_effect_analysis import MODEL_NAME, compute_pca_from_gram
from hierarchical_routing import build_candidate_state_dicts, batch_loss_at_weights, load_pretrained_checkpoint
from PCA import augment_gram_with_pretrained


ANALYSIS_DIR = "outputs/routing_PCA"
GRID_STEP = 0.05

FONT_SIZE_AXES = 22
FONT_SIZE_TICKS = 18
FONT_SIZE_LEGEND = 19
FONT_SIZE_ANNOTATION = 17

EDGE_LINEWIDTH = 2.2
TRAJECTORY_LINEWIDTH = 2.6
BEST_TRAJECTORY_LINEWIDTH = 4.0
LOSS_LINEWIDTH = 2.2
BEST_LOSS_LINEWIDTH = 3.6
REFERENCE_LINEWIDTH = 2.5


# ============================================================
# MODELS
# ============================================================

def load_routing_candidates(info, pretrained_model_name, bank_repo_id=None, pretrained_subfolder=None):
    """Reload the exact H candidates selected by Hierarchical Routing."""
    if "selected_candidates" not in info:
        raise ValueError("Routing info does not contain selected_candidates.")

    selected_candidates_df = pd.DataFrame(info["selected_candidates"])
    return build_candidate_state_dicts(
        selected_candidates_df=selected_candidates_df,
        pretrained_model_name=pretrained_model_name,
        device=torch.device("cpu"),
        bank_repo_id=bank_repo_id,
        pretrained_subfolder=pretrained_subfolder,
    )


def add_pretrained_reference(
    candidate_state_dicts,
    pretrained_model_name,
    bank_repo_id=None,
    pretrained_subfolder=None,
):
    """Add the exact theta_0 used by routing as PCA reference if needed."""
    if "theta_0" in candidate_state_dicts:
        return candidate_state_dicts

    model = load_pretrained_checkpoint(
        pretrained_model_name=pretrained_model_name,
        bank_repo_id=bank_repo_id,
        pretrained_subfolder=pretrained_subfolder,
    )
    pca_state_dicts = dict(candidate_state_dicts)
    pca_state_dicts["theta_0"] = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    del model
    return pca_state_dicts


# ============================================================
# PCA
# ============================================================

def compute_candidate_update_gram(candidate_state_dicts):
    """Compute Gram matrix of candidate updates relative to theta_0."""
    names = list(candidate_state_dicts.keys())
    if "theta_0" not in names:
        raise ValueError("PCA state dicts must contain theta_0 as reference.")

    pretrained_state = candidate_state_dicts["theta_0"]
    finetuned_names = [name for name in names if name != "theta_0"]

    if not finetuned_names:
        raise ValueError("PCA requires at least one fine-tuned model besides theta_0.")

    G = torch.zeros(
        (len(finetuned_names), len(finetuned_names)),
        dtype=torch.float64,
    )

    parameter_names = [
        name
        for name in pretrained_state
        if torch.is_floating_point(pretrained_state[name])
        and all(name in candidate_state_dicts[m] for m in finetuned_names)
    ]

    for tensor_idx, name in enumerate(parameter_names):
        p0 = pretrained_state[name].detach().cpu().float()

        updates = [
            (
                candidate_state_dicts[m][name]
                .detach()
                .cpu()
                .float()
                - p0
            ).reshape(-1)
            for m in finetuned_names
        ]

        D = torch.stack(updates, dim=0)
        G += (D @ D.T).double()

        del D, updates, p0

        if tensor_idx % 20 == 0 or tensor_idx == len(parameter_names) - 1:
            print(
                f"Processed tensor {tensor_idx + 1}/{len(parameter_names)}",
                flush=True,
            )

    return G, finetuned_names


def compute_candidate_pca(candidate_state_dicts):
    """Compute local 2D PCA using theta_0 as reference."""
    G, finetuned_names = compute_candidate_update_gram(candidate_state_dicts)

    coordinates, explained_var = compute_pca_from_gram(
        augment_gram_with_pretrained(G),
        n_components=2,
    )

    names = ["theta_0"] + finetuned_names

    return {
        name: coordinates[i]
        for i, name in enumerate(names)
    }, explained_var


def compute_trajectory_coordinates(info, coordinate_dict):
    """Convert EG trajectories into PCA coordinates."""
    coords = np.stack(
        [coordinate_dict[name] for name in info["candidate_names"]],
        axis=0,
    )

    rows = []

    for point in info["optimization_trajectories"]:
        position = np.asarray(point["weights"], dtype=float) @ coords

        rows.append({
            "start_id": point["start_id"],
            "start_name": point["start_name"],
            "iteration": point["iteration"],
            "loss": point["loss"],
            "PC1": position[0],
            "PC2": position[1],
        })

    return pd.DataFrame(rows)


def get_selected_solution_coordinate(info, coordinate_dict, trajectory_df):
    """Return PCA coordinate of selected hierarchical solution."""
    if info["best_is_vertex"]:
        return np.asarray(
            coordinate_dict[info["best_vertex_name"]],
            dtype=float,
        )

    points = trajectory_df[
        (trajectory_df["start_id"] == info["best_start_id"])
        & (trajectory_df["iteration"] == info["best_iteration"])
    ]

    if len(points) != 1:
        raise RuntimeError(
            "Could not identify the selected EG solution."
        )

    row = points.iloc[0]

    return np.array(
        [row["PC1"], row["PC2"]],
        dtype=float,
    )


def save_trajectory_coordinates(trajectory_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(
        output_dir,
        "hierarchical_pca_trajectory.csv",
    )

    trajectory_df.to_csv(
        path,
        index=False,
    )

    print(
        f"PCA trajectory saved to:\n{path}",
        flush=True,
    )

    return path


# ============================================================
# PLOT HELPERS
# ============================================================

def get_trajectory_colors(trajectory_df):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    starts = (
        trajectory_df[
            ["start_id", "start_name"]
        ]
        .drop_duplicates()
        .sort_values("start_id")
    )

    return {
        row["start_name"]: colors[i % len(colors)]
        for i, (_, row) in enumerate(starts.iterrows())
    }


def set_pca_limits(
    ax,
    coordinate_dict,
    right_margin=0.30,
    left_margin=0.16,
    bottom_margin=0.18,
    top_margin=0.18,
):
    coords = np.stack(
        list(coordinate_dict.values()),
        axis=0,
    )

    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()

    dx = max(x_max - x_min, 1.0)
    dy = max(y_max - y_min, 1.0)

    ax.set_xlim(
        x_min - left_margin * dx,
        x_max + right_margin * dx,
    )

    ax.set_ylim(
        y_min - bottom_margin * dy,
        y_max + top_margin * dy,
    )


def plot_candidate_edges(
    ax,
    info,
    coordinate_dict,
    color="gray",
    alpha=0.35,
    linestyle="--",
):
    names = info["candidate_names"]

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c1 = coordinate_dict[names[i]]
            c2 = coordinate_dict[names[j]]

            ax.plot(
                [c1[0], c2[0]],
                [c1[1], c2[1]],
                linestyle=linestyle,
                linewidth=EDGE_LINEWIDTH,
                alpha=alpha,
                color=color,
                zorder=1,
            )


def plot_candidate_vertices(ax, info, coordinate_dict):
    candidate_coords = np.stack(
        [
            coordinate_dict[name]
            for name in info["candidate_names"]
        ],
        axis=0,
    )

    x_min = candidate_coords[:, 0].min()
    x_max = candidate_coords[:, 0].max()
    x_span = max(x_max - x_min, 1.0)

    for name in info["candidate_names"]:
        coord = coordinate_dict[name]

        if name == "theta_0":
            ax.scatter(
                coord[0],
                coord[1],
                marker="D",
                s=190,
                color="black",
                zorder=8,
            )

            label = r"$\theta_0$"

        else:
            ax.scatter(
                coord[0],
                coord[1],
                s=170,
                edgecolor="black",
                linewidth=1.0,
                zorder=8,
            )

            label = name

            if name in info.get("selected_lambdas", {}):
                label = (
                    f"{name}, "
                    f"λ={info['selected_lambdas'][name]:g}"
                )

        # Keep annotations inside the plotting area.
        #
        # In particular, put the right-most candidate below
        # and to the left of its vertex. This prevents labels
        # such as "ArXiv, λ=..." from running underneath the
        # legend in the loss-landscape figure.
        if coord[0] >= x_max - 0.08 * x_span:
            xytext = (-18, -14)
            ha = "right"
            va = "top"

        elif coord[0] <= x_min + 0.08 * x_span:
            xytext = (12, 12)
            ha = "left"
            va = "bottom"

        else:
            xytext = (10, 10)
            ha = "left"
            va = "bottom"

        ax.annotate(
            label,
            coord,
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=FONT_SIZE_ANNOTATION,
            fontweight="bold",
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.82,
                pad=2.4,
            ),
            zorder=12,
        )


def plot_eg_trajectories(
    ax,
    info,
    trajectory_df,
    labels=True,
):
    color_map = get_trajectory_colors(
        trajectory_df
    )

    for start_id, group in trajectory_df.groupby(
        "start_id"
    ):
        group = group.sort_values(
            "iteration"
        )

        is_best = (
            start_id
            == info["best_start_id"]
        )

        start_name = group.iloc[0][
            "start_name"
        ]

        color = color_map[start_name]

        ax.plot(
            group["PC1"],
            group["PC2"],
            linewidth=(
                BEST_TRAJECTORY_LINEWIDTH
                if is_best
                else TRAJECTORY_LINEWIDTH
            ),
            alpha=(
                1.0
                if is_best
                else 0.62
            ),
            color=color,
            label=(
                start_name
                if labels
                else None
            ),
            zorder=(
                6
                if is_best
                else 4
            ),
        )

        start = group.iloc[0]

        ax.scatter(
            start["PC1"],
            start["PC2"],
            marker="x",
            s=(
                120
                if is_best
                else 90
            ),
            linewidth=(
                2.8
                if is_best
                else 2.0
            ),
            color=color,
            alpha=(
                1.0
                if is_best
                else 0.78
            ),
            zorder=7,
        )


def set_pca_axes(
    ax,
    explained_var,
):
    ax.set_xlabel(
        (
            f"PC1 "
            f"({explained_var[0] * 100:.1f}% "
            f"explained variance)"
        ),
        fontsize=FONT_SIZE_AXES,
    )

    ax.set_ylabel(
        (
            f"PC2 "
            f"({explained_var[1] * 100:.1f}% "
            f"explained variance)"
        ),
        fontsize=FONT_SIZE_AXES,
    )

    ax.tick_params(
        axis="both",
        labelsize=FONT_SIZE_TICKS,
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=1.2,
        alpha=0.32,
    )


# ============================================================
# PCA TRAJECTORY
# ============================================================

def plot_routing_pca_2d(
    info,
    coordinate_dict,
    trajectory_df,
    explained_var,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(15, 10.5)
    )

    plot_candidate_edges(
        ax,
        info,
        coordinate_dict,
    )

    plot_candidate_vertices(
        ax,
        info,
        coordinate_dict,
    )

    plot_eg_trajectories(
        ax,
        info,
        trajectory_df,
    )

    selected = (
        get_selected_solution_coordinate(
            info,
            coordinate_dict,
            trajectory_df,
        )
    )

    ax.scatter(
        selected[0],
        selected[1],
        marker="*",
        s=500,
        color="cyan",
        edgecolor="black",
        linewidth=1.2,
        zorder=10,
        label="Selected hierarchical solution",
    )

    text = (
        "Selected vertex"
        if info["best_is_vertex"]
        else (
            f"Best iteration: "
            f"{info['best_iteration']}"
        )
    )

    ax.annotate(
        text,
        selected,
        xytext=(12, -22),
        textcoords="offset points",
        fontsize=FONT_SIZE_ANNOTATION,
        fontweight="bold",
    )

    set_pca_axes(
        ax,
        explained_var,
    )

    set_pca_limits(
        ax,
        coordinate_dict,
        right_margin=0.34,
    )

    ax.legend(
        fontsize=FONT_SIZE_LEGEND,
        facecolor="white",
        framealpha=0.92,
        loc="upper right",
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    fig.tight_layout(
        pad=1.5
    )

    path = os.path.join(
        output_dir,
        "hierarchical_PCA_trajectory_2D.png",
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(fig)

    print(
        f"2D routing PCA saved to:\n{path}",
        flush=True,
    )

    return path


# ============================================================
# LOSS TRAJECTORY
# ============================================================

def plot_loss_trajectory(
    info,
    trajectory_df,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(11.5, 7.0)
    )

    color_map = get_trajectory_colors(
        trajectory_df
    )

    for start_id, group in trajectory_df.groupby(
        "start_id"
    ):
        group = group.sort_values(
            "iteration"
        )

        is_best = (
            start_id
            == info["best_start_id"]
        )

        name = group.iloc[0][
            "start_name"
        ]

        ax.plot(
            group["iteration"],
            group["loss"],
            linewidth=(
                BEST_LOSS_LINEWIDTH
                if is_best
                else LOSS_LINEWIDTH
            ),
            alpha=(
                1.0
                if is_best
                else 0.62
            ),
            color=color_map[name],
            label=name,
        )

    ax.axhline(
        info["batch_loss"],
        linestyle="--",
        linewidth=REFERENCE_LINEWIDTH,
        label="Selected routing loss",
    )

    ax.set_xlabel(
        "EG iteration",
        fontsize=FONT_SIZE_AXES,
    )

    ax.set_ylabel(
        "Routing batch loss",
        fontsize=FONT_SIZE_AXES,
    )

    ax.tick_params(
        axis="both",
        labelsize=FONT_SIZE_TICKS,
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=1.2,
        alpha=0.32,
    )

    ax.legend(
        fontsize=FONT_SIZE_LEGEND
    )

    fig.tight_layout(
        pad=1.5
    )

    path = os.path.join(
        output_dir,
        "hierarchical_loss_trajectory.png",
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Loss trajectory saved to:\n{path}",
        flush=True,
    )

    return path


# ============================================================
# GENERAL SIMPLEX GRID
# ============================================================

def simplex_grid_weights(
    num_candidates,
    step=GRID_STEP,
):
    """Generate a regular grid over a K-candidate simplex."""
    if num_candidates < 1:
        raise ValueError(
            "num_candidates must be >= 1."
        )

    n = round(
        1.0 / step
    )

    if not np.isclose(
        n * step,
        1.0,
    ):
        raise ValueError(
            "GRID_STEP must divide 1 exactly."
        )

    if num_candidates == 1:
        return np.ones(
            (1, 1),
            dtype=float,
        )

    def compositions(
        total,
        parts,
        prefix=None,
    ):
        prefix = (
            []
            if prefix is None
            else prefix
        )

        if parts == 1:
            yield prefix + [total]
            return

        for value in range(
            total + 1
        ):
            yield from compositions(
                total - value,
                parts - 1,
                prefix + [value],
            )

    return (
        np.asarray(
            list(
                compositions(
                    n,
                    num_candidates,
                )
            ),
            dtype=float,
        )
        / n
    )


def simplex_grid_size(
    num_candidates,
    step=GRID_STEP,
):
    n = round(
        1.0 / step
    )

    return math.comb(
        n + num_candidates - 1,
        num_candidates - 1,
    )


def run_simplex_grid_search(
    info,
    candidate_state_dicts,
    batch,
    device,
    pretrained_model_name,
    output_dir,
    step=GRID_STEP,
    bank_repo_id=None,
    pretrained_subfolder=None,
):
    """Evaluate routing loss over a regular K-candidate simplex grid."""
    candidate_names = (
        info["candidate_names"]
    )

    K = len(
        candidate_names
    )

    if K == 0:
        raise ValueError(
            "No routing candidates available."
        )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    batch = {
        k: v.to(device)
        for k, v in batch.items()
    }

    model_template = load_pretrained_checkpoint(
        pretrained_model_name=pretrained_model_name,
        device=device,
        bank_repo_id=bank_repo_id,
        pretrained_subfolder=pretrained_subfolder,
    )

    grid_weights = simplex_grid_weights(
        K,
        step,
    )

    print(
        (
            f"Evaluating {K}-candidate "
            f"simplex grid: "
            f"{len(grid_weights)} points..."
        ),
        flush=True,
    )

    rows = []
    start_time = time.time()

    for point_id, weights_np in enumerate(
        grid_weights
    ):
        weights = torch.tensor(
            weights_np,
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            loss = batch_loss_at_weights(
                model_template,
                candidate_state_dicts,
                weights,
                batch,
                device,
            ).item()

        row = {
            "point_id": point_id,
            "loss": float(loss),
        }

        row.update({
            f"weight_{name}": float(w)
            for name, w in zip(
                candidate_names,
                weights_np,
            )
        })

        rows.append(row)

        if (
            (point_id + 1) % 25 == 0
            or point_id
            == len(grid_weights) - 1
        ):
            print(
                (
                    f"Grid point "
                    f"{point_id + 1}/"
                    f"{len(grid_weights)}"
                ),
                flush=True,
            )

    grid_time = (
        time.time()
        - start_time
    )

    grid_df = pd.DataFrame(
        rows
    )

    best_row = grid_df.loc[
        grid_df["loss"].idxmin()
    ]

    best_weights = np.array(
        [
            best_row[
                f"weight_{name}"
            ]
            for name in candidate_names
        ],
        dtype=float,
    )

    grid_df.to_csv(
        os.path.join(
            output_dir,
            "hierarchical_simplex_grid.csv",
        ),
        index=False,
    )

    summary = {
        "num_candidates": K,
        "grid_step": step,
        "num_grid_points": len(grid_df),
        "grid_best_loss": float(
            best_row["loss"]
        ),
        "eg_best_loss": float(
            info["batch_loss"]
        ),
        "eg_minus_grid_loss": float(
            info["batch_loss"]
            - best_row["loss"]
        ),
        "grid_best_weights": dict(
            zip(
                candidate_names,
                best_weights.tolist(),
            )
        ),
        "eg_weights": info["weights"],
        "grid_time_sec": grid_time,
    }

    pd.DataFrame([
        {
            "num_candidates": K,
            "grid_step": step,
            "num_grid_points": len(
                grid_df
            ),
            "grid_best_loss": summary[
                "grid_best_loss"
            ],
            "eg_best_loss": summary[
                "eg_best_loss"
            ],
            "eg_minus_grid_loss": summary[
                "eg_minus_grid_loss"
            ],
            "grid_best_weights": str(
                summary[
                    "grid_best_weights"
                ]
            ),
            "eg_weights": str(
                summary["eg_weights"]
            ),
            "grid_time_sec": grid_time,
        }
    ]).to_csv(
        os.path.join(
            output_dir,
            "hierarchical_grid_summary.csv",
        ),
        index=False,
    )

    del model_template

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        (
            "Grid search complete: "
            f"best loss="
            f"{summary['grid_best_loss']:.6f}, "
            f"hierarchical loss="
            f"{summary['eg_best_loss']:.6f}, "
            f"gap="
            f"{summary['eg_minus_grid_loss']:.6f}, "
            f"time={grid_time:.1f}s."
        ),
        flush=True,
    )

    return grid_df, summary


# ============================================================
# GRID PCA
# ============================================================

def compute_grid_pca_coordinates(
    info,
    coordinate_dict,
    grid_df,
):
    names = info[
        "candidate_names"
    ]

    candidate_coords = np.stack(
        [
            coordinate_dict[name]
            for name in names
        ],
        axis=0,
    )

    weights = grid_df[
        [
            f"weight_{name}"
            for name in names
        ]
    ].to_numpy(
        dtype=float
    )

    positions = (
        weights
        @ candidate_coords
    )

    result = grid_df.copy()

    result["PC1"] = positions[:, 0]
    result["PC2"] = positions[:, 1]

    return result


def make_light_colormap():
    base = plt.get_cmap(
        "YlOrRd"
    )

    return (
        mcolors
        .LinearSegmentedColormap
        .from_list(
            "light_YlOrRd",
            base(
                np.linspace(
                    0.03,
                    0.62,
                    256,
                )
            ),
        )
    )


def get_grid_optimum_coordinate(
    info,
    coordinate_dict,
    grid_summary,
):
    names = info[
        "candidate_names"
    ]

    weights = np.array(
        [
            grid_summary[
                "grid_best_weights"
            ][name]
            for name in names
        ],
        dtype=float,
    )

    coords = np.stack(
        [
            coordinate_dict[name]
            for name in names
        ],
        axis=0,
    )

    return (
        weights
        @ coords
    )


def plot_optima(
    ax,
    info,
    coordinate_dict,
    trajectory_df,
    grid_summary,
):
    hierarchical = (
        get_selected_solution_coordinate(
            info,
            coordinate_dict,
            trajectory_df,
        )
    )

    grid = (
        get_grid_optimum_coordinate(
            info,
            coordinate_dict,
            grid_summary,
        )
    )

    ax.scatter(
        hierarchical[0],
        hierarchical[1],
        marker="X",
        s=300,
        color="white",
        edgecolor="black",
        linewidth=1.8,
        zorder=15,
        label="Hierarchical optimum",
    )

    ax.scatter(
        grid[0],
        grid[1],
        marker="*",
        s=600,
        color="white",
        edgecolor="black",
        linewidth=1.8,
        zorder=16,
        label="Grid-search optimum",
    )


# ============================================================
# LOSS LANDSCAPE
# ============================================================

def plot_pca_loss_landscape(
    info,
    coordinate_dict,
    trajectory_df,
    grid_df,
    grid_summary,
    explained_var,
    output_dir,
):
    """Visualize grid loss landscape for any H in the common PCA plane."""
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    grid_pca_df = compute_grid_pca_coordinates(
        info,
        coordinate_dict,
        grid_df,
    )

    x = grid_pca_df[
        "PC1"
    ].to_numpy()

    y = grid_pca_df[
        "PC2"
    ].to_numpy()

    losses = grid_pca_df[
        "loss"
    ].to_numpy()

    K = len(
        info["candidate_names"]
    )

    fig, ax = plt.subplots(
        figsize=(17, 11.5)
    )

    cmap = make_light_colormap()

    # H=3: true triangular 2D simplex -> contour heatmap.
    if K == 3:
        triangulation = (
            mtri.Triangulation(
                x,
                y,
            )
        )

        heatmap = ax.tricontourf(
            triangulation,
            losses,
            levels=30,
            cmap=cmap,
            alpha=0.55,
            zorder=0,
        )

        cbar = fig.colorbar(
            heatmap,
            ax=ax,
            pad=0.055,
            fraction=0.046,
        )

    # H=2: line simplex; H>3:
    # higher-dimensional simplex projected into PCA.
    else:
        scatter = ax.scatter(
            x,
            y,
            c=losses,
            cmap=cmap,
            s=(
                85
                if K > 3
                else 130
            ),
            alpha=0.75,
            edgecolor="none",
            zorder=2,
        )

        cbar = fig.colorbar(
            scatter,
            ax=ax,
            pad=0.055,
            fraction=0.046,
        )

    cbar.set_label(
        "Routing batch loss",
        fontsize=FONT_SIZE_AXES,
    )

    cbar.ax.tick_params(
        labelsize=FONT_SIZE_TICKS
    )

    plot_candidate_edges(
        ax,
        info,
        coordinate_dict,
        color="black",
        alpha=0.65,
        linestyle="-",
    )

    plot_candidate_vertices(
        ax,
        info,
        coordinate_dict,
    )

    plot_eg_trajectories(
        ax,
        info,
        trajectory_df,
        labels=True,
    )

    plot_optima(
        ax,
        info,
        coordinate_dict,
        trajectory_df,
        grid_summary,
    )

    set_pca_axes(
        ax,
        explained_var,
    )

    set_pca_limits(
        ax,
        coordinate_dict,
        right_margin=0.48,
        left_margin=0.18,
        bottom_margin=0.20,
        top_margin=0.20,
    )

    # Two-line title: keeps the loss information
    # away from the colorbar on the right.
    ax.set_title(
        (
            "Routing loss landscape\n"
            f"Hierarchical optimum loss: "
            f"{grid_summary['eg_best_loss']:.4f} | "
            f"Grid-search optimum loss: "
            f"{grid_summary['grid_best_loss']:.4f}"
        ),
        fontsize=FONT_SIZE_LEGEND,
        pad=16,
    )

    ax.legend(
        fontsize=FONT_SIZE_LEGEND,
        facecolor="white",
        framealpha=0.95,
        loc="upper right",
        bbox_to_anchor=(
            0.985,
            0.985,
        ),
        borderaxespad=0.0,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    # Slightly more room at the top for the two-line title,
    # while still reserving space for the colorbar.
    fig.subplots_adjust(
        left=0.10,
        right=0.86,
        bottom=0.12,
        top=0.88,
    )

    path = os.path.join(
        output_dir,
        "hierarchical_PCA_loss_landscape.png",
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(fig)

    grid_pca_df.to_csv(
        os.path.join(
            output_dir,
            "hierarchical_simplex_grid_pca.csv",
        ),
        index=False,
    )

    print(
        f"PCA loss landscape saved to:\n{path}",
        flush=True,
    )

    return path


# ============================================================
# COMPLETE ROUTING PCA
# ============================================================

def run_routing_pca(
    info,
    output_dir,
    batch=None,
    device=None,
    pretrained_model_name=MODEL_NAME,
    bank_repo_id=None,
    pretrained_subfolder=None,
    run_grid=True,
    grid_step=GRID_STEP,
):
    print(
        "\n============================================================"
    )
    print(
        "HIERARCHICAL ROUTING PCA"
    )
    print(
        "============================================================",
        flush=True,
    )

    candidate_state_dicts = (
        load_routing_candidates(
            info,
            pretrained_model_name,
            bank_repo_id,
            pretrained_subfolder,
        )
    )

    pca_state_dicts = (
        add_pretrained_reference(
            candidate_state_dicts,
            pretrained_model_name,
            bank_repo_id,
            pretrained_subfolder,
        )
    )

    grid_df = None
    grid_summary = None

    try:
        coordinate_dict, explained_var = (
            compute_candidate_pca(
                pca_state_dicts
            )
        )

        trajectory_df = (
            compute_trajectory_coordinates(
                info,
                coordinate_dict,
            )
        )

        save_trajectory_coordinates(
            trajectory_df,
            output_dir,
        )

        plot_routing_pca_2d(
            info,
            coordinate_dict,
            trajectory_df,
            explained_var,
            output_dir,
        )

        plot_loss_trajectory(
            info,
            trajectory_df,
            output_dir,
        )

        if run_grid:
            if (
                batch is None
                or device is None
            ):
                raise ValueError(
                    "batch and device are required "
                    "when run_grid=True."
                )

            grid_df, grid_summary = (
                run_simplex_grid_search(
                    info=info,
                    candidate_state_dicts=(
                        candidate_state_dicts
                    ),
                    batch=batch,
                    device=device,
                    pretrained_model_name=(
                        pretrained_model_name
                    ),
                    output_dir=output_dir,
                    step=grid_step,
                    bank_repo_id=(
                        bank_repo_id
                    ),
                    pretrained_subfolder=(
                        pretrained_subfolder
                    ),
                )
            )

            plot_pca_loss_landscape(
                info=info,
                coordinate_dict=coordinate_dict,
                trajectory_df=trajectory_df,
                grid_df=grid_df,
                grid_summary=grid_summary,
                explained_var=explained_var,
                output_dir=output_dir,
            )

    finally:
        del candidate_state_dicts
        del pca_state_dicts

    print(
        "\n============================================================"
    )
    print(
        "HIERARCHICAL ROUTING PCA COMPLETE"
    )
    print(
        "============================================================",
        flush=True,
    )

    return (
        coordinate_dict,
        trajectory_df,
        explained_var,
        grid_df,
        grid_summary,
    )