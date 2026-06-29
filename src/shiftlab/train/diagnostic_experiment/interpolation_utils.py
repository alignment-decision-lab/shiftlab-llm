import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import lambda_analysis_utils as lau
from transformers import AutoModelForCausalLM

def build_interpolation_configs(config):
    """ Create intervals for each lambda_beta with different width. """
    beta = config["interpolation"]["beta"]
    lambdas_beta = config["interpolation"]["lambdas_beta"]
    interval_widths = config["interpolation"]["interval_widths"]

    all_configs = []
    
    for lambda_beta in lambdas_beta:
        for width in interval_widths:
            lambda_left = lambda_beta - beta * width
            lambda_right = lambda_beta + (1 - beta) * width

            assert lambda_left >=0
            assert lambda_left < lambda_beta < lambda_right

            configs = {"lambda_beta": lambda_beta,
                       "beta": beta,
                      "width": width,
                      "lambda_left": lambda_left,
                      "lambda_right": lambda_right}
            all_configs.append(configs)
    return all_configs

def load_interpolation_models(one_config, saved_paths, config, device):
    """ Load 3 models for 3 lambdas: lambda_left, lambda_beta, lambda_right. """
    lambda_left = one_config["lambda_left"]
    lambda_right = one_config["lambda_right"]
    lambda_beta = one_config["lambda_beta"]

    model_left = lau.load_lambda_models(saved_paths, lambda_left, config, device)
    model_right = lau.load_lambda_models(saved_paths, lambda_right, config, device)
    model_trained = lau.load_lambda_models(saved_paths, lambda_beta, config, device)

    return model_left, model_right, model_trained

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

def parameter_l2_distance(model_a, model_b):
    """Compute L2 distance between two models using state_dict, as in delta_update."""
    diffs = []

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    for name in state_a.keys():
        if name in state_b:
            p_a = state_a[name]
            p_b = state_b[name]

            if torch.is_floating_point(p_a):
                diff = p_a.detach().cpu().float() - p_b.detach().cpu().float()
                diffs.append(diff.flatten())

    diff_vec = torch.cat(diffs)
    return torch.sqrt(torch.dot(diff_vec, diff_vec)).item()

def curvature_ratio(model_left, model_beta, model_right):
    """Compute ||theta_L - 2 theta_beta + theta_R|| / ||theta_L - theta_R||."""
    numerator_parts = []
    denominator_parts = []

    state_left = model_left.state_dict()
    state_beta = model_beta.state_dict()
    state_right = model_right.state_dict()

    for name in state_left.keys():
        if name in state_beta and name in state_right:
            p_l = state_left[name]
            p_b = state_beta[name]
            p_r = state_right[name]

            if torch.is_floating_point(p_l):
                second_diff = (
                    p_l.detach().cpu().float()
                    - 2 * p_b.detach().cpu().float()
                    + p_r.detach().cpu().float()
                )
                endpoint_diff = (
                    p_l.detach().cpu().float()
                    - p_r.detach().cpu().float()
                )

                numerator_parts.append(second_diff.flatten())
                denominator_parts.append(endpoint_diff.flatten())

    numerator = torch.cat(numerator_parts)
    denominator = torch.cat(denominator_parts)

    return (
        torch.sqrt(torch.dot(numerator, numerator))
        / (torch.sqrt(torch.dot(denominator, denominator)) + 1e-12)
    ).item()

def compute_interpolation_distances(initial_model, model_left, model_right, model_trained, model_interp, probe_dataloader, device):
    """Compute distances between trained, interpolated, and endpoint models."""

    delta_trained = lau.delta_update(initial_model, model_trained)
    delta_interp = lau.delta_update(initial_model, model_interp)

    d_interp = lau.d_step(delta_trained, delta_interp)
    d_init = lau.d_init(delta_trained)
    d_rel = d_interp / (d_init + 1e-12)

    # New diagnostics
    d_endpoints = parameter_l2_distance(model_left, model_right)
    d_left_trained = parameter_l2_distance(model_left, model_trained)
    d_right_trained = parameter_l2_distance(model_right, model_trained)

    endpoint_ratio = d_interp / (d_endpoints + 1e-12)

    D_pred = lau.D_pred(model_trained, model_interp, probe_dataloader, device)

    trained_loss, _, _ = utils.evaluation(model_trained, probe_dataloader, device)
    interp_loss, _, _ = utils.evaluation(model_interp, probe_dataloader, device)
    delta_loss = abs(trained_loss - interp_loss)
    loss_left, _, _ = utils.evaluation(model_left, probe_dataloader, device)
    loss_right, _, _ = utils.evaluation(model_right, probe_dataloader, device)

    best_endpoint_loss = min(loss_left, loss_right)
    delta_interp_best_endpoint = interp_loss - best_endpoint_loss

    c_curvature = curvature_ratio(model_left, model_trained, model_right)
    r_triangle = (d_left_trained + d_right_trained) / (d_endpoints + 1e-12)

    return {
        "d_interp": d_interp,
        "d_init": d_init,
        "d_rel": d_rel,
        "d_endpoints": d_endpoints,
        "d_left_trained": d_left_trained,
        "d_right_trained": d_right_trained,
        "endpoint_ratio": endpoint_ratio,
        "D_pred": D_pred,

        "loss_left": loss_left,
        "loss_right": loss_right,
        "loss_trained": trained_loss,
        "loss_interp": interp_loss,

        "delta_loss": delta_loss,
        "best_endpoint_loss": best_endpoint_loss,
        "delta_interp_best_endpoint": delta_interp_best_endpoint,

        "c_curvature": c_curvature,
        "r_triangle": r_triangle,
    }

def build_required_lambdas(config):
    """Build the list of lambda values needed for the interpolation experiment."""
    interpolation_configs = build_interpolation_configs(config)

    required_lambdas = set()

    for one_config in interpolation_configs:
        required_lambdas.add(round(one_config["lambda_left"], 6))
        required_lambdas.add(round(one_config["lambda_beta"], 6))
        required_lambdas.add(round(one_config["lambda_right"], 6))

    return sorted(required_lambdas)

def train_required_lambda_models(train_dataloader, val_dataloader, device, config):
    """ Train and save models for each λ. """
    
    lambdas = build_required_lambdas(config)
    saved_paths = {}
    for lambd in lambdas:
        model, _, _ = utils.setup_model_and_tokenizer(config, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"][ "learning_rate"]), weight_decay=float(config["training"].get("weight_decay",0.0)))
        if lambd == 0.0:
            print(f"\n ===== Training ERM (λ=0) =====", flush=True)
            for epoch in range(config["training"]["epochs"]):
                train_loss = utils.train_one_epoch(model, train_dataloader, optimizer, device=device)
                val_loss, val_ppl, val_acc = utils.evaluation(model, val_dataloader, device=device)
                print(f"[λ=0.0] Epoch {epoch+1}/{config['training']['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Val Acc: {val_acc:.4f}", flush=True)
        else:
            print(f"\n ===== Training KL-DRO-{lambd} =====", flush=True)
            for epoch in range(config["training"]["epochs"]):
                train_loss = utils.KL_DRO_one_epoch(model, train_dataloader, optimizer, gamma=config["training"]["gamma"], lambd=lambd, rho=config["training"]["rho"], device=device)
                val_loss, val_ppl, val_acc = utils.evaluation(model, val_dataloader, device=device)
                print(f"[λ={lambd}] Epoch {epoch+1}/{config['training']['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Val Acc: {val_acc:.4f}", flush=True)
        save_dir = lau.save_lambda_models(model, lambd, config)
        saved_paths[lambd] = save_dir
        
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3

            print(f"[λ={lambd}] GPU allocated : {allocated:.2f} GB")
            print(f"[λ={lambd}] GPU reserved  : {reserved:.2f} GB")
            print(f"[λ={lambd}] GPU peak      : {peak:.2f} GB")

        del model
        torch.cuda.empty_cache() # to free up GPU memory after each model is trained and saved.
    return saved_paths

def run_one_interpolation(one_config, saved_paths, config, device, probe_dataloader):
    """ Run one interpolation and return the metrics. """
    beta = one_config["beta"]
    lambda_left = one_config["lambda_left"]
    lambda_right = one_config["lambda_right"]
    lambda_beta = one_config["lambda_beta"]
    width = one_config["width"]
    print(
        f"\nInterpolation config: "
        f"lambda_beta={lambda_beta}, width={width}, "
        f"lambda_left={lambda_left}, lambda_right={lambda_right}",
        flush=True
    )

    model_left, model_right, model_trained = load_interpolation_models(one_config, saved_paths, config, device)
    model_interp = interpolate_models(model_left, model_right, beta, config, device)
    initial_model, _, _ = utils.setup_model_and_tokenizer(config, device)

    results = compute_interpolation_distances(
            initial_model,
            model_left,
            model_right,
            model_trained,
            model_interp,
            probe_dataloader,
            device )
    print(
        f"d_endpoints={results['d_endpoints']:.4f}, "
        f"d_interp={results['d_interp']:.4f}, "
        f"d_rel={results['d_rel']:.4f}, "
        f"endpoint_ratio={results['endpoint_ratio']:.4f}",
        f"c_curvature={results['c_curvature']:.4f}, "
        f"r_triangle={results['r_triangle']:.4f}", 
        flush=True
    )
    del model_left
    del model_right
    del model_trained
    del model_interp
    del initial_model
    torch.cuda.empty_cache()
    results["lambda_left"] = lambda_left
    results["lambda_right"] = lambda_right
    results["lambda_beta"] = lambda_beta
    results["beta"] = beta
    results["width"] = width

    return results

def run_all_interpolations(saved_paths, config, device, probe_dataloader, seed):
    """ Run all the interpolation and save the results in a CSV table. """

    interpolation_configs = build_interpolation_configs(config)

    rows = []

    for one_config in interpolation_configs:
        results = run_one_interpolation(
            one_config,
            saved_paths,
            config,
            device,
            probe_dataloader
        )
        if seed is not None:
            results["seed"] = seed
        rows.append(results)

    df = pd.DataFrame(rows)

    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    if seed is None:
        csv_path = os.path.join(output_dir, "interpolation_results.csv")
    else:
        csv_path = os.path.join(output_dir, f"interpolation_results_seed_{seed}.csv")
    df.to_csv(csv_path, index=False)

    print(df)
    print(f"Results saved to {csv_path}")

    return df

def aggregate_seed_results(df):
    """Compute mean and std across seeds."""
    metrics = [
    "d_interp",
    "d_init",
    "d_rel",
    "d_endpoints",
    "d_left_trained",
    "d_right_trained",
    "endpoint_ratio",
    "D_pred",

    "loss_left",
    "loss_right",
    "loss_trained",
    "loss_interp",

    "delta_loss",
    "best_endpoint_loss",
    "delta_interp_best_endpoint",

    "c_curvature",
    "r_triangle",
]
    grouped = df.groupby(["lambda_beta", "width", "beta"])[metrics]

    mean_df = grouped.mean().reset_index()
    std_df = grouped.std().reset_index().fillna(0.0)

    for metric in metrics:
        mean_df[f"{metric}_std"] = std_df[metric]

    return mean_df

def interpolation_bar_plots(df, config):
    """Build grouped bar plots for interpolation metrics."""
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    beta = config["interpolation"]["beta"]
    metrics = ["d_rel", "D_pred", "delta_loss"]

    widths = sorted(df["width"].unique())
    lambda_betas = sorted(df["lambda_beta"].unique())

    x = range(len(lambda_betas))
    bar_width = 0.8 / len(widths)

    plt.figure(figsize=(18, 6))

    for plot_idx, metric in enumerate(metrics):
        plt.subplot(1, 3, plot_idx + 1)

        for width_idx, width in enumerate(widths):
            df_width = df[df["width"] == width].sort_values("lambda_beta")
            y = df_width[metric].values
            yerr = df_width[f"{metric}_std"].values

            positions = [
                xi - 0.4 + bar_width / 2 + width_idx * bar_width
                for xi in x
            ]

            plt.bar(positions, y, width=bar_width, yerr=yerr, capsize=4, label=f"width={width}")


        plt.xticks(x, lambda_betas)
        plt.xlabel(r"$\lambda_\beta$")
        plt.ylabel(metric)
        plt.title(f"{metric} (β={beta})")

        if plot_idx == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "interpolation_barplots.png"))
    plt.close()

def geometric_diagnostic_plots(df, config):
    """Plot curvature ratio and triangle ratio."""
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    beta = config["interpolation"]["beta"]
    metrics = ["c_curvature", "r_triangle"]
    titles = [
        r"Curvature ratio $c$",
        r"Triangle ratio $r_{\triangle}$"
    ]

    widths = sorted(df["width"].unique())
    lambda_betas = sorted(df["lambda_beta"].unique())

    x = range(len(lambda_betas))
    bar_width = 0.8 / len(widths)

    plt.figure(figsize=(12, 5))

    for plot_idx, metric in enumerate(metrics):
        plt.subplot(1, 2, plot_idx + 1)

        for width_idx, width in enumerate(widths):
            df_width = df[df["width"] == width].sort_values("lambda_beta")

            y = df_width[metric].values

            if f"{metric}_std" in df_width.columns:
                yerr = df_width[f"{metric}_std"].values
            else:
                yerr = None

            positions = [
                xi - 0.4 + bar_width / 2 + width_idx * bar_width
                for xi in x
            ]

            plt.bar(
                positions,
                y,
                width=bar_width,
                yerr=yerr,
                capsize=4,
                label=f"width={width}",
            )

        plt.xticks(x, lambda_betas)
        plt.xlabel(r"$\lambda_\beta$")
        plt.ylabel(metric)
        plt.title(f"{titles[plot_idx]} ($\\beta={beta}$)")

        if metric == "r_triangle":
            plt.axhline(
                y=1.0,
                linestyle="--",
                linewidth=1,
                color="gray",
                label="perfect alignment"
            )

        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "geometric_diagnostics.png"))
    plt.close()

def performance_comparison_plots(df, config):
    """Plot performance-space comparison for interpolated models."""
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    beta = config["interpolation"]["beta"]

    metrics = [
        "delta_loss",
        "delta_interp_best_endpoint",
    ]

    titles = [
        r"Gap to directly trained target",
        r"Gap to best endpoint",
    ]

    ylabels = [
        r"$|L(\theta^{interp}_{\lambda_\beta}) - L(\theta^{trained}_{\lambda_\beta})|$",
        r"$L(\theta^{interp}_{\lambda_\beta}) - \min(L(\theta_{\lambda_L}), L(\theta_{\lambda_R}))$",
    ]

    widths = sorted(df["width"].unique())
    lambda_betas = sorted(df["lambda_beta"].unique())

    x = range(len(lambda_betas))
    bar_width = 0.8 / len(widths)

    plt.figure(figsize=(12, 5))

    for plot_idx, metric in enumerate(metrics):
        plt.subplot(1, 2, plot_idx + 1)

        for width_idx, width in enumerate(widths):
            df_width = df[df["width"] == width].sort_values("lambda_beta")

            y = df_width[metric].values

            if f"{metric}_std" in df_width.columns:
                yerr = df_width[f"{metric}_std"].values
            else:
                yerr = None

            positions = [
                xi - 0.4 + bar_width / 2 + width_idx * bar_width
                for xi in x
            ]

            plt.bar(
                positions,
                y,
                width=bar_width,
                yerr=yerr,
                capsize=4,
                label=f"width={width}",
            )

        plt.xticks(x, lambda_betas)
        plt.xlabel(r"$\lambda_\beta$")
        plt.ylabel(ylabels[plot_idx])
        plt.title(f"{titles[plot_idx]} ($\\beta={beta}$)")

        if metric == "delta_interp_best_endpoint":
            plt.axhline(
                y=0.0,
                linestyle="--",
                linewidth=1,
                color="gray",
                label="same as best endpoint",
            )

        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_comparison.png"))
    plt.close()