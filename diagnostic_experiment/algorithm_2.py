import torch
import pandas as pd
import shift_measurement as sm
import interpolation_utils as iu
from transformers import AutoModelForCausalLM

def load_model_bank_metadata(csv_path):
    """Load model bank metadata CSV."""
    return pd.read_csv(csv_path)

def select_closest_source(bank_df, deployment_batch, vocab_size, epsilon=1e-8):
    """Select j* = argmin_j KL(p_B || p_Dj)."""
    p_B = sm.compute_batch_token_distribution(deployment_batch, vocab_size, epsilon=epsilon)

    kl_values = {}

    for dataset_name in bank_df["dataset_name"].unique():
        dataset_rows = bank_df[bank_df["dataset_name"] == dataset_name]

        source_distribution_path = dataset_rows.iloc[0]["source_distribution_path"]
        p_Dj = torch.load(source_distribution_path, map_location=p_B.device)

        rho_j = sm.compute_kl(p_B, p_Dj)
        kl_values[dataset_name] = float(rho_j)

    selected_dataset = min(kl_values, key=kl_values.get)

    return selected_dataset, kl_values

def select_lambda_hat_from_curve(lambda_rho_curve_path, rho_B):
    """Select lambda corresponding to the closest rho in the lambda-rho curve."""
    curve_df = pd.read_csv(lambda_rho_curve_path)

    idx = (curve_df["rho"] - rho_B).abs().idxmin() # Select the index of the closest rho value by looking at the absolute difference between rho_B and rho values in the curve.
    selected_row = curve_df.loc[idx]
    lambda_hat = float(selected_row["lambda"])

    return lambda_hat

def select_lambda_interval_from_trained_grid(bank_df, selected_dataset, lambda_hat):
    """Select trained lambda interval [lambda_i, lambda_{i+1}] around predicted lambda_hat."""
    dataset_rows = bank_df[bank_df["dataset_name"] == selected_dataset].copy()
    trained_lambdas = sorted(dataset_rows["lambda"].astype(float).unique())
    lambda_hat = float(lambda_hat)

    if len(trained_lambdas) < 2:
        raise ValueError("Need at least two trained lambdas for midpoint interpolation.")

    if lambda_hat <= trained_lambdas[0]:
        lambda_left = trained_lambdas[0]
        lambda_right = trained_lambdas[1]

    elif lambda_hat >= trained_lambdas[-1]:
        lambda_left = trained_lambdas[-2]
        lambda_right = trained_lambdas[-1]

    else:
        for i in range(len(trained_lambdas) - 1):
            if trained_lambdas[i] <= lambda_hat <= trained_lambdas[i + 1]:
                lambda_left = trained_lambdas[i]
                lambda_right = trained_lambdas[i + 1]
                break

    return lambda_left, lambda_right

def select_model_interval_from_bank(bank_df, selected_dataset, lambda_left, lambda_right):
    """Select model paths for the two endpoint lambdas."""
    dataset_rows = bank_df[bank_df["dataset_name"] == selected_dataset].copy()
    dataset_rows["lambda_float"] = dataset_rows["lambda"].astype(float)

    left_idx = dataset_rows[dataset_rows["lambda_float"] == float(lambda_left)].index[0]
    right_idx = dataset_rows[dataset_rows["lambda_float"] == float(lambda_right)].index[0]

    left_model_row = dataset_rows.loc[left_idx]
    right_model_row = dataset_rows.loc[right_idx]

    return left_model_row, right_model_row


def run_algorithm_2(
    model_bank_metadata_path,
    deployment_batch,
    vocab_size,
    config,
    device,
    epsilon=1e-8,
):
    """Run Algorithm 2 and return the midpoint-interpolated model for deployment batch B."""

    bank_df = load_model_bank_metadata(model_bank_metadata_path)

    selected_dataset, kl_values = select_closest_source(
        bank_df=bank_df,
        deployment_batch=deployment_batch,
        vocab_size=vocab_size,
        epsilon=epsilon,
    )

    rho_B = kl_values[selected_dataset]

    dataset_rows = bank_df[bank_df["dataset_name"] == selected_dataset]
    lambda_rho_curve_path = dataset_rows.iloc[0]["lambda_rho_curve_path"]

    lambda_hat = select_lambda_hat_from_curve(
        lambda_rho_curve_path=lambda_rho_curve_path,
        rho_B=rho_B,
    )

    lambda_left, lambda_right = select_lambda_interval_from_trained_grid(
        bank_df=bank_df,
        selected_dataset=selected_dataset,
        lambda_hat=lambda_hat,
    )

    left_model_row, right_model_row = select_model_interval_from_bank(
        bank_df=bank_df,
        selected_dataset=selected_dataset,
        lambda_left=lambda_left,
        lambda_right=lambda_right,
    )

    model_left = AutoModelForCausalLM.from_pretrained(
        left_model_row["save_dir"]
    ).to(device)

    model_right = AutoModelForCausalLM.from_pretrained(
        right_model_row["save_dir"]
    ).to(device)

    theta_B = iu.interpolate_models(
        model_left=model_left,
        model_right=model_right,
        beta=0.5,
        config=config,
        device=device,
    )

    del model_left
    del model_right
    if device.type == "cuda":
        torch.cuda.empty_cache()

    info = {
        "selected_dataset": selected_dataset,
        "rho_B": rho_B,
        "kl_values": kl_values,
        "lambda_hat": lambda_hat,
        "lambda_left": lambda_left,
        "lambda_right": lambda_right,
        "left_model_path": left_model_row["save_dir"],
        "right_model_path": right_model_row["save_dir"],
        "beta": 0.5,
    }

    return theta_B, info