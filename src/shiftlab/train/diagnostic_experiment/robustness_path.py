import os
import copy
import torch
import utils
import time
import pandas as pd
from shiftlab.data.load_datasets import load_dataset_from_config
import interpolation_utils as iu


def main(config):
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_output_dir = config["outputs"]["dir"]
    os.makedirs(base_output_dir, exist_ok=True)

    train_seed = config["training"]["seed"]
    probe_seeds = config["experiment"]["probe_seeds"]

    config_train = copy.deepcopy(config)
    config_train["training"]["seed"] = train_seed
    config_train["outputs"]["dir"] = os.path.join(
        base_output_dir,
        f"train_seed_{train_seed}"
    )

    print(f"\n===== Training once with seed {train_seed} =====")
    print(f"Output dir: {config_train['outputs']['dir']}")

    base_model, tokenizer, data_collator = utils.setup_model_and_tokenizer(
        config_train,
        device
    )
    del base_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw_dataset = load_dataset_from_config(config_train)
    train_raw, _ = utils.split_train_test_dataset(raw_dataset, config_train)
    train_pool = utils.tokenize_dataset(train_raw, tokenizer, config_train)

    train_dataset, val_dataset, train_dataloader, val_dataloader = (
        utils.create_mixture_train_val_loaders(
            train_pool,
            data_collator,
            config_train
        )
    )

    saved_paths = iu.train_required_lambda_models(
        train_dataloader,
        val_dataloader,
        device,
        config_train
    )

    all_dfs = []

    for probe_seed in probe_seeds:
        print(f"\n===== Evaluating probe seed {probe_seed} =====")

        probe_dataloader = utils.create_probe_dataloader(
            val_dataset=val_dataset,
            data_collator=data_collator,
            batch_size=config_train["training"]["batch_size"],
            probe_seed=probe_seed,
            probe_size=config_train["experiment"].get("probe_size", None),
        )

        df_probe = iu.run_all_interpolations(
            saved_paths=saved_paths,
            config=config_train,
            device=device,
            probe_dataloader=probe_dataloader,
            seed=probe_seed,
        )

        df_probe["train_seed"] = train_seed
        df_probe["probe_seed"] = probe_seed

        all_dfs.append(df_probe)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    df_all = pd.concat(all_dfs, ignore_index=True)

    all_csv_path = os.path.join(
        base_output_dir,
        "interpolation_results_all_probe_seeds.csv"
    )
    df_all.to_csv(all_csv_path, index=False)
    print(f"All probe-seed results saved to {all_csv_path}")

    df_agg = iu.aggregate_seed_results(df_all)

    agg_csv_path = os.path.join(
        base_output_dir,
        "interpolation_results_mean_std_probe_seeds.csv"
    )
    df_agg.to_csv(agg_csv_path, index=False)
    print(f"Aggregated results saved to {agg_csv_path}")

    iu.interpolation_bar_plots(df_agg, config_train)
    iu.geometric_diagnostic_plots(df_agg, config_train)
    iu.performance_comparison_plots(df_agg, config)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Total time: {(end_time - start_time) / 60:.2f} minutes")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nGlobal peak GPU memory : {peak:.2f} GB")

    print("Done")


if __name__ == "__main__":
    config = utils.load_config()
    main(config)