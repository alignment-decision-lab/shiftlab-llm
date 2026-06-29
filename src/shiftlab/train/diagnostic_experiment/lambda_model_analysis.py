import torch
import utils
import time
from shiftlab.data.load_datasets import load_dataset_from_config
import lambda_analysis_utils as lau


# -- RUN --
def run_lambda_analysis(config):
    device = utils.get_device()
    _ , tokenizer, data_collator = utils.setup_model_and_tokenizer(config, device)

    if device.type == "cuda":
        gpu_id = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU id: {gpu_id}")
        print(f"GPU name: {torch.cuda.get_device_name(gpu_id)}")
    
    raw_dataset = load_dataset_from_config(config)
    train_raw, _ = utils.split_train_test_dataset(raw_dataset, config)
    train_pool = utils.tokenize_dataset(train_raw, tokenizer, config)
    train_dataset, val_dataset, train_dataloader, val_dataloader = utils.create_mixture_train_val_loaders(train_pool, data_collator, config)

    start_time = time.time()
    saved_paths = lau.train_lambda_models(train_dataloader, val_dataloader, device, config)
    lambdas, d_init_values, d_step_values, cos_sim_values, D_pred_values, X_2D, explained_var = lau.compute_lambda_metrics(saved_paths, config, device, val_dataloader)
    end_time = time.time()
    print(f"Total time for training and computing metrics: {end_time - start_time:.2f} seconds")
    print(f"Total time for training and computing metrics: {(end_time - start_time)/60:.2f} minutes")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nGlobal peak GPU memory : {peak:.2f} GB")
    lau.plot_lambda_metrics(lambdas, d_init_values, d_step_values, cos_sim_values, D_pred_values, config)
    lau.lambda_analysis_table(lambdas, d_init_values, d_step_values, cos_sim_values, D_pred_values, config)
    lau.plot_pca_updates(X_2D, lambdas, explained_var, config)
    print("Done")

if __name__ == "__main__":
    config = utils.load_config()
    run_lambda_analysis(config)
