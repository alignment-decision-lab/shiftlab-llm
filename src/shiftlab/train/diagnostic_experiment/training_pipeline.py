import utils
import torch
from shiftlab.data.load_datasets import load_dataset_from_config, load_dataset_from_subconfig
import time


# def run_training(config):
#     device = utils.get_device()
#     model, tokenizer, data_collator = utils.setup_model_and_tokenizer(config, device)

#     if device.type == "cuda":
#         gpu_id = torch.cuda.current_device()
#         torch.cuda.reset_peak_memory_stats()
#         print(f"GPU id: {gpu_id}")
#         print(f"GPU name: {torch.cuda.get_device_name(gpu_id)}")
    
#     dataset = load_dataset_from_config(config)
#     tokenized_dataset = utils.tokenize_dataset(dataset, tokenizer, config)

#     losses = utils.compute_sample_losses(model, tokenized_dataset, data_collator, device, config)
#     easy_dataset, hard_dataset = utils.split_easy_hard(tokenized_dataset, losses, config["diagnostic"]["alpha"])
#     mixture_dataset = utils.create_mixture_dataset(easy_dataset, hard_dataset, config["diagnostic"]["beta_train"], config["diagnostic"]["train_size"])

#     train_dataset, val_dataset, train_dataloader, val_dataloader = utils.create_mixture_train_val_loaders(mixture_dataset, data_collator, config)
#     print(f"Dataset size: train={len(train_dataset)}, val={len(val_dataset)}")

#     initial_val_loss, initial_val_ppl, initial_val_acc = utils.evaluation(model, val_dataloader, device)

#     print(
#         f"Initial Val Loss: {initial_val_loss:.4f} | "
#         f"Initial PPL: {initial_val_ppl:.2f} | "
#         f"Initial Acc: {initial_val_acc:.4f}",
#         flush=True
#     )
#     start_time = time.time()
#     train_losses, val_losses, val_perplexities, val_accuracies, test_losses_by_epoch = utils.train_mixture(model, train_dataloader, val_dataloader, easy_dataset, hard_dataset, data_collator, device, config)
#     end_time = time.time()
#     training_time = end_time - start_time

#     print(f"Training time: {training_time:.2f} seconds", flush=True)
#     print(f"Training time: {training_time/60:.2f} minutes", flush=True)
#     if device.type == "cuda":
#         max_memory = torch.cuda.max_memory_allocated() / 1024**3
#         print(f"Max GPU memory allocated: {max_memory:.2f} GB", flush=True)

#     val_losses_initial = [initial_val_loss] * len(train_losses)
#     val_perplexities_initial = [initial_val_ppl] * len(train_losses)

#     final_test_losses, final_tail_losses, hist_losses_by_beta = utils.evaluate_mixture(model, easy_dataset, hard_dataset, data_collator, device, config)

#     utils.plot_training_curves(val_losses_initial, train_losses, val_losses, val_perplexities_initial, val_perplexities, val_accuracies, config)
#     utils.plot_diagnostic_curves(config["diagnostic"]["betas_test"], config["diagnostic"]["histogram_betas"], final_test_losses, final_tail_losses, hist_losses_by_beta, test_losses_by_epoch, config)

# #     results = {
# #     "ERM": {
# #         "train_losses": ...,
# #         "val_losses": ...,
# #         "val_perplexities": ...,
# #         "val_accuracies": ...,
# #         "eval": {
# #             "mostly_easy": {
# #                 "mean_loss": ...,
# #                 "tail_loss": ...,
# #                 "sample_losses": ...
# #             },
# #             "balanced": {
# #                 ...
# #             },
# #             "mostly_hard": {
# #                 ...
# #             }
# #         }
# #     },
# #     "rho_small": {
# #         ...
# #     },
# #     "rho_medium": {
# #         ...
# #     },
# #     "rho_large": {
# #         ...
# #     }
# # }

# def run_training_rho(config):
#     device = utils.get_device()
#     model, tokenizer, data_collator = utils.setup_model_and_tokenizer(config, device)

#     if device.type == "cuda":
#         gpu_id = torch.cuda.current_device()
#         torch.cuda.reset_peak_memory_stats()
#         print(f"GPU id: {gpu_id}")
#         print(f"GPU name: {torch.cuda.get_device_name(gpu_id)}")

#     dataset = load_dataset_from_config(config)
#     tokenized_dataset = utils.tokenize_dataset(dataset, tokenizer, config)
#     train_pool, test_pool = utils.split_train_test_dataset(tokenized_dataset, config)

#     train_losses = utils.compute_sample_losses(
#         model,
#         train_pool,
#         data_collator,
#         device,
#         config
#     )
#     test_losses = utils.compute_sample_losses(
#         model,
#         test_pool,
#         data_collator,
#         device,
#         config
#     )


#     easy_train, medium_train, hard_train = utils.split_easy_medium_hard(
#         train_pool,
#         train_losses,
#         config["diagnostic"]["alpha_easy"],
#         config["diagnostic"]["alpha_hard"]
#     )

#     easy_test, medium_test, hard_test = utils.split_easy_medium_hard(
#         test_pool,
#         test_losses,
#         config["diagnostic"]["alpha_easy"],
#         config["diagnostic"]["alpha_hard"]
#     )

#     mixture_dataset = utils.create_mixture2_dataset(
#         easy_train,
#         medium_train,
#         hard_train,
#         config["diagnostic"]["alpha_train"],
#         config["diagnostic"]["beta_train"],
#         config["diagnostic"]["train_size"]
#     )

#     train_dataset, val_dataset, train_dataloader, val_dataloader = utils.create_mixture_train_val_loaders(
#         mixture_dataset,
#         data_collator,
#         config
#     )

#     print(f"Dataset size: train={len(train_dataset)}, val={len(val_dataset)}", flush=True)
#     print(
#         f"Easy={len(easy_train)} | Medium={len(medium_train)} | Hard={len(hard_train)}",
#         flush=True
#     )

#     start_time = time.time()

#     results = utils.train_mixture_rho(
#         train_dataloader,
#         val_dataloader,
#         easy_test,
#         medium_test,
#         hard_test,
#         data_collator,
#         device,
#         config
#     )

#     end_time = time.time()
#     training_time = end_time - start_time

#     print(f"Total training time: {training_time:.2f} seconds", flush=True)
#     print(f"Total training time: {training_time/60:.2f} minutes", flush=True)

#     if device.type == "cuda":
#         max_memory = torch.cuda.max_memory_allocated() / 1024**3
#         print(f"Max GPU memory allocated: {max_memory:.2f} GB", flush=True)

#     utils.plot_multi_model_performance(results, config)
#     utils.plot_multi_model_diagnostic_rho(results, config)
#     utils.plot_multi_model_histograms(results, config)

# def run_training_lambd(config):
#     device = utils.get_device()
#     model, tokenizer, data_collator = utils.setup_model_and_tokenizer(config, device)

#     if device.type == "cuda":
#         gpu_id = torch.cuda.current_device()
#         torch.cuda.reset_peak_memory_stats()
#         print(f"GPU id: {gpu_id}")
#         print(f"GPU name: {torch.cuda.get_device_name(gpu_id)}")

#     dataset = load_dataset_from_config(config)
#     tokenized_dataset = utils.tokenize_dataset(dataset, tokenizer, config)
#     train_pool, test_pool = utils.split_train_test_dataset(tokenized_dataset, config)

#     train_losses = utils.compute_sample_losses(
#         model,
#         train_pool,
#         data_collator,
#         device,
#         config
#     )
#     test_losses = utils.compute_sample_losses(
#         model,
#         test_pool,
#         data_collator,
#         device,
#         config
#     )

#     easy_train, medium_train, hard_train = utils.split_easy_medium_hard(
#         train_pool,
#         train_losses,
#         config["diagnostic"]["alpha_easy"],
#         config["diagnostic"]["alpha_hard"]
#     )

#     easy_test, medium_test, hard_test = utils.split_easy_medium_hard(
#         test_pool,
#         test_losses,
#         config["diagnostic"]["alpha_easy"],
#         config["diagnostic"]["alpha_hard"]
#     )

#     mixture_dataset = utils.create_mixture2_dataset(
#         easy_train,
#         medium_train,
#         hard_train,
#         config["diagnostic"]["alpha_train"],
#         config["diagnostic"]["beta_train"],
#         config["diagnostic"]["train_size"]
#     )

#     train_dataset, val_dataset, train_dataloader, val_dataloader = utils.create_mixture_train_val_loaders(
#         mixture_dataset,
#         data_collator,
#         config
#     )

#     print(f"Dataset size: train={len(train_dataset)}, val={len(val_dataset)}", flush=True)
#     print(
#         f"Easy={len(easy_train)} | Medium={len(medium_train)} | Hard={len(hard_train)}",
#         flush=True
#     )

#     start_time = time.time()

#     results = utils.train_mixture_lambd(
#         train_dataloader,
#         val_dataloader,
#         easy_test,
#         medium_test,
#         hard_test,
#         data_collator,
#         device,
#         config
#     )

#     end_time = time.time()
#     training_time = end_time - start_time

#     print(f"Total training time: {training_time:.2f} seconds", flush=True)
#     print(f"Total training time: {training_time/60:.2f} minutes", flush=True)

#     if device.type == "cuda":
#         max_memory = torch.cuda.max_memory_allocated() / 1024**3
#         print(f"Max GPU memory allocated: {max_memory:.2f} GB", flush=True)

#     utils.plot_multi_model_performance(results, config)
#     utils.plot_multi_model_diagnostic_lambd(results, config)
#     utils.plot_multi_model_histograms(results, config)

# def run_training_lambd2(config):
#     device = utils.get_device()
#     model, tokenizer, data_collator = utils.setup_model_and_tokenizer(config, device)

#     if device.type == "cuda":
#         gpu_id = torch.cuda.current_device()
#         torch.cuda.reset_peak_memory_stats()
#         print(f"GPU id: {gpu_id}")
#         print(f"GPU name: {torch.cuda.get_device_name(gpu_id)}")

#     dataset = load_dataset_from_config(config)
#     tokenized_dataset = utils.tokenize_dataset(dataset, tokenizer, config)
#     train_pool, close = utils.split_train_test_dataset(tokenized_dataset, config)

#     mid_dataset = load_dataset_from_subconfig(
#     config["diagnostic"]["mid_dataset"],
#     config["training"]
#     )

#     far_dataset = load_dataset_from_subconfig(
#     config["diagnostic"]["far_dataset"],
#     config["training"]
#     )

#     mid = utils.tokenize_dataset(mid_dataset, tokenizer, config)
#     far = utils.tokenize_dataset(far_dataset, tokenizer, config)

#     train_losses = utils.compute_sample_losses(
#         model,
#         train_pool,
#         data_collator,
#         device,
#         config
#     )


#     easy_train, medium_train, hard_train = utils.split_easy_medium_hard(
#         train_pool,
#         train_losses,
#         config["diagnostic"]["alpha_easy"],
#         config["diagnostic"]["alpha_hard"]
#     )


#     mixture_dataset = utils.create_mixture2_dataset(
#         easy_train,
#         medium_train,
#         hard_train,
#         config["diagnostic"]["alpha_train"],
#         config["diagnostic"]["beta_train"],
#         config["diagnostic"]["train_size"]
#     )

#     train_dataset, val_dataset, train_dataloader, val_dataloader = utils.create_mixture_train_val_loaders(
#         mixture_dataset,
#         data_collator,
#         config
#     )

#     print(f"Dataset size: train={len(train_dataset)}, val={len(val_dataset)}", flush=True)
#     print(
#         f"Easy={len(easy_train)} | Medium={len(medium_train)} | Hard={len(hard_train)}",
#         flush=True
#     )

#     start_time = time.time()

#     results = utils.train_mixture_lambd2(
#         train_dataloader,
#         val_dataloader,
#         close,
#         mid,
#         far,
#         data_collator,
#         device,
#         config
#     )

#     end_time = time.time()
#     training_time = end_time - start_time

#     print(f"Total training time: {training_time:.2f} seconds", flush=True)
#     print(f"Total training time: {training_time/60:.2f} minutes", flush=True)

#     if device.type == "cuda":
#         max_memory = torch.cuda.max_memory_allocated() / 1024**3
#         print(f"Max GPU memory allocated: {max_memory:.2f} GB", flush=True)

#     utils.plot_multi_model_performance(results, config)
#     utils.plot_multi_model_diagnostic_lambd(results, config)
#     utils.plot_multi_model_histograms(results, config)


def run_training(config):
    device = utils.get_device()
    model, tokenizer, data_collator = utils.setup_model_and_tokenizer(config, device)

    if device.type == "cuda":
        gpu_id = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU id: {gpu_id}")
        print(f"GPU name: {torch.cuda.get_device_name(gpu_id)}")

    # Load dataset
    raw_dataset = load_dataset_from_config(config)
    train_raw, test_raw = utils.split_train_test_dataset(raw_dataset, config)


    # Tokenize clean PG-19 for easy/medium/hard training split
    train_pool = utils.tokenize_dataset(train_raw, tokenizer, config)

    print("Computing pretrained sample losses...", flush=True)
    pretrained_losses = utils.compute_sample_losses(model, train_pool, data_collator, device, config)

    easy_dataset, medium_dataset, hard_dataset = utils.split_easy_medium_hard(train_pool, pretrained_losses, 
        alpha_easy=config["diagnostic"]["alpha_easy"], alpha_hard=config["diagnostic"]["alpha_hard"])

    # Training mixture
    train_mixture = utils.create_three_way_mixture_dataset(easy_dataset, medium_dataset, hard_dataset,
        alpha=config["diagnostic"]["train_alpha"],
        beta=config["diagnostic"]["train_beta"],
        size=config["diagnostic"]["train_size"],
        seed=config["training"]["seed"])
    
    train_dataset, val_dataset, train_dataloader, val_dataloader = utils.create_mixture_train_val_loaders(train_mixture, data_collator, config)

    # Shift datasets: clean reference / close / mid / far
    clean_ref_raw, close_raw, mid_raw, far_raw = utils.create_shift_datasets(test_raw, config, seed=config["training"]["seed"])
    clean_ref_dataset = utils.tokenize_dataset(clean_ref_raw, tokenizer, config)
    close_dataset = utils.tokenize_dataset(close_raw, tokenizer, config)
    mid_dataset = utils.tokenize_dataset(mid_raw, tokenizer, config)
    far_dataset = utils.tokenize_dataset(far_raw, tokenizer, config)

    # Training
    start_time = time.time()
    results = {}
    for method_name in config["methods"]:
        results[method_name] = utils.train_method(method_name, train_dataloader, val_dataloader, clean_ref_dataset, close_dataset,
                mid_dataset, far_dataset, data_collator, device, config)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds", flush=True)
    print(f"Total training time: {training_time/60:.2f} minutes", flush=True)

    if device.type == "cuda":
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Max GPU memory allocated: {max_memory:.2f} GB", flush=True)
    
    utils.plot_training_curves(results, config)
    utils.plot_shift_visual(results, config)
    utils.plot_robustness_boxplots(results, config)


    print("Done.", flush=True)

