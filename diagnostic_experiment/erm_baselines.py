import os
import torch
import utils
import pandas as pd
import torch.optim as optim
from shiftlab.data.load_datasets import load_dataset_from_subconfig
import copy
import plotting
import time
from torch.utils.data import DataLoader
import math
from torch.optim.lr_scheduler import LambdaLR


datasets_config = {
      "PG-19": {     # This dataset's configurations have been verified and thus should not change.
      "dataset": {
          "type": "hf_text",
          "name": "emozilla/pg19",
          "split": "train",
          "text_column": "text",
          "streaming": True,
      },
      "training": {
          "batch_size": 4,
          "gradient_accumulation_steps": 4,
          "learning_rate": 0.00002,
          "context_length": 512,
          "weight_decay": 0.0,
          "dataset_size": 10000,
          "seed": 42,
          "val_split_ratio": 0.1,
          "gamma": 0.7,
          "rho": 0.0
      }},

     "PubMed Abstracts": {
     "dataset": {
         "type": "hf_text",
         "name": "timaeus/pile-pubmed_abstracts",
         "split": "train",
         "text_column": "text",
         "streaming": True,
     },
     "training": {
         "batch_size": 4,
         "gradient_accumulation_steps": 4,
         "learning_rate": 0.00002,
         "context_length": 512,
         "weight_decay": 0.0,
         "dataset_size": 50000,  # 50 000 is taking too long to train, maybe try 30 000.

         "seed": 42,
         "val_split_ratio": 0.1,
         "gamma": 0.7,
         "rho": 0.0
     }},

     "GitHub": {     
     "dataset": {
         "type": "hf_text",
         "name": "timaeus/pile-github",
         "split": "train",
         "text_column": "text",
         "streaming": True,
     },
     "training": {
         "batch_size": 4,
         "gradient_accumulation_steps": 4,
         "learning_rate": 0.00002,
         "context_length": 512,
         "weight_decay": 0.0,
         "dataset_size": 10000,

         "seed": 42,
         "val_split_ratio": 0.1,
         "gamma": 0.7,
         "rho": 0.0
     }},

    "Ubuntu IRC": {
        "dataset": {
            "type": "hf_text",
            "name": "common-pile/ubuntu_irc",
            "split": "train",
            "text_column": "text",
            "streaming": True,

            # Reserve everything after the first 10,000 raw documents
            # for the held-out deployment evaluation.
            "dataset_offset": 0,
        },
        "training": {
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.00002,
            "context_length": 512,
            "weight_decay": 0.0,

            # Raw documents used to construct the oracle training corpus.
            "dataset_size": 10000,

            "seed": 42,
            "val_split_ratio": 0.1,
            "gamma": 0.7,
            "rho": 0.0,
        },
    },

    "YouTube Subtitles": {
        "dataset": {
            "type": "hf_text",
            "name": "suolyer/pile_youtubesubtitles",
            "split": "validation",
            "text_column": "text",
            "streaming": True,

            # Train the oracle only on the first 500 raw transcripts.
            "dataset_offset": 0,
        },
        "training": {
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.00002,
            "context_length": 512,
            "weight_decay": 0.0,

            # Keep documents 500 onward for deployment evaluation.
            "dataset_size": 500,

            "seed": 42,
            "val_split_ratio": 0.1,
            "gamma": 0.7,
            "rho": 0.0,
        },
    },
}



def collect_erm_configs(datasets_config):
    """Create one complete ERM configuration per dataset."""
    one_configs = []

    for dataset_name, dataset_config in datasets_config.items():
        cfg = copy.deepcopy(dataset_config)

        cfg["dataset_name"] = dataset_name
        cfg["models"] = {
            "name": "gpt2-medium",
        }

        one_configs.append(cfg)

    return one_configs


def train_one_erm_model(one_config, device, max_epochs=30, patience=3, min_delta= 0.001):
    """ Train an ERM model for a given dataset using early stopping. """
    dataset_cfg = one_config["dataset"]
    training_cfg = one_config["training"]
    dataset_name = one_config["dataset_name"]
    print(f"\n ===== Training ERM with dataset {dataset_name} =====", flush=True)
    seed = training_cfg.get("seed", 42)
    utils.set_seed(seed)

    model, tokenizer, data_collator = utils.setup_model_and_tokenizer(one_config, device)
    model_name = f"{one_config['models']['name']}_ERM_dataset_{dataset_name}"

    dataset = load_dataset_from_subconfig(dataset_cfg, training_cfg)
    tokenized_dataset = utils.tokenize_and_group_dataset(dataset, tokenizer, one_config)
    train_dataset, val_dataset, train_dataloader, val_dataloader = (
        utils.create_mixture_train_val_loaders(
            tokenized_dataset,
            data_collator,
            one_config
        )
    )
    train_eval_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=training_cfg["batch_size"],
        collate_fn=data_collator,
    )


    optimizer = optim.AdamW(
            model.parameters(),
            lr=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg.get("weight_decay", 0.0))
        )
    accumulation_steps = training_cfg.get("gradient_accumulation_steps", 1)

    
    final_lr_ratio = training_cfg.get("final_lr_ratio") # Final learning rate as a fraction of the initial learning rate.
    scheduler = None
    if final_lr_ratio is not None:
        num_optimizer_step_per_epoch = math.ceil(len(train_dataloader) / accumulation_steps) # Number of updates per epoch.
        num_training_steps = max_epochs * num_optimizer_step_per_epoch # Maximum number of updates for the entire training process.
        def linear_decay(step):
            """Linear decay function for learning rate scheduling."""
            progress = min(step / num_training_steps, 1.0) # Progress is the fraction of training completed.
            return 1.0 - progress * (1.0 - final_lr_ratio) # Linear decay from 1.0 to final_lr_ratio over the course of training.
        scheduler = LambdaLR(optimizer, lr_lambda=linear_decay)

    dataset_dir = os.path.join(
        "outputs/erm_baselines",
        dataset_name.replace(" ", "_"),
    )

    analysis_dir = os.path.join(
        dataset_dir,
        "training_analysis",
    )

    model_dir = os.path.join(
         dataset_dir,
         "model",
     )  ###

    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)  ###

    optim_losses = []
    train_losses = []
    val_losses = []
    train_ppls = []
    val_ppls = []
    train_accs = []
    val_accs = []
    learning_rates = []
    cumulative_steps = []
    cumulative_tokens = []
    total_steps = 0
    total_tokens = 0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopping_reason = "max_epochs_reached"

    for epoch in range(max_epochs):
        optim_loss, num_steps, num_tokens = utils.train_one_epoch_accumulated(model, train_dataloader, optimizer, device, accumulation_steps, scheduler)
        optim_losses.append(optim_loss)
        total_steps += num_steps
        total_tokens += num_tokens
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        train_loss, train_ppl, train_acc, _ = utils.evaluation(model, train_eval_dataloader, device=device)
        val_loss, val_ppl, val_acc, _ = utils.evaluation(model, val_dataloader, device=device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        cumulative_steps.append(total_steps)
        cumulative_tokens.append(total_tokens)

        improvement = best_val_loss - val_loss

        if improvement > min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            model.save_pretrained(model_dir)  ###
            tokenizer.save_pretrained(model_dir)  ###

            print(
                 f"New best model saved at epoch {epoch + 1} "
                 f"with validation loss {val_loss:.4f}.",
                 flush=True,
             )  ###
        else:
            epochs_without_improvement += 1
        
        print(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"lr: {current_lr:.2e} | "
            f"train loss: {train_loss:.4f} | "
            f"train ppl: {train_ppl:.4f} | "
            f"optim loss: {optim_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val ppl: {val_ppl:.4f} | "
            f"best val loss: {min(best_val_loss, val_loss):.4f}",
            flush=True,
        )
        
        if epochs_without_improvement >= patience:
            stopping_reason = "early_stopping"
            print(f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}.", flush=True)
            break
    
    epochs_run = len(train_losses)

    history_df = pd.DataFrame({
        "epoch": range(1, epochs_run + 1),
        "cumulative_steps": cumulative_steps,
        "cumulative_tokens": cumulative_tokens,
        "learning_rate": learning_rates,
        "optim_loss": optim_losses,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_ppl": train_ppls,
        "val_ppl": val_ppls,
        "train_acc": train_accs,
        "val_acc": val_accs,
    })
    history_path = os.path.join(analysis_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)

    results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_ppls": train_ppls,
        "val_ppls": val_ppls,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "cumulative_steps": cumulative_steps,
        "cumulative_tokens": cumulative_tokens,
    }
    training_curves_paths = plotting.plot_training_loss_curves(
        {dataset_name: results},
        output_dir=analysis_dir,
        filename_prefix="training_curves",
        title_prefix="ERM -- ",
    )

    summary = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_size": training_cfg.get("dataset_size"),
        "batch_size": training_cfg.get("batch_size"),
        "context_length": training_cfg.get("context_length"),
        "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps"),
        "effective_batch_size": training_cfg.get("batch_size") * training_cfg.get("gradient_accumulation_steps", 1),
        "learning_rate": training_cfg.get("learning_rate"),
        "weight_decay": training_cfg.get("weight_decay"),
        "seed": seed,

        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val_loss,
        "best_val_ppl": val_ppls[best_epoch],
        "best_val_acc": val_accs[best_epoch],
        "stopping_reason": stopping_reason,
        "history_path": history_path,
        "training_curves_by_epoch_path": training_curves_paths["epoch"],
        "training_curves_by_step_path": training_curves_paths["step"],
        "training_curves_by_token_path": training_curves_paths["token"],
        "model_dir": model_dir,  ###
    }
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(analysis_dir, "training_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    metadata_path = os.path.join(
             model_dir,
             "metadata.pth",
         )  ###

    torch.save(
             {
                 **summary,
                 "dataset_config": dataset_cfg,
                 "training_config": training_cfg,
             },
             metadata_path,
         )  ###

    print(
            f"Training analysis saved at: {analysis_dir}",
            flush=True,
        )

    print(
             f"Best model and tokenizer saved at: {model_dir}",
             flush=True,
         )  ###

    return summary
    

def run_erm_baselines(datasets_config, device):
    """Train one oracle ERM model for each deployment dataset."""
    save_path = "outputs/erm_baselines"
    os.makedirs(save_path, exist_ok=True)

    one_configs = collect_erm_configs(datasets_config)
    results = []

    for one_config in one_configs:

        result = train_one_erm_model(one_config, device, max_epochs=30, patience=3, min_delta= 0.001)

        results.append(result)

    df = pd.DataFrame(results)

    csv_path = os.path.join(save_path, "oracle_erm_models_metadata.csv")
    df.to_csv(csv_path, index=False)

    # print(" ERM Model bank training completed.", flush=True)
    print(f"Metadata saved at: {csv_path}", flush=True)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    run_erm_baselines(datasets_config, device)
    end_time = time.time()
    print(f"Done. Total time: {end_time - start_time:.2f} seconds --> {(end_time - start_time)/60:.2f} minutes")
