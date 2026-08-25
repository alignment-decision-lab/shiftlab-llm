import os
import torch
import utils
import pandas as pd
import torch.optim as optim
from shiftlab.data.load_datasets import load_dataset_from_subconfig
import matplotlib.pyplot as plt
import time
import copy

MODEL_NAME = "gpt2-medium"

TRAINING_CONFIG = {
    "seed": 42,
    "context_length": 512,
    "max_tokens": 5_120_000,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "weight_decay": 0.0,
    "val_split_ratio": 0.1,
    "max_epochs": 10,
    "gamma": 0.7,
    "rho": 0.0,
    "eval_every_optimizer_steps": 500,
    "eval_first_epoch_only": False,
}

datasets_config = {
      "PG-19": {     
      "dataset": {
          "type": "hf_text",
          "name": "emozilla/pg19",
          "split": "train",
          "text_column": "text",
          "streaming": True,
      },
      },

     "PubMed Abstracts": {
     "dataset": {
         "type": "hf_text",
         "name": "timaeus/pile-pubmed_abstracts",
         "split": "train",
         "text_column": "text",
         "streaming": True,
     },
     },

     "GitHub": {     
     "dataset": {
         "type": "hf_text",
         "name": "timaeus/pile-github",
         "split": "train",
         "text_column": "text",
         "streaming": True,
     },
     },

    "Ubuntu IRC": {
        "dataset": {
            "type": "hf_text",
            "name": "common-pile/ubuntu_irc",
            "split": "train",
            "text_column": "text",
            "streaming": True,
    },
    },

    "YouTube Subtitles": {
        "dataset": {
            "type": "hf_text",
            "name": "suolyer/pile_youtubesubtitles",
            "split": "validation",
            "text_column": "text",
            "streaming": True,
    },
    },
}



def collect_erm_configs(datasets_config):
    """Create one complete ERM configuration per deployment dataset."""
    one_configs = []

    for dataset_name, dataset_config in datasets_config.items():
        cfg = copy.deepcopy(dataset_config)

        cfg["dataset_name"] = dataset_name
        cfg["models"] = {
            "name": MODEL_NAME,
        }
        cfg["training"] = copy.deepcopy(TRAINING_CONFIG)

        one_configs.append(cfg)

    return one_configs


def plot_model_training_curves(step_history, epoch_history, dataset_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    step_df = pd.DataFrame(step_history)
    epoch_df = pd.DataFrame(epoch_history)

    # Loss vs epochs
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
    plt.title(f"{dataset_name} — ERM loss vs epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "loss_vs_epochs.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Loss vs optimizer steps
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
    plt.title(f"{dataset_name} — ERM loss vs optimizer steps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            "loss_vs_optimizer_steps.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Loss vs cumulative tokens
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
    plt.title(f"{dataset_name} — ERM loss vs tokens")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            "loss_vs_cumulative_tokens.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def train_one_erm_model(one_config, device):
    dataset_cfg = one_config["dataset"]
    training_cfg = one_config["training"]
    dataset_name = one_config["dataset_name"]

    print(f"\n===== Training oracle ERM on {dataset_name} =====", flush=True)

    seed = training_cfg["seed"]
    utils.set_seed(seed)

    # --------------------------------------------------------
    # MODEL / TOKENIZER
    # --------------------------------------------------------

    model, tokenizer, data_collator = utils.setup_model_and_tokenizer(one_config, device)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    raw_dataset = load_dataset_from_subconfig(dataset_cfg, training_cfg)

    tokenized_dataset, dataset_stats = (
        utils.tokenize_and_group_with_token_budget(
            dataset=raw_dataset,
            tokenizer=tokenizer,
            config=one_config,
        )
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    (
        train_optim_dataloader,
        train_dataloader,
        val_dataloader,
        train_step_eval_dataloader,
    ) = utils.create_training_dataloaders(
        tokenized_dataset=tokenized_dataset,
        data_collator=data_collator,
        training_config=training_cfg,
    )

    num_train_sequences = len(train_dataloader.dataset)
    num_val_sequences = len(val_dataloader.dataset)

    print(
        "=============================\n"
        f"Train sequences:         {num_train_sequences:,}\n"
        f"Validation sequences:    {num_val_sequences:,}\n"
        f"Train tokens / epoch:    "
        f"{num_train_sequences * training_cfg['context_length']:,}\n"
        f"Validation tokens:       "
        f"{num_val_sequences * training_cfg['context_length']:,}\n"
        "=============================\n",
        flush=True,
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    # --------------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------------

    dataset_dir = os.path.join(
        "outputs",
        "erm_baselines",
        dataset_name.replace(" ", "_"),
    )

    analysis_dir = os.path.join(dataset_dir, "training_analysis")
    model_dir = os.path.join(dataset_dir, "model")

    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --------------------------------------------------------
    # TRAINING -- FIXED 10 EPOCHS
    # --------------------------------------------------------

    step_history, epoch_history = utils.train_with_step_logging(
        model=model,
        train_optim_dataloader=train_optim_dataloader,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_step_eval_dataloader=train_step_eval_dataloader,
        optimizer=optimizer,
        device=device,
        accumulation_steps=training_cfg[
            "gradient_accumulation_steps"
        ],
        eval_every_optimizer_steps=training_cfg[
            "eval_every_optimizer_steps"
        ],
        eval_first_epoch_only=training_cfg[
            "eval_first_epoch_only"
        ],
        max_epochs=training_cfg["max_epochs"],
    )

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(
        f"Final ERM model saved to: {model_dir}",
        flush=True,
    )

    # --------------------------------------------------------
    # SAVE HISTORIES
    # --------------------------------------------------------

    step_df = pd.DataFrame(step_history)
    epoch_df = pd.DataFrame(epoch_history)

    step_history_path = os.path.join(analysis_dir, "step_history.csv")
    epoch_history_path = os.path.join(analysis_dir, "epoch_history.csv")

    step_df.to_csv(step_history_path, index=False)
    epoch_df.to_csv(epoch_history_path, index=False)

    # --------------------------------------------------------
    # TRAINING CURVES
    # --------------------------------------------------------

    plot_model_training_curves(
        step_history=step_history,
        epoch_history=epoch_history,
        dataset_name=dataset_name,
        output_dir=analysis_dir,
    )

    # --------------------------------------------------------
    # SUMMARY / METADATA
    # --------------------------------------------------------

    last_epoch = epoch_history[-1]

    summary = {
        "model_name": one_config["models"]["name"],
        "dataset_name": dataset_name,
        "model_dir": model_dir,

        "seed": training_cfg["seed"],
        "context_length": training_cfg["context_length"],
        "max_tokens": training_cfg["max_tokens"],
        "batch_size": training_cfg["batch_size"],
        "gradient_accumulation_steps":
            training_cfg["gradient_accumulation_steps"],
        "effective_batch_size":
            training_cfg["batch_size"]
            * training_cfg["gradient_accumulation_steps"],

        "max_epochs": training_cfg["max_epochs"],

        "num_documents_used":
            dataset_stats["num_documents_used"],
        "raw_tokens_seen":
            dataset_stats["raw_tokens_seen"],
        "num_sequences":
            dataset_stats["num_sequences"],
        "effective_tokens":
            dataset_stats["effective_tokens"],

        "num_train_sequences": num_train_sequences,
        "num_val_sequences": num_val_sequences,

        "final_optimizer_step":
            last_epoch["optimizer_step"],
        "final_tokens_seen":
            last_epoch["cumulative_tokens_seen"],

        "final_train_loss":
            last_epoch["train_loss"],
        "final_val_loss":
            last_epoch["val_loss"],
        "final_train_ppl":
            last_epoch["train_ppl"],
        "final_val_ppl":
            last_epoch["val_ppl"],
        "final_train_acc":
            last_epoch["train_acc"],
        "final_val_acc":
            last_epoch["val_acc"],

        "step_history_path": step_history_path,
        "epoch_history_path": epoch_history_path,
    }

    summary_path = os.path.join(
        analysis_dir,
        "training_summary.csv",
    )

    pd.DataFrame([summary]).to_csv(
        summary_path,
        index=False,
    )

    torch.save(
        {
            **summary,
            "dataset_config": dataset_cfg,
            "training_config": training_cfg,
        },
        os.path.join(model_dir, "metadata.pth"),
    )

    # --------------------------------------------------------
    # CLEAN GPU
    # --------------------------------------------------------

    del model
    del optimizer

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary   

def erm_model_exists(dataset_name):
    """Check whether a complete saved ERM model already exists."""

    dataset_dir = os.path.join("outputs", "erm_baselines", dataset_name.replace(" ", "_"))

    model_dir = os.path.join(dataset_dir, "model")

    model_exists = (
        os.path.isfile(
            os.path.join(
                model_dir,
                "config.json",
            )
        )
        and (
            os.path.isfile(
                os.path.join(
                    model_dir,
                    "model.safetensors",
                )
            )
            or os.path.isfile(
                os.path.join(
                    model_dir,
                    "pytorch_model.bin",
                )
            )
        )
    )

    return model_exists

def load_existing_erm_summary(dataset_name):
    """Load the training summary of an already trained ERM model."""

    summary_path = os.path.join(
        "outputs",
        "erm_baselines",
        dataset_name.replace(" ", "_"),
        "training_analysis",
        "training_summary.csv",
    )

    if not os.path.isfile(summary_path):
        return None

    summary_df = pd.read_csv(summary_path)

    if len(summary_df) == 0:
        return None

    return summary_df.iloc[0].to_dict()

def run_erm_baselines(datasets_config, device):
    """Train one oracle ERM model for each deployment dataset."""

    save_path = "outputs/erm_baselines"
    os.makedirs(save_path, exist_ok=True)

    one_configs = collect_erm_configs(datasets_config)
    results = []

    for one_config in one_configs:
        dataset_name = one_config["dataset_name"]

        # ----------------------------------------------------
        # REUSE EXISTING MODEL
        # ----------------------------------------------------

        if erm_model_exists(dataset_name):
            model_dir = os.path.join(
                save_path,
                dataset_name.replace(" ", "_"),
                "model",
            )
            print(
                f"\n[{dataset_name}] Existing ERM model found: "
                f"{model_dir}",
                flush=True,
            )
            print(
                "Skipping training.",
                flush=True,
            )
            existing_summary = (
                load_existing_erm_summary(
                    dataset_name
                )
            )
            if existing_summary is not None:
                results.append(existing_summary)

            else:
                print(
                    f"WARNING: model exists for "
                    f"{dataset_name}, but no "
                    "training_summary.csv was found.",
                    flush=True,
                )
            continue

        # ----------------------------------------------------
        # TRAIN MODEL
        # ----------------------------------------------------

        result = train_one_erm_model(one_config, device)
        results.append(result)

        # ----------------------------------------------------
        # SAVE GLOBAL METADATA AFTER EACH MODEL
        # ----------------------------------------------------

        df = pd.DataFrame(results)
        csv_path = os.path.join(save_path, "oracle_erm_models_metadata.csv")
        df.to_csv(csv_path, index=False)

        print(
            f"Metadata updated: {csv_path}",
            flush=True,
        )

    # --------------------------------------------------------
    # FINAL METADATA
    # --------------------------------------------------------

    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(save_path, "oracle_erm_models_metadata.csv")
        df.to_csv(csv_path, index=False)

        print(
            f"\nFinal metadata saved at: "
            f"{csv_path}",
            flush=True,
        )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    run_erm_baselines(datasets_config, device)
    end_time = time.time()
    print(f"Done. Total time: {end_time - start_time:.2f} seconds --> {(end_time - start_time)/60:.2f} minutes")
