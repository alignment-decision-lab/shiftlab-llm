import argparse
import yaml
import torch
import math
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import matplotlib.pyplot as plt
import os



# ------- GENERAL SETTINGS -------

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True) # in the terminal
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    return config

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()} 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------- MODEL, TOKENIZER AND DATALOADER SETTINGS -------

def setup_model_and_tokenizer(config, device):
    model = AutoModelForCausalLM.from_pretrained(
        config["models"]["name"]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["name"]
    )
    
    model.to(device)

    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return model, tokenizer, data_collator


# ------- DATASETS UTILITIES -------

def split_train_test_dataset(dataset, config):
    train_pool_size = config["diagnostic"]["train_pool_size"]
    test_pool_size = config["diagnostic"]["test_pool_size"]

    assert train_pool_size + test_pool_size <= len(dataset)

    train_pool = dataset.select(range(train_pool_size))
    test_pool = dataset.select(range(train_pool_size, train_pool_size + test_pool_size))

    return train_pool, test_pool

def add_ocr_noise(example, p, seed=None):

    if seed is not None:
        random.seed(seed)

    replacements = {
        "o": "0",
        "i": "l",
        "e": "c",
        "a": "@",
        "s": "5",
        "œ": "oe",
        "l": "1",
        "O": "0"
    }
    text = example["text"]
    noisy_text = ""

    for char in text:
        if char in replacements and random.random() < p:
            noisy_text += replacements[char]
        else:
            noisy_text += char

    example["text"] = noisy_text
    return example

def create_shift_datasets(dataset, config, seed):
    total_size = len(dataset)
    dataset = dataset.select(range(total_size))

    quarter = total_size // 4

    clean_reference_dataset = dataset.select(range(0, quarter))
    close_clean_dataset = dataset.select(range(quarter, 2 * quarter))
    mid_clean_dataset = dataset.select(range(2 * quarter, 3 * quarter))
    far_clean_dataset = dataset.select(range(3 * quarter, total_size))

    close_dataset = close_clean_dataset.map(lambda x: add_ocr_noise(x, p=config["diagnostic"]["close_noise"], seed=seed))
    mid_dataset = mid_clean_dataset.map(lambda x: add_ocr_noise(x, p=config["diagnostic"]["mid_noise"], seed=seed))
    far_dataset = far_clean_dataset.map(lambda x: add_ocr_noise(x, p=config["diagnostic"]["far_noise"], seed=seed))

    return clean_reference_dataset, close_dataset, mid_dataset, far_dataset

def tokenize_and_group_dataset(
    dataset,
    tokenizer,
    config,
):
    context_length = config["training"]["context_length"]

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], add_special_tokens=False)

        tokenized["input_ids"] = [input_ids + [tokenizer.eos_token_id] for input_ids in tokenized["input_ids"]]

        tokenized["attention_mask"] = [attention_mask + [1] for attention_mask in tokenized["attention_mask"]]

        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names,)

    def group_texts(examples):
        concatenated = {key: sum(examples[key], []) for key in examples.keys()}

        total_length = len(concatenated["input_ids"])
        total_length = (total_length // context_length) * context_length

        result = {
            key: [
                values[i:i + context_length]
                for i in range(
                    0,
                    total_length,
                    context_length,
                )
            ]
            for key, values in concatenated.items()
        }

        return result

    grouped_dataset = tokenized_dataset.map(group_texts, batched=True)

    return grouped_dataset


def tokenize_and_group_with_token_budget(
    dataset,
    tokenizer,
    config,
    add_eos_between_documents=True,
):
    """
    Tokenize a dataset progressively and build fixed-length token blocks
    until a predefined token budget is reached.

    This function is designed to work with both Hugging Face Dataset
    and IterableDataset objects.

    Parameters
    ----------
    dataset:
        Hugging Face Dataset or IterableDataset.

    tokenizer:
        Hugging Face tokenizer.

    config:
        Configuration dictionary containing:
            config["dataset"]["text_column"]
            config["training"]["context_length"]
            config["training"]["max_tokens"]

    add_eos_between_documents:
        If True, insert one EOS token between consecutive documents.

    Returns
    -------
    Dataset
        A finite Hugging Face Dataset containing fixed-length sequences
        with columns:
            - input_ids
            - attention_mask

    Notes
    -----
    - Only complete context windows are returned.
    - The effective number of retained tokens is therefore:
          n_sequences * context_length
    - The function stops as soon as enough tokens have been collected
      to satisfy the requested token budget.
    """

    dataset_cfg = config["dataset"]
    training_cfg = config["training"]

    text_column = dataset_cfg.get("text_column", "text")
    context_length = int(training_cfg["context_length"])
    max_tokens = int(training_cfg["max_tokens"])

    if context_length <= 0:
        raise ValueError(
            "context_length must be a positive integer."
        )

    if max_tokens <= 0:
        raise ValueError(
            "max_tokens must be a positive integer."
        )

    # We only keep complete context windows.
    max_sequences = max_tokens // context_length

    if max_sequences == 0:
        raise ValueError(
            f"max_tokens={max_tokens} is smaller than "
            f"context_length={context_length}."
        )

    # Exact number of tokens that will actually be used.
    usable_token_budget = max_sequences * context_length

    input_blocks = []
    attention_blocks = []

    # Temporary buffer containing tokens not yet assigned
    # to a complete context window.
    token_buffer = []
    buffer_start = 0

    num_documents = 0
    num_raw_tokens = 0
    num_sequences = 0

    eos_token_id = tokenizer.eos_token_id

    if add_eos_between_documents and eos_token_id is None:
        raise ValueError(
            "add_eos_between_documents=True but the tokenizer "
            "does not define an eos_token_id."
        )

    for example in dataset:

        if num_sequences >= max_sequences:
            break

        text = example.get(text_column)

        # Normalize possible non-string values.
        if isinstance(text, list):
            text = " ".join(map(str, text))

        if text is None:
            continue

        text = str(text).strip()

        if len(text) <= 5:
            continue

        # Exact tokenization of the complete document.
        #
        # verbose=False prevents Hugging Face from printing the
        # "sequence length > model_max_length" warning. The long
        # sequence is NEVER passed to the model: it is immediately
        # split below into context_length blocks.
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]

        if not token_ids:
            continue

        num_documents += 1
        num_raw_tokens += len(token_ids)

        # Separate documents explicitly instead of concatenating
        # unrelated texts with no boundary marker.
        if add_eos_between_documents:
            token_ids.append(eos_token_id)

        token_buffer.extend(token_ids)

        # Consume complete context windows from the buffer.
        while (
            len(token_buffer) - buffer_start >= context_length
            and num_sequences < max_sequences
        ):
            end = buffer_start + context_length

            block = token_buffer[buffer_start:end]

            input_blocks.append(block)
            attention_blocks.append(
                [1] * context_length
            )

            buffer_start = end
            num_sequences += 1

        # Periodically remove already-consumed tokens from memory.
        if buffer_start >= 100_000:
            token_buffer = token_buffer[buffer_start:]
            buffer_start = 0

    effective_tokens = num_sequences * context_length

    if num_sequences < max_sequences:
        print(
            "\nWARNING: The dataset ended before the requested "
            "token budget was reached.",
            flush=True,
        )

    print(
        "\n===== Token-budget dataset construction =====\n"
        f"Documents consumed:       {num_documents:,}\n"
        f"Raw text tokens seen:     {num_raw_tokens:,}\n"
        f"Requested max tokens:     {max_tokens:,}\n"
        f"Usable token budget:      {usable_token_budget:,}\n"
        f"Sequences created:        {num_sequences:,}\n"
        f"Context length:           {context_length:,}\n"
        f"Effective tokens retained:{effective_tokens:,}\n"
        "=============================================\n",
        flush=True,
    )

    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": input_blocks,
            "attention_mask": attention_blocks,
        }
    )

    dataset_stats = {
        "num_documents_used": num_documents,
        "raw_tokens_seen": num_raw_tokens,
        "requested_max_tokens": max_tokens,
        "usable_token_budget": usable_token_budget,
        "num_sequences": num_sequences,
        "context_length": context_length,
        "effective_tokens": effective_tokens,
    }

    return tokenized_dataset, dataset_stats
    

def split_easy_medium_hard(tokenized_dataset, losses, alpha_easy, alpha_hard):
    assert len(tokenized_dataset) == len(losses)
    assert 0 <= alpha_easy <= 1
    assert 0 <= alpha_hard <= 1
    assert 0 <= alpha_easy + alpha_hard < 1

    # Sort dataset:
    indexed_losses = []
    for i in range(len(losses)):
        indexed_losses.append((i, losses[i]))
    indexed_losses.sort(key=lambda x : x[1]) # sort according to the loss from easy to hard.

    n_easy = int(len(losses) * alpha_easy)
    n_medium = int(len(losses) * (1 - alpha_hard))
    easy_samples = indexed_losses[:n_easy]
    medium_samples = indexed_losses[n_easy:n_medium]
    hard_samples = indexed_losses[n_medium:]

    easy_indices = []
    for sample in easy_samples:
        easy_indices.append(sample[0])
    medium_indices = []
    for sample in medium_samples:
        medium_indices.append(sample[0])
    hard_indices = []
    for sample in hard_samples:
        hard_indices.append(sample[0])
    
    easy_dataset = tokenized_dataset.select(easy_indices)
    medium_dataset = tokenized_dataset.select(medium_indices)
    hard_dataset = tokenized_dataset.select(hard_indices)

    return easy_dataset, medium_dataset, hard_dataset

def create_three_way_mixture_dataset(easy_dataset, medium_dataset, hard_dataset, alpha, beta, size, seed):
    assert 0 <= alpha <= 1
    assert 0 <= beta <= 1
    assert 0 <= alpha + beta < 1

    n_easy = int(alpha * size)
    n_medium = int(beta * size)
    n_hard = size - n_easy - n_medium

    assert n_easy <= len(easy_dataset)
    assert n_medium <= len(medium_dataset)
    assert n_hard <= len(hard_dataset)

    random.seed(seed)
    easy_indices = random.sample(range(len(easy_dataset)), n_easy) # choose randomly n_easy indices.
    medium_indices = random.sample(range(len(medium_dataset)), n_medium)
    hard_indices = random.sample(range(len(hard_dataset)), n_hard)

    selected_easy = easy_dataset.select(easy_indices)
    selected_medium = medium_dataset.select(medium_indices)
    selected_hard = hard_dataset.select(hard_indices)

    mixture_dataset = concatenate_datasets([selected_easy, selected_medium, selected_hard]) # Concatenate easy+medium+hard.
    mixture_dataset = mixture_dataset.shuffle(seed=seed) # Mix the dataset.

    return mixture_dataset

def create_training_dataloaders(
    tokenized_dataset,
    data_collator,
    training_config,
    step_eval_size=512,
):
    """
    Create all dataloaders required for fine-tuning experiments.

    Returns
    -------
    train_optim_dataloader:
        Dataloader used for optimization steps.
        It uses the training split with shuffle=True.

    train_dataloader:
        Dataloader used for full training-set evaluation
        (train loss evolution after each epoch).

    val_dataloader:
        Dataloader used for validation evaluation.
        It is a held-out split never used for optimization.

    train_step_eval_dataloader:
        Small fixed subset of the training set used for
        frequent evaluation during the first optimization steps.
        This avoids evaluating the full training set every few steps.
    """

    seed = training_config["seed"]

    # Split dataset into optimization and validation subsets
    split_dataset = tokenized_dataset.train_test_split(
        test_size=training_config["val_split_ratio"],
        seed=seed,
    )

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    generator = torch.Generator()
    generator.manual_seed(seed)

    # -------------------------------------------------
    # Optimization dataloader
    # Used for gradient updates
    # -------------------------------------------------
    train_optim_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        collate_fn=data_collator,
        generator=generator,
    )

    # -------------------------------------------------
    # Full training evaluation dataloader
    # Used to compute train loss after each epoch
    # -------------------------------------------------
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        collate_fn=data_collator,
    )

    # -------------------------------------------------
    # Validation dataloader
    # Held-out data
    # -------------------------------------------------
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        collate_fn=data_collator,
    )

    # -------------------------------------------------
    # Step evaluation dataloader
    # Fixed subset for frequent evaluation
    # -------------------------------------------------
    step_eval_dataset = train_dataset.select(
        range(min(step_eval_size, len(train_dataset)))
    )

    train_step_eval_dataloader = DataLoader(
        step_eval_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        collate_fn=data_collator,
    )

    return (
        train_optim_dataloader,
        train_dataloader,
        val_dataloader,
        train_step_eval_dataloader,
    )

# ------- METRICS -------

def compute_sample_losses(model, tokenized_dataset, data_collator, device, config):
    dataloader = DataLoader(tokenized_dataset, batch_size=config["training"]["batch_size"], collate_fn=data_collator, shuffle=False)

    model.eval()
    losses = []
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)

            logits = outputs.logits
            labels = batch["labels"]

            logits = logits[:, :-1, :] #shifted.
            labels = labels[:, 1:]

            mask = labels != -100

            logits = logits.reshape(-1, logits.size(-1))# logits:(batch, sequence_length, vocab_size) -> (batch*sequence_length, vocab_size)
            labels = labels.reshape(-1) # labels:(batch, sequence_length) -> (batch*sequence_length)

            token_losses = loss_function(logits, labels) # loss per token
            token_losses = token_losses.reshape(batch["input_ids"].size(0), -1) # reshape back to (batch, sequence_length)
            token_losses = token_losses * mask # puting the pad loss to 0.

            sample_loss = token_losses.sum(dim=1) / mask.sum(dim=1) # compute mean loss per sample without the loss of padding token.

            losses.extend(sample_loss.cpu().tolist()) # concatenate the loss per sample from each batch.
    return losses

def compute_tail_loss(sample_losses, tail_ratio=0.1):
    sorted_losses = sorted(sample_losses)
    k = int(len(sorted_losses) * tail_ratio)

    if k == 0:
        k = 1

    tail_losses = sorted_losses[-k:]
    tail_loss = sum(tail_losses) / len(tail_losses)

    return tail_loss

def evaluation(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct = 0
    loss_function = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            batch_loss = loss_function(flat_logits, flat_labels) # compute the sum of the loss for the batch
            mask = flat_labels != -100

            total_loss += batch_loss.item()
            total_tokens += mask.sum().item()

            predictions = flat_logits.argmax(dim=-1)
            correct += (predictions[mask] == flat_labels[mask]).sum().item()
    if total_tokens == 0:
        raise ValueError("No valid tokens found in the dataloader.")


    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    acc = correct / total_tokens if total_tokens > 0 else 0

    return avg_loss, ppl, acc, total_tokens

def compute_shift_severity(config, clean_dataset, shifted_dataset, data_collator, device):
    model, _, _ = setup_model_and_tokenizer(config, device)

    clean_losses = compute_sample_losses(model, clean_dataset, data_collator, device, config)
    shifted_losses = compute_sample_losses(model, shifted_dataset, data_collator, device, config)

    clean_ppl = math.exp(np.mean(clean_losses))
    shifted_ppl = math.exp(np.mean(shifted_losses))

    return shifted_ppl - clean_ppl

# ------- TRAINING -------

# ============================================================
# ERM TRAINING WITH STEP LOGGING - FIXED NUMBER OF EPOCHS
# ============================================================

def train_with_step_logging(
    model,
    train_optim_dataloader,
    train_dataloader,
    val_dataloader,
    train_step_eval_dataloader,
    optimizer,
    device,
    accumulation_steps,
    eval_every_optimizer_steps=10,
    eval_first_epoch_only=True,
    max_epochs=10,
    scheduler=None,
):
    """
    Train an ERM model for a fixed number of epochs.

    Logs:
        - optimizer steps,
        - cumulative number of tokens seen,
        - train CE loss,
        - validation CE loss.

    Frequent step-level evaluation can optionally be restricted
    to the first epoch, but optimizer steps and tokens are counted
    during the entire training.

    Returns
    -------
    step_history : list of dict
        Fine-grained evaluations indexed by optimizer step.

    epoch_history : list of dict
        Full train/validation evaluations after each epoch.
    """

    if accumulation_steps <= 0:
        raise ValueError(
            "accumulation_steps must be a positive integer."
        )

    global_step = 0
    cumulative_tokens_seen = 0

    step_history = []
    epoch_history = []

    # --------------------------------------------------------
    # STEP 0
    # --------------------------------------------------------

    train_loss, train_ppl, train_acc, _ = evaluation(
        model,
        train_step_eval_dataloader,
        device=device,
    )

    val_loss, val_ppl, val_acc, _ = evaluation(
        model,
        val_dataloader,
        device=device,
    )

    step_history.append(
        {
            "optimizer_step": 0,
            "cumulative_tokens_seen": 0,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_ppl": train_ppl,
            "val_ppl": val_ppl,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
    )

    optimizer.zero_grad(set_to_none=True)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(max_epochs):

        model.train()

        num_batches = len(train_optim_dataloader)
        total_optimization_loss = 0.0

        for i, batch in enumerate(train_optim_dataloader):

            batch = move_batch_to_device(batch, device)

            outputs = model(**batch)
            loss = outputs.loss

            total_optimization_loss += loss.item()

            # Count the actual number of tokens processed.
            if "attention_mask" in batch:
                cumulative_tokens_seen += (
                    batch["attention_mask"].sum().item()
                )
            else:
                cumulative_tokens_seen += (
                    batch["input_ids"].numel()
                )

            # ------------------------------------------------
            # GRADIENT ACCUMULATION
            # ------------------------------------------------

            group_start = (
                i // accumulation_steps
            ) * accumulation_steps

            current_group_size = min(
                accumulation_steps,
                num_batches - group_start,
            )

            scaled_loss = loss / current_group_size
            scaled_loss.backward()

            is_end_of_group = (
                (i + 1) % accumulation_steps == 0
            )

            is_last_batch = (
                i == num_batches - 1
            )

            if is_end_of_group or is_last_batch:

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                # --------------------------------------------
                # FINE-GRAINED STEP EVALUATION
                # --------------------------------------------

                should_evaluate = (
                    global_step % eval_every_optimizer_steps == 0
                    and (
                        not eval_first_epoch_only
                        or epoch == 0
                    )
                )

                if should_evaluate:

                    step_train_loss, step_train_ppl, step_train_acc, _ = (
                        evaluation(
                            model,
                            train_step_eval_dataloader,
                            device=device,
                        )
                    )

                    step_val_loss, step_val_ppl, step_val_acc, _ = (
                        evaluation(
                            model,
                            val_dataloader,
                            device=device,
                        )
                    )

                    step_history.append(
                        {
                            "optimizer_step": global_step,
                            "cumulative_tokens_seen": cumulative_tokens_seen,
                            "train_loss": step_train_loss,
                            "val_loss": step_val_loss,
                            "train_ppl": step_train_ppl,
                            "val_ppl": step_val_ppl,
                            "train_acc": step_train_acc,
                            "val_acc": step_val_acc,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    )

                    model.train()

        # ----------------------------------------------------
        # END-OF-EPOCH EVALUATION
        # ----------------------------------------------------

        train_loss, train_ppl, train_acc, train_tokens = evaluation(
            model,
            train_dataloader,
            device=device,
        )

        val_loss, val_ppl, val_acc, val_tokens = evaluation(
            model,
            val_dataloader,
            device=device,
        )

        avg_optimization_loss = (
            total_optimization_loss / num_batches
        )

        epoch_history.append(
            {
                "epoch": epoch + 1,
                "optimizer_step": global_step,
                "cumulative_tokens_seen": cumulative_tokens_seen,
                "optimization_loss": avg_optimization_loss,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_ppl": train_ppl,
                "val_ppl": val_ppl,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_eval_tokens": train_tokens,
                "val_eval_tokens": val_tokens,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Also keep the end of each epoch in step_history.
        if step_history[-1]["optimizer_step"] != global_step:

            step_history.append(
                {
                    "optimizer_step": global_step,
                    "cumulative_tokens_seen": cumulative_tokens_seen,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_ppl": train_ppl,
                    "val_ppl": val_ppl,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        print(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"step={global_step} | "
            f"tokens={cumulative_tokens_seen:,} | "
            f"optim_loss={avg_optimization_loss:.4f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_ppl={val_ppl:.2f}",
            flush=True,
        )

    return step_history, epoch_history

# ============================================================
# ERM TRAINING WITH STEP LOGGING + EARLY STOPPING
# ============================================================

def train_with_step_logging_early_stopping(
    model,
    train_optim_dataloader,
    train_dataloader,
    val_dataloader,
    train_step_eval_dataloader,
    optimizer,
    device,
    accumulation_steps,
    eval_every_optimizer_steps=10,
    eval_first_epoch_only=True,
    max_epochs=30,
    patience=3,
    min_delta=0.0,
    scheduler=None,
):
    """
    ERM training with optimizer-step/token logging and early stopping.

    Early stopping is based on the standard validation
    cross-entropy loss.

    At the end, the model is restored to the checkpoint
    with the lowest validation loss.
    """

    if accumulation_steps <= 0:
        raise ValueError(
            "accumulation_steps must be a positive integer."
        )

    if patience <= 0:
        raise ValueError(
            "patience must be a positive integer."
        )

    global_step = 0
    cumulative_tokens_seen = 0

    step_history = []
    epoch_history = []

    best_val_loss = float("inf")
    best_epoch = None
    best_optimizer_step = None
    best_state_dict = None

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # STEP 0
    # --------------------------------------------------------

    train_loss, train_ppl, train_acc, _ = evaluation(
        model,
        train_step_eval_dataloader,
        device=device,
    )

    val_loss, val_ppl, val_acc, _ = evaluation(
        model,
        val_dataloader,
        device=device,
    )

    step_history.append(
        {
            "optimizer_step": 0,
            "cumulative_tokens_seen": 0,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_ppl": train_ppl,
            "val_ppl": val_ppl,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
    )

    optimizer.zero_grad(set_to_none=True)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(max_epochs):

        model.train()

        num_batches = len(train_optim_dataloader)
        total_optimization_loss = 0.0

        for i, batch in enumerate(train_optim_dataloader):

            batch = move_batch_to_device(batch, device)

            outputs = model(**batch)
            loss = outputs.loss

            total_optimization_loss += loss.item()

            if "attention_mask" in batch:
                cumulative_tokens_seen += (
                    batch["attention_mask"].sum().item()
                )
            else:
                cumulative_tokens_seen += (
                    batch["input_ids"].numel()
                )

            group_start = (
                i // accumulation_steps
            ) * accumulation_steps

            current_group_size = min(
                accumulation_steps,
                num_batches - group_start,
            )

            scaled_loss = loss / current_group_size
            scaled_loss.backward()

            is_end_of_group = (
                (i + 1) % accumulation_steps == 0
            )

            is_last_batch = (
                i == num_batches - 1
            )

            if is_end_of_group or is_last_batch:

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                should_evaluate = (
                    global_step % eval_every_optimizer_steps == 0
                    and (
                        not eval_first_epoch_only
                        or epoch == 0
                    )
                )

                if should_evaluate:

                    step_train_loss, step_train_ppl, step_train_acc, _ = (
                        evaluation(
                            model,
                            train_step_eval_dataloader,
                            device=device,
                        )
                    )

                    step_val_loss, step_val_ppl, step_val_acc, _ = (
                        evaluation(
                            model,
                            val_dataloader,
                            device=device,
                        )
                    )

                    step_history.append(
                        {
                            "optimizer_step": global_step,
                            "cumulative_tokens_seen": cumulative_tokens_seen,
                            "train_loss": step_train_loss,
                            "val_loss": step_val_loss,
                            "train_ppl": step_train_ppl,
                            "val_ppl": step_val_ppl,
                            "train_acc": step_train_acc,
                            "val_acc": step_val_acc,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    )

                    model.train()

        # ----------------------------------------------------
        # END-OF-EPOCH EVALUATION
        # ----------------------------------------------------

        train_loss, train_ppl, train_acc, train_tokens = evaluation(
            model,
            train_dataloader,
            device=device,
        )

        val_loss, val_ppl, val_acc, val_tokens = evaluation(
            model,
            val_dataloader,
            device=device,
        )

        avg_optimization_loss = (
            total_optimization_loss / num_batches
        )

        epoch_history.append(
            {
                "epoch": epoch + 1,
                "optimizer_step": global_step,
                "cumulative_tokens_seen": cumulative_tokens_seen,
                "optimization_loss": avg_optimization_loss,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_ppl": train_ppl,
                "val_ppl": val_ppl,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_eval_tokens": train_tokens,
                "val_eval_tokens": val_tokens,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if step_history[-1]["optimizer_step"] != global_step:
            step_history.append(
                {
                    "optimizer_step": global_step,
                    "cumulative_tokens_seen": cumulative_tokens_seen,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_ppl": train_ppl,
                    "val_ppl": val_ppl,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        print(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"step={global_step} | "
            f"tokens={cumulative_tokens_seen:,} | "
            f"optim_loss={avg_optimization_loss:.4f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_ppl={val_ppl:.2f}",
            flush=True,
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if val_loss < best_val_loss - min_delta:

            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_optimizer_step = global_step

            # Store the best model on CPU so we do not duplicate
            # the complete GPT-2 model in GPU memory.
            best_state_dict = {
                name: param.detach().cpu().clone()
                for name, param in model.state_dict().items()
            }

            epochs_without_improvement = 0

            print(
                f"New best validation loss: "
                f"{best_val_loss:.4f} "
                f"(epoch={best_epoch}, "
                f"step={best_optimizer_step})",
                flush=True,
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No validation improvement for "
                f"{epochs_without_improvement}/{patience} epoch(s).",
                flush=True,
            )

            if epochs_without_improvement >= patience:

                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best epoch={best_epoch}, "
                    f"best step={best_optimizer_step}, "
                    f"best val loss={best_val_loss:.4f}.",
                    flush=True,
                )

                break

    # --------------------------------------------------------
    # RESTORE BEST CHECKPOINT
    # --------------------------------------------------------

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    best_info = {
        "best_epoch": best_epoch,
        "best_optimizer_step": best_optimizer_step,
        "best_val_loss": best_val_loss,
    }

    return step_history, epoch_history, best_info


# ============================================================
# KL-DRO TRAINING WITH STEP LOGGING - FIXED EPOCHS
# ============================================================

def KL_DRO_train_with_step_logging(
    model,
    train_optim_dataloader,
    train_dataloader,
    val_dataloader,
    train_step_eval_dataloader,
    optimizer,
    gamma,
    lambd,
    rho,
    device,
    accumulation_steps,
    eval_every_optimizer_steps=10,
    eval_first_epoch_only=True,
    max_epochs=10,
    scheduler=None,
):
    """
    Train a KL-DRO model for a fixed number of epochs.

    If lambda == 0, standard ERM training is used.

    Training objective:
        mix_loss =
            (1 - gamma) * ERM_loss
            + gamma * DRO_loss

    where:
        DRO_loss =
            lambda * logmeanexp(sample_loss / lambda)
            + lambda * rho

    Train/validation curves are always standard CE losses so
    that all lambda models are directly comparable.
    """

    # --------------------------------------------------------
    # lambda = 0 -> ERM
    # --------------------------------------------------------

    if lambd == 0 or lambd == 0.0:

        print(
            "\nλ=0 -> standard ERM training.",
            flush=True,
        )

        return train_with_step_logging(
            model=model,
            train_optim_dataloader=train_optim_dataloader,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            train_step_eval_dataloader=train_step_eval_dataloader,
            optimizer=optimizer,
            device=device,
            accumulation_steps=accumulation_steps,
            eval_every_optimizer_steps=eval_every_optimizer_steps,
            eval_first_epoch_only=eval_first_epoch_only,
            max_epochs=max_epochs,
            scheduler=scheduler,
        )

    if lambd < 0:
        raise ValueError(
            "lambda must be >= 0."
        )

    if accumulation_steps <= 0:
        raise ValueError(
            "accumulation_steps must be a positive integer."
        )

    global_step = 0
    cumulative_tokens_seen = 0

    step_history = []
    epoch_history = []

    loss_function = torch.nn.CrossEntropyLoss(
        reduction="none",
        ignore_index=-100,
    )

    # --------------------------------------------------------
    # STEP 0
    # --------------------------------------------------------

    train_loss, train_ppl, train_acc, _ = evaluation(
        model,
        train_step_eval_dataloader,
        device=device,
    )

    val_loss, val_ppl, val_acc, _ = evaluation(
        model,
        val_dataloader,
        device=device,
    )

    step_history.append(
        {
            "optimizer_step": 0,
            "cumulative_tokens_seen": 0,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_ppl": train_ppl,
            "val_ppl": val_ppl,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
    )

    optimizer.zero_grad(set_to_none=True)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(max_epochs):

        model.train()

        num_batches = len(train_optim_dataloader)

        total_mix_loss = 0.0
        total_erm_component = 0.0
        total_dro_component = 0.0

        for i, batch in enumerate(train_optim_dataloader):

            batch = move_batch_to_device(batch, device)

            outputs = model(**batch)

            # Standard autoregressive CE averaged over valid tokens.
            avg_loss = outputs.loss

            logits = outputs.logits
            labels = batch["labels"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            batch_size = shift_logits.size(0)

            flat_logits = shift_logits.view(
                -1,
                shift_logits.size(-1),
            )

            flat_labels = shift_labels.view(-1)

            token_losses = loss_function(
                flat_logits,
                flat_labels,
            )

            token_losses = token_losses.view(
                batch_size,
                -1,
            )

            valid_mask = (
                shift_labels != -100
            ).float()

            # Mean CE for each sequence.
            valid_tokens_per_sample = (
                valid_mask.sum(dim=1).clamp_min(1.0)
            )

            sample_losses = (
                token_losses.sum(dim=1)
                / valid_tokens_per_sample
            )

            # -----------------------------------------------
            # KL-DRO objective
            # -----------------------------------------------

            log_mean_exp = (
                torch.logsumexp(
                    sample_losses / lambd,
                    dim=0,
                )
                - math.log(sample_losses.numel())
            )

            dro_loss = (
                lambd * log_mean_exp
                + lambd * rho
            )

            mix_loss = (
                (1.0 - gamma) * avg_loss
                + gamma * dro_loss
            )

            total_mix_loss += mix_loss.item()
            total_erm_component += avg_loss.item()
            total_dro_component += dro_loss.item()

            if "attention_mask" in batch:
                cumulative_tokens_seen += (
                    batch["attention_mask"].sum().item()
                )
            else:
                cumulative_tokens_seen += (
                    batch["input_ids"].numel()
                )

            # ------------------------------------------------
            # GRADIENT ACCUMULATION
            # ------------------------------------------------

            group_start = (
                i // accumulation_steps
            ) * accumulation_steps

            current_group_size = min(
                accumulation_steps,
                num_batches - group_start,
            )

            scaled_loss = (
                mix_loss / current_group_size
            )

            scaled_loss.backward()

            is_end_of_group = (
                (i + 1) % accumulation_steps == 0
            )

            is_last_batch = (
                i == num_batches - 1
            )

            if is_end_of_group or is_last_batch:

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                should_evaluate = (
                    global_step % eval_every_optimizer_steps == 0
                    and (
                        not eval_first_epoch_only
                        or epoch == 0
                    )
                )

                if should_evaluate:

                    step_train_loss, step_train_ppl, step_train_acc, _ = (
                        evaluation(
                            model,
                            train_step_eval_dataloader,
                            device=device,
                        )
                    )

                    step_val_loss, step_val_ppl, step_val_acc, _ = (
                        evaluation(
                            model,
                            val_dataloader,
                            device=device,
                        )
                    )

                    step_history.append(
                        {
                            "optimizer_step": global_step,
                            "cumulative_tokens_seen": cumulative_tokens_seen,
                            "train_loss": step_train_loss,
                            "val_loss": step_val_loss,
                            "train_ppl": step_train_ppl,
                            "val_ppl": step_val_ppl,
                            "train_acc": step_train_acc,
                            "val_acc": step_val_acc,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    )

                    model.train()

        # ----------------------------------------------------
        # END-OF-EPOCH STANDARD CE EVALUATION
        # ----------------------------------------------------

        train_loss, train_ppl, train_acc, train_tokens = evaluation(
            model,
            train_dataloader,
            device=device,
        )

        val_loss, val_ppl, val_acc, val_tokens = evaluation(
            model,
            val_dataloader,
            device=device,
        )

        avg_mix_loss = (
            total_mix_loss / num_batches
        )

        avg_erm_component = (
            total_erm_component / num_batches
        )

        avg_dro_component = (
            total_dro_component / num_batches
        )

        epoch_history.append(
            {
                "epoch": epoch + 1,
                "optimizer_step": global_step,
                "cumulative_tokens_seen": cumulative_tokens_seen,

                "optimization_loss": avg_mix_loss,
                "erm_component": avg_erm_component,
                "dro_component": avg_dro_component,

                "train_loss": train_loss,
                "val_loss": val_loss,

                "train_ppl": train_ppl,
                "val_ppl": val_ppl,

                "train_acc": train_acc,
                "val_acc": val_acc,

                "train_eval_tokens": train_tokens,
                "val_eval_tokens": val_tokens,

                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if step_history[-1]["optimizer_step"] != global_step:

            step_history.append(
                {
                    "optimizer_step": global_step,
                    "cumulative_tokens_seen": cumulative_tokens_seen,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_ppl": train_ppl,
                    "val_ppl": val_ppl,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        print(
            f"[λ={lambd}] "
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"step={global_step} | "
            f"tokens={cumulative_tokens_seen:,} | "
            f"mix_loss={avg_mix_loss:.4f} | "
            f"ERM_component={avg_erm_component:.4f} | "
            f"DRO_component={avg_dro_component:.4f} | "
            f"train_CE={train_loss:.4f} | "
            f"val_CE={val_loss:.4f} | "
            f"val_ppl={val_ppl:.2f}",
            flush=True,
        )

    return step_history, epoch_history

# ============================================================
# KL-DRO TRAINING WITH STEP LOGGING + EARLY STOPPING
# ============================================================

def KL_DRO_train_with_step_logging_early_stopping(
    model,
    train_optim_dataloader,
    train_dataloader,
    val_dataloader,
    train_step_eval_dataloader,
    optimizer,
    gamma,
    lambd,
    rho,
    device,
    accumulation_steps,
    eval_every_optimizer_steps=10,
    eval_first_epoch_only=True,
    max_epochs=30,
    patience=3,
    min_delta=0.0,
    scheduler=None,
):
    """
    KL-DRO training with step/token logging and early stopping.

    If lambda == 0, standard ERM training with early stopping
    is used.

    Early stopping is based on standard validation CE loss.
    The best checkpoint is restored before returning.
    """

    # --------------------------------------------------------
    # lambda = 0 -> ERM
    # --------------------------------------------------------

    if lambd == 0 or lambd == 0.0:

        print(
            "\nλ=0 -> standard ERM training with early stopping.",
            flush=True,
        )

        return train_with_step_logging_early_stopping(
            model=model,
            train_optim_dataloader=train_optim_dataloader,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            train_step_eval_dataloader=train_step_eval_dataloader,
            optimizer=optimizer,
            device=device,
            accumulation_steps=accumulation_steps,
            eval_every_optimizer_steps=eval_every_optimizer_steps,
            eval_first_epoch_only=eval_first_epoch_only,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            scheduler=scheduler,
        )

    if lambd < 0:
        raise ValueError(
            "lambda must be >= 0."
        )

    if accumulation_steps <= 0:
        raise ValueError(
            "accumulation_steps must be a positive integer."
        )

    if patience <= 0:
        raise ValueError(
            "patience must be a positive integer."
        )

    global_step = 0
    cumulative_tokens_seen = 0

    step_history = []
    epoch_history = []

    best_val_loss = float("inf")
    best_epoch = None
    best_optimizer_step = None
    best_state_dict = None

    epochs_without_improvement = 0

    loss_function = torch.nn.CrossEntropyLoss(
        reduction="none",
        ignore_index=-100,
    )

    # --------------------------------------------------------
    # STEP 0
    # --------------------------------------------------------

    train_loss, train_ppl, train_acc, _ = evaluation(
        model,
        train_step_eval_dataloader,
        device=device,
    )

    val_loss, val_ppl, val_acc, _ = evaluation(
        model,
        val_dataloader,
        device=device,
    )

    step_history.append(
        {
            "optimizer_step": 0,
            "cumulative_tokens_seen": 0,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_ppl": train_ppl,
            "val_ppl": val_ppl,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
    )

    optimizer.zero_grad(set_to_none=True)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(max_epochs):

        model.train()

        num_batches = len(train_optim_dataloader)

        total_mix_loss = 0.0
        total_erm_component = 0.0
        total_dro_component = 0.0

        for i, batch in enumerate(train_optim_dataloader):

            batch = move_batch_to_device(
                batch,
                device,
            )

            outputs = model(**batch)

            avg_loss = outputs.loss

            logits = outputs.logits
            labels = batch["labels"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            batch_size = shift_logits.size(0)

            flat_logits = shift_logits.view(
                -1,
                shift_logits.size(-1),
            )

            flat_labels = shift_labels.view(-1)

            token_losses = loss_function(
                flat_logits,
                flat_labels,
            )

            token_losses = token_losses.view(
                batch_size,
                -1,
            )

            valid_mask = (
                shift_labels != -100
            ).float()

            valid_tokens_per_sample = (
                valid_mask.sum(dim=1).clamp_min(1.0)
            )

            sample_losses = (
                token_losses.sum(dim=1)
                / valid_tokens_per_sample
            )

            log_mean_exp = (
                torch.logsumexp(
                    sample_losses / lambd,
                    dim=0,
                )
                - math.log(sample_losses.numel())
            )

            dro_loss = (
                lambd * log_mean_exp
                + lambd * rho
            )

            mix_loss = (
                (1.0 - gamma) * avg_loss
                + gamma * dro_loss
            )

            total_mix_loss += mix_loss.item()
            total_erm_component += avg_loss.item()
            total_dro_component += dro_loss.item()

            if "attention_mask" in batch:
                cumulative_tokens_seen += (
                    batch["attention_mask"].sum().item()
                )
            else:
                cumulative_tokens_seen += (
                    batch["input_ids"].numel()
                )

            group_start = (
                i // accumulation_steps
            ) * accumulation_steps

            current_group_size = min(
                accumulation_steps,
                num_batches - group_start,
            )

            scaled_loss = (
                mix_loss / current_group_size
            )

            scaled_loss.backward()

            is_end_of_group = (
                (i + 1) % accumulation_steps == 0
            )

            is_last_batch = (
                i == num_batches - 1
            )

            if is_end_of_group or is_last_batch:

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                should_evaluate = (
                    global_step % eval_every_optimizer_steps == 0
                    and (
                        not eval_first_epoch_only
                        or epoch == 0
                    )
                )

                if should_evaluate:

                    step_train_loss, step_train_ppl, step_train_acc, _ = (
                        evaluation(
                            model,
                            train_step_eval_dataloader,
                            device=device,
                        )
                    )

                    step_val_loss, step_val_ppl, step_val_acc, _ = (
                        evaluation(
                            model,
                            val_dataloader,
                            device=device,
                        )
                    )

                    step_history.append(
                        {
                            "optimizer_step": global_step,
                            "cumulative_tokens_seen": cumulative_tokens_seen,
                            "train_loss": step_train_loss,
                            "val_loss": step_val_loss,
                            "train_ppl": step_train_ppl,
                            "val_ppl": step_val_ppl,
                            "train_acc": step_train_acc,
                            "val_acc": step_val_acc,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    )

                    model.train()

        # ----------------------------------------------------
        # END-OF-EPOCH EVALUATION
        # ----------------------------------------------------

        train_loss, train_ppl, train_acc, train_tokens = evaluation(
            model,
            train_dataloader,
            device=device,
        )

        val_loss, val_ppl, val_acc, val_tokens = evaluation(
            model,
            val_dataloader,
            device=device,
        )

        avg_mix_loss = (
            total_mix_loss / num_batches
        )

        avg_erm_component = (
            total_erm_component / num_batches
        )

        avg_dro_component = (
            total_dro_component / num_batches
        )

        epoch_history.append(
            {
                "epoch": epoch + 1,
                "optimizer_step": global_step,
                "cumulative_tokens_seen": cumulative_tokens_seen,

                "optimization_loss": avg_mix_loss,
                "erm_component": avg_erm_component,
                "dro_component": avg_dro_component,

                "train_loss": train_loss,
                "val_loss": val_loss,

                "train_ppl": train_ppl,
                "val_ppl": val_ppl,

                "train_acc": train_acc,
                "val_acc": val_acc,

                "train_eval_tokens": train_tokens,
                "val_eval_tokens": val_tokens,

                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if step_history[-1]["optimizer_step"] != global_step:

            step_history.append(
                {
                    "optimizer_step": global_step,
                    "cumulative_tokens_seen": cumulative_tokens_seen,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_ppl": train_ppl,
                    "val_ppl": val_ppl,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        print(
            f"[λ={lambd}] "
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"step={global_step} | "
            f"tokens={cumulative_tokens_seen:,} | "
            f"mix_loss={avg_mix_loss:.4f} | "
            f"ERM_component={avg_erm_component:.4f} | "
            f"DRO_component={avg_dro_component:.4f} | "
            f"train_CE={train_loss:.4f} | "
            f"val_CE={val_loss:.4f} | "
            f"val_ppl={val_ppl:.2f}",
            flush=True,
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if val_loss < best_val_loss - min_delta:

            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_optimizer_step = global_step

            best_state_dict = {
                name: param.detach().cpu().clone()
                for name, param in model.state_dict().items()
            }

            epochs_without_improvement = 0

            print(
                f"[λ={lambd}] New best validation loss: "
                f"{best_val_loss:.4f} "
                f"(epoch={best_epoch}, "
                f"step={best_optimizer_step})",
                flush=True,
            )

        else:

            epochs_without_improvement += 1

            print(
                f"[λ={lambd}] No validation improvement for "
                f"{epochs_without_improvement}/{patience} epoch(s).",
                flush=True,
            )

            if epochs_without_improvement >= patience:

                print(
                    f"[λ={lambd}] Early stopping at "
                    f"epoch {epoch + 1}. "
                    f"Best epoch={best_epoch}, "
                    f"best step={best_optimizer_step}, "
                    f"best val loss={best_val_loss:.4f}.",
                    flush=True,
                )

                break

    # --------------------------------------------------------
    # RESTORE BEST CHECKPOINT
    # --------------------------------------------------------

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    best_info = {
        "best_epoch": best_epoch,
        "best_optimizer_step": best_optimizer_step,
        "best_val_loss": best_val_loss,
    }

    return step_history, epoch_history, best_info

# ------- GRAPHS -------

def plot_comparison_curve(
    df,
    x_column,
    y_column,
    xlabel,
    ylabel,
    title,
    output_path,
):
    """
    Plot one loss metric for all injection rates on the same figure.

    Each injection rate corresponds to one curve.
    """

    plt.figure(figsize=(10, 6))

    injection_rates = sorted(df["injection_rate"].unique())

    for injection_rate in injection_rates:
        subset = df[df["injection_rate"] == injection_rate].sort_values(x_column)
        label = f"p = {100 * injection_rate:.0f}%"
        plt.plot(subset[x_column], subset[y_column], marker="o", label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300,)
    plt.close()