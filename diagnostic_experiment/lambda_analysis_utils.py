import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import utils
from transformers import AutoModelForCausalLM
from sklearn.decomposition import PCA

# -- METRICS --
def delta_update(initial_model, model):
    """ Calculate Δ_λ​=θ_λ​−θ_0 the update associated with each λ. """
    deltas = []
    initial_state = initial_model.state_dict()
    model_state = model.state_dict()
    for name in initial_state.keys():
        if name in model_state:
            p_0 = initial_state[name]
            p_lambd = model_state[name]
            if torch.is_floating_point(p_0): # verify if p_0 is a floating point tensor.
                delta = p_lambd.detach().cpu().float() - p_0.detach().cpu().float() # detach() to get a tensor without gradient, cpu() to move it to CPU because the RAM is bigger and because some libraries only work on CPU.
                deltas.append(delta.flatten()) # the shape of deltas is a list of matrices for the different levels of weights in the model, so we flatten the matrices to get a list of vectors.
    return torch.cat(deltas) # concatenate the list of vectors to get a single big vector.

def d_init(delta_lambda):
    """ Calculate the l2 norm of delta_lambda. """
    return torch.sqrt(torch.dot(delta_lambda, delta_lambda)).item() # without item() it will return a tensor of size 1.

def d_step(delta_lambda, delta_lambda_next):
    """ Calculate the l2 distance between delta_lambda and delta_lambda_next. """
    step = delta_lambda_next - delta_lambda
    return torch.sqrt(torch.dot(step, step)).item()

def cos_sim(delta_lambda_i, delta_lambda_j):
    """ Calculate the cosine similarity between delta_lambda_i and delta_lambda_j. """
    dot_ij = torch.dot(delta_lambda_i, delta_lambda_j)
    dot_ii = torch.dot(delta_lambda_i, delta_lambda_i)
    dot_jj = torch.dot(delta_lambda_j, delta_lambda_j)
    return (dot_ij / (torch.sqrt(dot_ii * dot_jj) + 1e-12)).item() # add a small value to avoid division by zero.

def D_pred(model_i, model_j, probe_dataloader, device):
    """ Calculate the predictive divergence from the standard fine-tuned model. """
    model_i.eval()
    model_j.eval()
    total_divergence = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in probe_dataloader:
            batch = utils.move_batch_to_device(batch, device)
            outputs_i = model_i(**batch)
            outputs_j = model_j(**batch)

            logits_i = outputs_i.logits
            logits_j = outputs_j.logits

            probs_i = torch.nn.functional.softmax(logits_i, dim=-1)
            log_probs_j = torch.nn.functional.log_softmax(logits_j, dim=-1)

            kl_ij = torch.nn.functional.kl_div(log_probs_j, probs_i, reduction='batchmean') # reduction='batchmean' to get the average KL.
            total_divergence += kl_ij.item() * batch["input_ids"].size(0) # multiply to get only the sum of divergences and not the mean over batches.
            total_examples += batch["input_ids"].size(0)
    
    return total_divergence / total_examples

def compute_pca_updates(deltas, lambdas):
    """ Project updates vectors in 2D using PCA for visualization. """
    M = torch.stack([deltas[lambd] for lambd in lambdas])
    M = M.numpy() # numpy table of shape (n_lambdas, n_parameters).
    print("Matrix shape:", M.shape, flush=True)
    pca = PCA(n_components=2)
    M_2D = pca.fit_transform(M) # convert n_parameters into 2 principal components, so the shape is (n_lambdas, 2).
    return M_2D, pca.explained_variance_ratio_ # return the 2D projection for each lambda and the pourcentage of information explained by the 2 principal components.


# -- GENERATE MODELS --
def save_lambda_models(model, lambd, config):
    """ Save the model for a given λ. """
    output_dir = config["outputs"]["dir"]
    save_dir = os.path.join(output_dir, "lambda_models",f"lambda_{lambd}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    return save_dir

def build_lambdas(config):
    """ Generate a list of λ values based on the config. """
    lambda_step = config["diagnostic"]["epsilon"]
    lambda_max = config["diagnostic"]["lambda_max"]
    n = int(lambda_max / lambda_step)
    return [round(i * lambda_step, 6) for i in range(n+1)]

def train_lambda_models(train_dataloader, val_dataloader, device, config):
    """ Train and save models for each λ. """
    
    lambdas = build_lambdas(config)
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
        save_dir = save_lambda_models(model, lambd, config)
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

# -- ANALYSIS --
def load_lambda_models(saved_paths, lambd, config, device):
    """ Load the model for a given λ. """
    model_path = saved_paths[lambd]
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)

    return model

def compute_lambda_metrics(saved_paths, config, device, probe_dataloader):
    """ Compute the metrics for each lambda model."""
    initial_model, _, _ = utils.setup_model_and_tokenizer(config, device)
    lambdas = sorted(saved_paths.keys())

    deltas = {}
    D_pred_values = {}
    model_0 = load_lambda_models(saved_paths, 0.0, config, device)
    # Compute the deltas:
    for lambd in lambdas:
        model = load_lambda_models(saved_paths, lambd, config, device)
        deltas[lambd] = delta_update(initial_model, model)
        D_pred_values[lambd] = D_pred(model, model_0, probe_dataloader, device)
        del model 
        torch.cuda.empty_cache()
    del model_0
    torch.cuda.empty_cache()

    delta_0 = deltas[0.0]
    d_init_values = []
    d_step_values = []
    cos_sim_values = []

   
    for lambd in lambdas:
        d_init_value = d_init(deltas[lambd])
        d_init_values.append(d_init_value)
        cos_sim_value = cos_sim(deltas[lambd], delta_0)
        cos_sim_values.append(cos_sim_value)
    
    for i in range(len(lambdas) - 1):
        lambd = lambdas[i]
        next_lambd = lambdas[i+1]
        d_step_value = d_step(deltas[lambd], deltas[next_lambd])
        d_step_values.append(d_step_value)
    X_2D, explained_var = compute_pca_updates(deltas, lambdas)
    del deltas
    del initial_model
    torch.cuda.empty_cache()
    return lambdas, d_init_values, d_step_values, cos_sim_values, D_pred_values, X_2D, explained_var

# -- GRAPHS --
def plot_lambda_metrics(lambdas, d_init_values, d_step_values, cos_sim_values, D_pred_values, config):
    """ Plot the metrics VS lambdas."""
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 12))

    # Plot distance from initialization VS lambdas
    plt.subplot(2, 2, 1)
    plt.plot(lambdas, d_init_values, marker='o', label='d_init')
    plt.xlabel('λ')
    plt.ylabel('Distance from Initialization')
    plt.title('Distance from Initialization VS λ')
    plt.legend()

    # Plot distance between consecutive robustness levels VS lambdas
    plt.subplot(2, 2, 2)
    plt.plot(lambdas[:-1], d_step_values, marker='o', label='d_step')
    plt.xlabel('λ')
    plt.ylabel('Distance between Consecutive λs')
    plt.title('Distance between Consecutive Robustness Levels VS λ')
    plt.legend()

    # Plot cosine similarity with ERM VS lambdas
    plt.subplot(2, 2, 3)
    plt.plot(lambdas, cos_sim_values, marker='o', label='cos_sim')
    plt.xlabel('λ')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity with ERM VS λ')
    plt.legend()

    # Plot predictive divergence from ERM model VS lambdas
    plt.subplot(2, 2, 4)
    D_pred_list = [D_pred_values[lambd] for lambd in lambdas]
    plt.plot(lambdas, D_pred_list, marker='o', label='D_pred')
    plt.xlabel('λ')
    plt.ylabel('Predictive Divergence from ERM')
    plt.title('Predictive Divergence from ERM VS λ')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/lambda_metrics_curves.png")
    plt.close()

def plot_pca_updates(X_2D, lambdas, explained_var, config):
    """ Plot the PCA projection of updates. """
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 7))

    plt.plot(X_2D[:,0], X_2D[:,1], marker='o')
    for i, lambd in enumerate(lambdas):
        plt.annotate(f"λ={lambd}", (X_2D[i, 0], X_2D[i, 1]))
    plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
    plt.title("PCA of updates vectors")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/PCA_updates.png")
    plt.close()


# -- TABLE --
def lambda_analysis_table(lambdas, d_init_values, d_step_values, cos_sim_values, D_pred_values, config):
    """ Create a table with the metrics for each lambda models."""
    output_dir = config["outputs"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for i, lambd in enumerate(lambdas):
        rows.append({
            "lambda": lambd,
            "d_init": d_init_values[i],
            "d_step": d_step_values[i] if i < len(d_step_values) else None,
            "cos_sim": cos_sim_values[i],
            "D_pred": D_pred_values[lambd]
        })  
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "lambda_analysis_table.csv")
    df.to_csv(csv_path, index=False)
    print(df)
    print(f"Lambda analysis table saved to {csv_path}") 