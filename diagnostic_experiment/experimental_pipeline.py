import os
import json
import math
import time
import copy
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import utils
import hierarchical_routing as hr
import routing_baselines as rb
from shiftlab.data.load_datasets import load_dataset_from_subconfig
BANK_REPO_ID = 'alignment-decision-lab/robustness-model-bank'
OUTPUT_ROOT = 'outputs/experimental_pipeline'
EXPERIMENT_CONFIG = {'experiment_name': 'main_comparison', 'model': 'small', 'sources': ['ArXiv', 'FreeLaw', 'PubMed Central'], 'deployments': ['PG-19', 'PubMed Abstracts', 'GitHub'], 'context_length': 512, 'batch_size': 16, 'num_batches': 10, 'deployment_offset_tokens': 512000, 'methods': ['pretrained', 'best_single_ft', 'mixed_ft', 'first_batch_routing', 'tent', 'hard', 'flat', 'hierarchical', 'oracle'], 'hierarchical': {'H': 2, 'num_iters': 100, 'lr': 0.1, 'num_random_starts': 5, 'dirichlet_concentration': 1.0, 'flat_tau': 1.0, 'seed': 42}, 'flat': {'tau': 1.0}, 'tent': {'lr': 0.001, 'num_steps': 1}, 'run_pca': False, 'progressive_save': True, 'seed': 42}
MODEL_REGISTRY = {'tiny': {'model_name': 'sshleifer/tiny-gpt2', 'bank_prefix': 'tiny-gpt2', 'pretrained_subfolder': 'tiny-gpt2/pretrained/model', 'mixed_ft_subfolder': None, 'lambdas': ['0.0', '1e-05', '0.0001', '0.001', '0.01', '0.1', '0.3', '0.5', '1.0', '2.0']}, 'small': {'model_name': 'gpt2', 'bank_prefix': 'gpt2-small', 'pretrained_subfolder': 'gpt2-small/pretrained/model', 'mixed_ft_subfolder': 'gpt2-small/Mixed_FT_ArXiv_FreeLaw_PubMed_Central/model', 'lambdas': ['0.0', '1e-05', '0.0001', '0.001', '0.01', '0.1', '0.3', '0.5', '1.0', '2.0']}, 'medium': {'model_name': 'gpt2-medium', 'bank_prefix': 'gpt2Medium', 'pretrained_subfolder': None, 'mixed_ft_subfolder': None, 'lambdas': ['0.0', '0.02', '0.05', '0.10', '0.20', '0.50', '0.70', '1.00', '1.50', '2.00']}, 'large': {'model_name': 'gpt2-large', 'bank_prefix': 'gpt2-large', 'pretrained_subfolder': None, 'mixed_ft_subfolder': None, 'lambdas': ['0.0', '0.02', '0.05', '0.10', '0.20', '0.50', '0.70', '1.00', '1.50', '2.00']}, 'xlarge': {'model_name': 'gpt2-xl', 'bank_prefix': 'gpt2-xl', 'pretrained_subfolder': None, 'mixed_ft_subfolder': None, 'lambdas': ['0.0', '0.02', '0.05', '0.10', '0.20', '0.50', '0.70', '1.00', '1.50', '2.00']}}
DATASET_REGISTRY = {'ArXiv': {'bank_name': 'ArXiv', 'dataset_config': {'type': 'hf_text', 'name': 'timaeus/pile-arxiv', 'split': 'train', 'text_column': 'text', 'streaming': True}, 'oracle_name': 'ArXiv'}, 'DM Mathematics': {'bank_name': 'DM Mathematics', 'dataset_config': {'type': 'hf_text', 'name': 'timaeus/pile-dm_mathematics', 'split': 'train', 'text_column': 'text', 'streaming': True}, 'oracle_name': 'DM Mathematics'}, 'EuroParl': {'bank_name': 'EuroParl', 'dataset_config': {'type': 'translation', 'name': 'Helsinki-NLP/europarl', 'config': 'en-fr', 'split': 'train', 'language': 'en', 'streaming': True}, 'oracle_name': 'EuroParl'}, 'FreeLaw': {'bank_name': 'FreeLaw', 'dataset_config': {'type': 'hf_text', 'name': 'timaeus/pile-freelaw', 'split': 'train', 'text_column': 'text', 'streaming': True}, 'oracle_name': 'FreeLaw'}, 'GitHub': {'bank_name': 'Github', 'dataset_config': {'type': 'hf_text', 'name': 'timaeus/pile-github', 'split': 'train', 'text_column': 'text', 'streaming': True}, 'oracle_name': 'Github'}, 'PG-19': {'bank_name': 'Gutenberg (PG-19)', 'dataset_config': {'type': 'hf_text', 'name': 'emozilla/pg19', 'split': 'train', 'text_column': 'text', 'streaming': True}, 'oracle_name': 'Gutenberg (PG-19)'}, 'OWT2': {'bank_name': 'OpenWebText2', 'dataset_config': {'type': 'hf_text', 'name': 'suolyer/pile_openwebtext2', 'split': 'validation', 'text_column': 'text', 'streaming': True}, 'oracle_name': 'OpenWebText2'}, 'PubMed Abstracts': {'bank_name': 'PubMed Abstracts', 'dataset_config': {'type': 'hf_text', 'name': 'timaeus/pile-pubmed_abstracts', 'split': 'train', 'text_column': 'text', 'streaming': True}, 'oracle_name': 'PubMed Abstracts'}, 'PubMed Central': {'bank_name': 'PubMed Central', 'dataset_config': {'type': 'hf_text', 'name': 'datajuicer/the-pile-pubmed-central-refined-by-data-juicer', 'split': 'train', 'text_column': 'text', 'streaming': True}, 'oracle_name': 'PubMed Central'}, 'StackExchange': {'bank_name': 'StackExchange', 'dataset_config': {'type': 'hf_text', 'name': 'flax-sentence-embeddings/stackexchange_title_body_jsonl', 'split': 'train', 'text_column': 'texts', 'streaming': True}, 'oracle_name': 'StackExchange'}, 'Wikipedia': {'bank_name': 'Wikipedia (en)', 'dataset_config': {'type': 'hf_text', 'name': 'wikimedia/wikipedia', 'config': '20231101.en', 'split': 'train', 'text_column': 'text', 'streaming': True}, 'oracle_name': 'Wikipedia (en)'}}
METHOD_DISPLAY_NAMES = {'pretrained': 'Pretrained', 'best_single_ft': 'Best Single FT', 'mixed_ft': 'Mixed-source FT', 'first_batch_routing': 'Top-H + routing + only 1st batch', 'tent': 'Test-time adaptation', 'hard': 'Hard Routing', 'flat': 'Flat Routing', 'hierarchical': 'Hierarchical Routing', 'oracle': 'Target-FT Oracle'}

def get_model_config():
    model_key = EXPERIMENT_CONFIG['model']
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_key]

def get_model_name():
    return get_model_config()['model_name']

def get_bank_prefix():
    return get_model_config()['bank_prefix']

def get_output_dir():
    cfg = EXPERIMENT_CONFIG
    hier = cfg['hierarchical']
    run_name = f"{cfg['model']}_bs{cfg['batch_size']}_H{hier['H']}_iter{hier['num_iters']}"
    return os.path.join(OUTPUT_ROOT, cfg['experiment_name'], run_name)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def validate_config():
    cfg = EXPERIMENT_CONFIG
    if cfg['model'] not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {cfg['model']}")
    for dataset_name in cfg['deployments']:
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Deployment dataset '{dataset_name}' is not in DATASET_REGISTRY.")
    valid_methods = set(METHOD_DISPLAY_NAMES)
    for method in cfg['methods']:
        if method not in valid_methods:
            raise ValueError(f'Unknown method: {method}')
    if cfg['context_length'] <= 1:
        raise ValueError('context_length must be > 1.')
    if cfg['batch_size'] <= 0:
        raise ValueError('batch_size must be > 0.')
    if cfg['num_batches'] <= 0:
        raise ValueError('num_batches must be > 0.')
    if cfg['deployment_offset_tokens'] % cfg['context_length'] != 0:
        raise ValueError('deployment_offset_tokens must be divisible by context_length.')
    if cfg['hierarchical']['H'] <= 0:
        raise ValueError('Hierarchical H must be > 0.')
    if cfg['hierarchical']['num_iters'] <= 0:
        raise ValueError('Hierarchical num_iters must be > 0.')
    if cfg['tent']['num_steps'] <= 0:
        raise ValueError('Tent num_steps must be > 0.')

def save_experiment_config(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    payload = {'experiment': EXPERIMENT_CONFIG, 'model_config': get_model_config(), 'bank_repo_id': BANK_REPO_ID}
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

def build_model_bank_metadata(output_dir):
    model_cfg = get_model_config()
    sources = EXPERIMENT_CONFIG['sources']
    lambdas = model_cfg['lambdas']
    rows = []
    for dataset_name in sources:
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Source dataset '{dataset_name}' is not in DATASET_REGISTRY.")
        bank_name = DATASET_REGISTRY[dataset_name]['bank_name']
        for lambda_str in lambdas:
            rows.append({'dataset_name': dataset_name, 'bank_name': bank_name, 'lambda': float(lambda_str), 'lambda_str': lambda_str, 'subfolder': f"{model_cfg['bank_prefix']}/{bank_name}/lambda_{lambda_str}/model", 'status': 'trained'})
    bank_df = pd.DataFrame(rows)
    metadata_path = os.path.join(output_dir, 'model_bank_metadata.csv')
    bank_df.to_csv(metadata_path, index=False)
    print(f'Built metadata for {len(bank_df)} checkpoints ({len(sources)} sources x {len(lambdas)} lambdas).', flush=True)
    return (metadata_path, bank_df)

def load_hub_model(subfolder, device):
    model = AutoModelForCausalLM.from_pretrained(BANK_REPO_ID, subfolder=subfolder).to(device)
    model.eval()
    return model

def clear_model(model, device):
    if model is not None:
        del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def load_pretrained_model(device):
    model_cfg = get_model_config()
    subfolder = model_cfg.get('pretrained_subfolder')
    if subfolder is not None:
        model = AutoModelForCausalLM.from_pretrained(BANK_REPO_ID, subfolder=subfolder).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_cfg['model_name']).to(device)
    model.eval()
    return model

def get_oracle_subfolder(dataset_name):
    oracle_name = DATASET_REGISTRY[dataset_name]['oracle_name']
    if oracle_name is None:
        return None
    return f'{get_bank_prefix()}/{oracle_name}/lambda_0.0/model'

def load_deployment_data(dataset_name, tokenizer):
    cfg = EXPERIMENT_CONFIG
    context_length = cfg['context_length']
    batch_size = cfg['batch_size']
    num_batches = cfg['num_batches']
    offset_tokens = cfg['deployment_offset_tokens']
    dataset_config = DATASET_REGISTRY[dataset_name]['dataset_config']
    print(f'\nLoading deployment dataset: {dataset_name}', flush=True)
    routing_max_tokens = num_batches * batch_size * context_length
    total_max_tokens = offset_tokens + routing_max_tokens
    training_config = {'dataset_offset': 0, 'context_length': context_length, 'max_tokens': total_max_tokens}
    dataset = load_dataset_from_subconfig(dataset_config=dataset_config, training_config=training_config)
    tokenization_config = {'dataset': {'text_column': dataset_config.get('text_column', 'text')}, 'training': {'context_length': context_length, 'max_tokens': total_max_tokens}}
    tokenized_dataset, dataset_stats = utils.tokenize_and_group_with_token_budget(dataset=dataset, tokenizer=tokenizer, config=tokenization_config)
    offset_blocks = offset_tokens // context_length
    routing_size = num_batches * batch_size
    required_blocks = offset_blocks + routing_size
    if len(tokenized_dataset) < required_blocks:
        raise RuntimeError(f'{dataset_name}: only {len(tokenized_dataset)} token blocks obtained, but {required_blocks} are required ({offset_blocks} skipped + {routing_size} deployment).')
    routing_dataset = tokenized_dataset.select(range(offset_blocks, offset_blocks + routing_size))
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    routing_loader = DataLoader(routing_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    routing_batches = list(routing_loader)
    if len(routing_batches) != num_batches:
        raise RuntimeError(f'{dataset_name}: expected {num_batches} batches, obtained {len(routing_batches)}.')
    print(f'{dataset_name}: skipped {offset_tokens:,} tokens, then built {num_batches} consecutive deployment batches of {batch_size} x {context_length} tokens.', flush=True)
    return (routing_batches, dataset_stats)

def evaluate_batch(model, batch, device):
    return utils.evaluation(model, [batch], device)

def loss_to_perplexity(loss):
    try:
        return math.exp(float(loss))
    except (OverflowError, ValueError):
        return float('inf')

def run_pretrained(batch, device):
    start = time.time()
    model = load_pretrained_model(device)
    loss, _, _, num_tokens = evaluate_batch(model, batch, device)
    total_time = time.time() - start
    clear_model(model, device)
    return {'method': 'Pretrained', 'loss': float(loss), 'perplexity': loss_to_perplexity(loss), 'total_time_sec': total_time, 'num_tokens': num_tokens}

def select_best_single_ft(bank_df, first_batch, device):
    erm_df = bank_df[bank_df['lambda'].astype(float) == 0.0].copy()
    expected = len(EXPERIMENT_CONFIG['sources'])
    if len(erm_df) != expected:
        raise RuntimeError(f'Expected {expected} source ERM checkpoints, found {len(erm_df)}.')
    batch_device = utils.move_batch_to_device(first_batch, device)
    rows = []
    start = time.time()
    for _, row in erm_df.iterrows():
        model = hr.load_bank_checkpoint(row, BANK_REPO_ID).to(device)
        model.eval()
        with torch.no_grad():
            loss = model(**batch_device).loss.item()
        rows.append({'dataset_name': row['dataset_name'], 'subfolder': row['subfolder'], 'batch_loss': loss})
        clear_model(model, device)
    selection_time = time.time() - start
    scores_df = pd.DataFrame(rows)
    best_row = scores_df.loc[scores_df['batch_loss'].idxmin()]
    selected = {'dataset_name': best_row['dataset_name'], 'subfolder': best_row['subfolder'], 'selection_loss': float(best_row['batch_loss'])}
    return (selected, selection_time, scores_df)

def run_best_single_ft(batch, selected_model, selection_time, batch_id, device):
    start = time.time()
    model = load_hub_model(selected_model['subfolder'], device)
    loss, _, _, num_tokens = evaluate_batch(model, batch, device)
    evaluation_time = time.time() - start
    total_time = evaluation_time
    if batch_id == 0:
        total_time += selection_time
    clear_model(model, device)
    return {'method': 'Best Single FT', 'loss': float(loss), 'perplexity': loss_to_perplexity(loss), 'total_time_sec': total_time, 'num_tokens': num_tokens, 'selected_source': selected_model['dataset_name'], 'selection_loss': selected_model['selection_loss']}

def run_mixed_ft(batch, device):
    subfolder = get_model_config()['mixed_ft_subfolder']
    if subfolder is None:
        return {'method': 'Mixed-source FT', 'loss': float('nan'), 'perplexity': float('nan'), 'total_time_sec': float('nan'), 'num_tokens': None, 'status': 'mixed_ft_not_configured'}
    start = time.time()
    model = load_hub_model(subfolder, device)
    loss, _, _, num_tokens = evaluate_batch(model, batch, device)
    total_time = time.time() - start
    clear_model(model, device)
    return {'method': 'Mixed-source FT', 'loss': float(loss), 'perplexity': loss_to_perplexity(loss), 'total_time_sec': total_time, 'num_tokens': num_tokens}

def configure_layernorm_affine(model):
    """
    GPT-2 version of the constrained Tent adaptation idea.

    GPT-2 uses LayerNorm rather than BatchNorm. We therefore freeze
    all parameters and optimize only LayerNorm affine parameters
    (weight / bias).

    This is a Tent-inspired baseline, not the literal original
    BatchNorm-based Tent algorithm.
    """
    for param in model.parameters():
        param.requires_grad = False
    trainable = []
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            if module.weight is not None:
                module.weight.requires_grad = True
                trainable.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad = True
                trainable.append(module.bias)
    return trainable

def run_tent(batch, device):
    """
    Main experimental protocol:

        full incoming B
            -> test-time adaptation on B
            -> evaluate adapted model on the same full B

    The starting point is the mixed-source FT model, matching the
    baseline requested in the method-comparison table.
    """
    tent_cfg = EXPERIMENT_CONFIG['tent']
    mixed_subfolder = get_model_config()['mixed_ft_subfolder']
    if mixed_subfolder is None:
        return {'method': 'Test-time adaptation', 'loss': float('nan'), 'perplexity': float('nan'), 'total_time_sec': float('nan'), 'num_tokens': None, 'status': 'mixed_ft_not_configured'}
    start = time.time()
    model = load_hub_model(mixed_subfolder, device)
    batch_device = utils.move_batch_to_device(batch, device)
    trainable_params = configure_layernorm_affine(model)
    if len(trainable_params) == 0:
        raise RuntimeError('Tent baseline found no LayerNorm affine parameters.')
    num_trainable_params = sum((p.numel() for p in trainable_params))
    model.eval()
    with torch.no_grad():
        loss_before = model(**batch_device).loss.item()
    optimizer = torch.optim.Adam(trainable_params, lr=tent_cfg['lr'])
    model.train()
    for _ in range(tent_cfg['num_steps']):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch_device)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        loss_after = model(**batch_device).loss.item()
    total_time = time.time() - start
    num_tokens = batch_device['attention_mask'].sum().item() if 'attention_mask' in batch_device else batch_device['input_ids'].numel()
    clear_model(model, device)
    return {'method': 'Test-time adaptation', 'loss': float(loss_after), 'perplexity': loss_to_perplexity(loss_after), 'total_time_sec': total_time, 'num_tokens': int(num_tokens), 'loss_before': float(loss_before), 'loss_after': float(loss_after), 'loss_improvement': float(loss_before - loss_after), 'num_trainable_params': int(num_trainable_params), 'num_adaptation_steps': tent_cfg['num_steps'], 'adaptation_lr': tent_cfg['lr']}

def run_oracle(dataset_name, batch, device):
    subfolder = get_oracle_subfolder(dataset_name)
    if subfolder is None:
        return {'method': 'Target-FT Oracle', 'loss': float('nan'), 'perplexity': float('nan'), 'total_time_sec': float('nan'), 'num_tokens': None, 'oracle_subfolder': None}
    start = time.time()
    model = load_hub_model(subfolder, device)
    loss, _, _, num_tokens = evaluate_batch(model, batch, device)
    total_time = time.time() - start
    clear_model(model, device)
    return {'method': 'Target-FT Oracle', 'loss': float(loss), 'perplexity': loss_to_perplexity(loss), 'total_time_sec': total_time, 'num_tokens': num_tokens, 'oracle_subfolder': subfolder}

def compute_shared_routing_scores(bank_df, batch, device):
    print('Scoring model bank once for all routing strategies...', flush=True)
    start = time.time()
    batch_device = utils.move_batch_to_device(batch, device)
    scored_bank_df = hr.compute_bank_batch_losses(bank_df=bank_df, batch=batch_device, device=device, bank_repo_id=BANK_REPO_ID)
    pretrained_subfolder = get_model_config().get('pretrained_subfolder')
    pretrained_loss = rb.compute_pretrained_loss(pretrained_model_name=get_model_name(), pretrained_subfolder=pretrained_subfolder, batch=batch_device, device=device, bank_repo_id=BANK_REPO_ID)
    scoring_time = time.time() - start
    print(f'Shared scoring complete: {len(scored_bank_df)} bank checkpoints + theta_0 in {scoring_time:.1f}s.', flush=True)
    return (scored_bank_df, pretrained_loss, scoring_time)

def run_hard(metadata_path, bank_df, scored_bank_df, pretrained_loss, shared_scoring_time, batch, device):
    start = time.time()
    model, info = rb.run_hard_routing(model_bank_metadata_path=metadata_path, batch=batch, pretrained_model_name=get_model_name(), device=device, bank_repo_id=BANK_REPO_ID, bank_df=bank_df, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss)
    method_time = time.time() - start
    total_time = shared_scoring_time + method_time
    loss = float(info['batch_loss'])
    result = {'method': 'Hard Routing', 'loss': loss, 'perplexity': loss_to_perplexity(loss), 'total_time_sec': total_time, 'method_time_sec': method_time, 'shared_scoring_time_sec': shared_scoring_time, 'selected_candidate': info.get('selected')}
    clear_model(model, device)
    return result

def run_flat(metadata_path, bank_df, scored_bank_df, pretrained_loss, shared_scoring_time, batch, device):
    start = time.time()
    model, info = rb.run_flat_routing(model_bank_metadata_path=metadata_path, batch=batch, pretrained_model_name=get_model_name(), device=device, tau=EXPERIMENT_CONFIG['flat']['tau'], H=EXPERIMENT_CONFIG['hierarchical']['H'], bank_repo_id=BANK_REPO_ID, bank_df=bank_df, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss)
    loss, _, _, num_tokens = evaluate_batch(model, batch, device)
    method_time = time.time() - start
    total_time = shared_scoring_time + method_time
    result = {'method': 'Flat Routing', 'loss': float(loss), 'perplexity': loss_to_perplexity(loss), 'total_time_sec': total_time, 'method_time_sec': method_time, 'shared_scoring_time_sec': shared_scoring_time, 'num_tokens': num_tokens, 'selected_candidates': str(info.get('candidate_names')), 'weights': str(info.get('weights'))}
    clear_model(model, device)
    return result

def run_hierarchical(metadata_path, bank_df, scored_bank_df, pretrained_loss, shared_scoring_time, batch, device):
    start = time.time()
    model, info = hr.run_hierarchical_routing(model_bank_metadata_path=metadata_path, batch=batch, pretrained_model_name=get_model_name(), device=device, config=EXPERIMENT_CONFIG['hierarchical'], bank_repo_id=BANK_REPO_ID, bank_df=bank_df, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss)
    method_time = time.time() - start
    total_time = shared_scoring_time + method_time
    loss = float(info['batch_loss'])
    result = {'method': 'Hierarchical Routing', 'loss': loss, 'perplexity': loss_to_perplexity(loss), 'total_time_sec': total_time, 'method_time_sec': method_time, 'shared_scoring_time_sec': shared_scoring_time, 'selected_sources': str(info.get('selected_sources')), 'selected_lambdas': str(info.get('selected_lambdas')), 'weights': str(info.get('weights')), 'best_start': info.get('best_start_name'), 'best_iteration': info.get('best_iteration'), 'best_is_vertex': info.get('best_is_vertex')}
    clear_model(model, device)
    return (result, info)

def run_first_batch_routing(batch, frozen_model, routing_info, batch_id, device):
    """
    Evaluate the model obtained by hierarchical routing on B1.

    No routing or weight optimization is performed after B1.
    """
    start = time.time()
    loss, _, _, num_tokens = evaluate_batch(frozen_model, batch, device)
    evaluation_time = time.time() - start
    total_time = evaluation_time
    if batch_id == 0:
        total_time += routing_info['routing_time_sec']
    return {'method': 'Top-H + routing + only 1st batch', 'loss': float(loss), 'perplexity': loss_to_perplexity(loss), 'total_time_sec': total_time, 'evaluation_time_sec': evaluation_time, 'routing_time_sec': routing_info['routing_time_sec'] if batch_id == 0 else 0.0, 'num_tokens': num_tokens, 'selected_sources': routing_info['selected_sources'], 'selected_lambdas': routing_info['selected_lambdas'], 'weights': routing_info['weights'], 'best_start': routing_info['best_start'], 'best_iteration': routing_info['best_iteration'], 'best_is_vertex': routing_info['best_is_vertex']}

def build_first_batch_routing_model(metadata_path, bank_df, first_batch, device):
    """
    Run the complete hierarchical routing algorithm once on B1.

    The resulting composed model theta_B1 is then frozen and reused
    unchanged for every deployment batch B1, ..., BT.
    """
    print('\nBuilding Top-H + routing + only 1st batch model...', flush=True)
    scored_bank_df, pretrained_loss, shared_scoring_time = compute_shared_routing_scores(bank_df=bank_df, batch=first_batch, device=device)
    start = time.time()
    model, info = hr.run_hierarchical_routing(model_bank_metadata_path=metadata_path, batch=first_batch, pretrained_model_name=get_model_name(), device=device, config=EXPERIMENT_CONFIG['hierarchical'], bank_repo_id=BANK_REPO_ID, bank_df=bank_df, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss)
    routing_method_time = time.time() - start
    routing_time = shared_scoring_time + routing_method_time
    model.eval()
    routing_info = {'routing_time_sec': routing_time, 'shared_scoring_time_sec': shared_scoring_time, 'routing_method_time_sec': routing_method_time, 'selected_sources': str(info.get('selected_sources')), 'selected_lambdas': str(info.get('selected_lambdas')), 'weights': str(info.get('weights')), 'best_start': info.get('best_start_name'), 'best_iteration': info.get('best_iteration'), 'best_is_vertex': info.get('best_is_vertex')}
    return (model, routing_info, scored_bank_df, info)

def method_result_to_wide(prefix, result):
    row = {}
    for key, value in result.items():
        if key == 'method':
            continue
        row[f'{prefix}_{key}'] = value
    return row

def save_method_details(batch_output_dir, method_key, result):
    pd.DataFrame([result]).to_csv(os.path.join(batch_output_dir, f'{method_key}_result.csv'), index=False)

def run_one_batch(dataset_name, batch_id, batch, metadata_path, bank_df, best_single_ft, best_single_selection_time, first_batch_routing_model, first_batch_routing_info, device):
    cfg = EXPERIMENT_CONFIG
    methods = cfg['methods']
    print('\n============================================================')
    print(f"{dataset_name} - BATCH {batch_id + 1}/{cfg['num_batches']}")
    print('============================================================', flush=True)
    output_dir = get_output_dir()
    dataset_dir = os.path.join(output_dir, dataset_name.replace(' ', '_'))
    batch_output_dir = os.path.join(dataset_dir, f'batch_{batch_id + 1:02d}')
    os.makedirs(batch_output_dir, exist_ok=True)
    base_result = {'experiment_name': cfg['experiment_name'], 'model': cfg['model'], 'model_name': get_model_name(), 'dataset': dataset_name, 'batch_id': batch_id + 1, 'context_length': cfg['context_length'], 'batch_size': cfg['batch_size'], 'num_batches': cfg['num_batches'], 'deployment_offset_tokens': cfg['deployment_offset_tokens'], 'num_sources': len(cfg['sources']), 'sources': str(cfg['sources']), 'H': cfg['hierarchical']['H'], 'hierarchical_num_iters': cfg['hierarchical']['num_iters'], 'hierarchical_lr': cfg['hierarchical']['lr'], 'hierarchical_num_random_starts': cfg['hierarchical']['num_random_starts'], 'flat_tau': cfg['flat']['tau'], 'tent_lr': cfg['tent']['lr'], 'tent_num_steps': cfg['tent']['num_steps']}
    method_results = []
    if 'pretrained' in methods:
        result = run_pretrained(batch, device)
        method_results.append(result)
        save_method_details(batch_output_dir, 'pretrained', result)
    if 'best_single_ft' in methods:
        result = run_best_single_ft(batch=batch, selected_model=best_single_ft, selection_time=best_single_selection_time, batch_id=batch_id, device=device)
        method_results.append(result)
        save_method_details(batch_output_dir, 'best_single_ft', result)
    if 'mixed_ft' in methods:
        result = run_mixed_ft(batch, device)
        method_results.append(result)
        save_method_details(batch_output_dir, 'mixed_ft', result)
    if 'first_batch_routing' in methods:
        result = run_first_batch_routing(batch=batch, frozen_model=first_batch_routing_model, routing_info=first_batch_routing_info, batch_id=batch_id, device=device)
        method_results.append(result)
        save_method_details(batch_output_dir, 'first_batch_routing', result)
    if 'tent' in methods:
        result = run_tent(batch, device)
        method_results.append(result)
        save_method_details(batch_output_dir, 'tent', result)
    if 'oracle' in methods:
        result = run_oracle(dataset_name, batch, device)
        method_results.append(result)
        save_method_details(batch_output_dir, 'oracle', result)
    routing_methods = {'hard', 'flat', 'hierarchical'}
    needs_routing_scores = any((method in methods for method in routing_methods))
    if needs_routing_scores:
        scored_bank_df, pretrained_loss, shared_scoring_time = compute_shared_routing_scores(bank_df, batch, device)
        scored_bank_df.to_csv(os.path.join(batch_output_dir, 'bank_batch_losses.csv'), index=False)
        if 'hard' in methods:
            result = run_hard(metadata_path=metadata_path, bank_df=bank_df, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss, shared_scoring_time=shared_scoring_time, batch=batch, device=device)
            method_results.append(result)
            save_method_details(batch_output_dir, 'hard', result)
        if 'flat' in methods:
            result = run_flat(metadata_path=metadata_path, bank_df=bank_df, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss, shared_scoring_time=shared_scoring_time, batch=batch, device=device)
            method_results.append(result)
            save_method_details(batch_output_dir, 'flat', result)
        if 'hierarchical' in methods:
            result, info = run_hierarchical(metadata_path=metadata_path, bank_df=bank_df, scored_bank_df=scored_bank_df, pretrained_loss=pretrained_loss, shared_scoring_time=shared_scoring_time, batch=batch, device=device)
            method_results.append(result)
            save_method_details(batch_output_dir, 'hierarchical', result)
            if info.get('source_relevance') is not None:
                pd.DataFrame(info['source_relevance']).to_csv(os.path.join(batch_output_dir, 'hierarchical_source_relevance.csv'), index=False)
            if info.get('optimization_trajectories') is not None:
                pd.DataFrame(info['optimization_trajectories']).to_csv(os.path.join(batch_output_dir, 'hierarchical_optimization_trajectories.csv'), index=False)
    comparison_df = pd.DataFrame(method_results)
    preferred_columns = ['method', 'loss', 'perplexity', 'total_time_sec', 'num_tokens']
    ordered_columns = [col for col in preferred_columns if col in comparison_df.columns]
    remaining_columns = [col for col in comparison_df.columns if col not in ordered_columns]
    comparison_df = comparison_df[ordered_columns + remaining_columns]
    comparison_df.to_csv(os.path.join(batch_output_dir, 'comparison_table.csv'), index=False)
    print('\nBatch comparison:')
    print(comparison_df[[col for col in ['method', 'loss', 'perplexity', 'total_time_sec'] if col in comparison_df.columns]].to_string(index=False), flush=True)
    wide_result = copy.deepcopy(base_result)
    prefix_mapping = {'Pretrained': 'pretrained', 'Best Single FT': 'best_single_ft', 'Mixed-source FT': 'mixed_ft', 'Top-H + routing + only 1st batch': 'first_batch_routing', 'Test-time adaptation': 'tent', 'Hard Routing': 'hard', 'Flat Routing': 'flat', 'Hierarchical Routing': 'hierarchical', 'Target-FT Oracle': 'oracle'}
    for result in method_results:
        prefix = prefix_mapping[result['method']]
        wide_result.update(method_result_to_wide(prefix, result))
    pd.DataFrame([wide_result]).to_csv(os.path.join(batch_output_dir, 'general_results.csv'), index=False)
    long_rows = []
    for result in method_results:
        row = copy.deepcopy(base_result)
        row.update(result)
        long_rows.append(row)
    return (wide_result, long_rows)

def build_dataset_summary(dataset_name, long_results):
    df = pd.DataFrame(long_results)
    dataset_df = df[df['dataset'] == dataset_name].copy()
    rows = []
    for method, group in dataset_df.groupby('method'):
        losses = pd.to_numeric(group['loss'], errors='coerce')
        times = pd.to_numeric(group['total_time_sec'], errors='coerce')
        ppls = pd.to_numeric(group['perplexity'], errors='coerce')
        rows.append({'experiment_name': EXPERIMENT_CONFIG['experiment_name'], 'model': EXPERIMENT_CONFIG['model'], 'model_name': get_model_name(), 'dataset': dataset_name, 'method': method, 'num_batches': len(group), 'batch_size': EXPERIMENT_CONFIG['batch_size'], 'context_length': EXPERIMENT_CONFIG['context_length'], 'H': EXPERIMENT_CONFIG['hierarchical']['H'], 'hierarchical_num_iters': EXPERIMENT_CONFIG['hierarchical']['num_iters'], 'loss_mean': losses.mean(), 'loss_std': losses.std(), 'perplexity_mean': ppls.mean(), 'perplexity_std': ppls.std(), 'total_time_mean_sec': times.mean(), 'total_time_std_sec': times.std()})
    return pd.DataFrame(rows)

def build_method_comparison_table(output_dir):
    rows = [{'Method': 'Best single FT', 'Model bank': 'Yes', 'Deployment signal': 'First batch', 'Candidate selection': 'Single source ERM', 'Composition': '--', 'Batch-adaptive': 'No', 'Deployment variables': 'Source choice'}, {'Method': 'Mixed-source FT', 'Model bank': 'No', 'Deployment signal': '--', 'Candidate selection': '--', 'Composition': 'Joint training', 'Batch-adaptive': 'No', 'Deployment variables': 'None'}, {'Method': 'Top-H + routing + only 1st batch', 'Model bank': 'Yes', 'Deployment signal': 'Unlabeled B1', 'Candidate selection': 'Top-H + lambda*_j(B1)', 'Composition': 'w_j(B1)', 'Batch-adaptive': 'First batch only', 'Deployment variables': 'w(B1)'}, {'Method': 'Test-time adaptation (Tent-style on mixed-source FT)', 'Model bank': 'No', 'Deployment signal': 'Unlabeled B', 'Candidate selection': '--', 'Composition': 'Parameter update', 'Batch-adaptive': 'Yes', 'Deployment variables': 'LayerNorm affine parameters'}, {'Method': 'Hard routing', 'Model bank': 'Yes', 'Deployment signal': 'Unlabeled B', 'Candidate selection': 'Best checkpoint', 'Composition': 'Selection', 'Batch-adaptive': 'Yes', 'Deployment variables': 'Discrete choice'}, {'Method': 'Flat routing', 'Model bank': 'Yes', 'Deployment signal': 'Unlabeled B', 'Candidate selection': 'All bank / Top-H', 'Composition': 'w_k(B)', 'Batch-adaptive': 'Yes', 'Deployment variables': 'w'}, {'Method': 'Hierarchical routing (ours)', 'Model bank': 'Yes', 'Deployment signal': 'Unlabeled B', 'Candidate selection': 'Top-H + lambda*_j(B)', 'Composition': 'w_j(B)', 'Batch-adaptive': 'Yes', 'Deployment variables': 'w'}, {'Method': 'Target-FT Oracle', 'Model bank': 'No', 'Deployment signal': 'Full target dataset', 'Candidate selection': '--', 'Composition': 'Direct fine-tuning', 'Batch-adaptive': 'No', 'Deployment variables': 'theta'}]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'method_comparison.csv'), index=False)
    return df

def write_method_comparison_latex(comparison_df, output_dir):
    latex_path = os.path.join(output_dir, 'method_comparison.tex')
    latex = '\\begin{table}[t]\n\\centering\n\\caption{Comparison with alternative adaptation and model-composition approaches.}\n\\label{tab:method-comparison}\n\\small\n\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lcccccc}\n\\toprule\nMethod\n& Model bank\n& Deployment signal\n& Candidate selection\n& Composition\n& Batch-adaptive\n& Deployment variables \\\\\n\\midrule\n'
    latex_rows = ['Best single FT\n& Yes\n& First batch\n& Single source ERM\n& --\n& No\n& Source choice \\\\', '\nMixed-source FT\n& No\n& --\n& --\n& Joint training\n& No\n& None \\\\', '\nTop-$H$ + routing + only $1^{st}$ batch\n& Yes\n& Unlabeled $B_1$\n& Top-$H$ + $\\lambda_j^\\star(B_1)$\n& Shared $\\gamma$\n& Partially\n& Selection / $\\gamma$ \\\\', '\nTest-time adaptation (Tent-style on mixed-source FT)\n& No\n& Unlabeled $B$\n& --\n& Parameter update\n& Yes\n& LayerNorm affine params \\\\', '\nHard routing\n& Yes\n& Unlabeled $B$\n& Best checkpoint\n& Selection\n& Yes\n& Discrete choice \\\\', '\nFlat routing\n& Yes\n& Unlabeled $B$\n& All bank / Top-$H$\n& $w_k(B)$\n& Yes\n& $w$ \\\\', '\n\\textbf{Hierarchical routing (ours)}\n& Yes\n& Unlabeled $B$\n& Top-$H$ + $\\lambda_j^\\star(B)$\n& $w_j(B)$\n& Yes\n& $w$ \\\\', '\n\\midrule\n\nTarget-FT Oracle\n& No\n& Full target dataset\n& --\n& Direct fine-tuning\n& No\n& $\\theta$ \\\\']
    latex += '\n'.join(latex_rows)
    latex += '\n\n\\bottomrule\n\\end{tabular}%\n}\n\n\\end{table}\n'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex)

def save_global_results(wide_results, long_results, summaries, output_dir):
    pd.DataFrame(wide_results).to_csv(os.path.join(output_dir, 'all_batches_results.csv'), index=False)
    pd.DataFrame(long_results).to_csv(os.path.join(output_dir, 'all_method_results.csv'), index=False)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(os.path.join(output_dir, 'summary_results.csv'), index=False)

def main():
    validate_config()
    set_seed(EXPERIMENT_CONFIG['seed'])
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    save_experiment_config(output_dir)
    comparison_df = build_method_comparison_table(output_dir)
    write_method_comparison_latex(comparison_df, output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('============================================================')
    print('EXPERIMENTAL PIPELINE')
    print('============================================================')
    print(f'Device:       {device}')
    print(f"Experiment:   {EXPERIMENT_CONFIG['experiment_name']}")
    print(f"Model key:    {EXPERIMENT_CONFIG['model']}")
    print(f'Model:        {get_model_name()}')
    print(f"Sources:      {EXPERIMENT_CONFIG['sources']}")
    print(f"Deployments:  {EXPERIMENT_CONFIG['deployments']}")
    print(f"Batch size:   {EXPERIMENT_CONFIG['batch_size']}")
    print(f"Context:      {EXPERIMENT_CONFIG['context_length']}")
    print(f"Num batches:  {EXPERIMENT_CONFIG['num_batches']}")
    print(f"H:            {EXPERIMENT_CONFIG['hierarchical']['H']}")
    print(f"Hier. iters:  {EXPERIMENT_CONFIG['hierarchical']['num_iters']}")
    print(f"Methods:      {EXPERIMENT_CONFIG['methods']}")
    print(f'Output:       {output_dir}')
    print('============================================================', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(get_model_name())
    tokenizer.pad_token = tokenizer.eos_token
    metadata_path, bank_df = build_model_bank_metadata(output_dir)
    print('\nRouting model bank:')
    print(bank_df[['dataset_name', 'lambda', 'subfolder']].to_string(index=False), flush=True)
    all_wide_results = []
    all_long_results = []
    all_summaries = []
    for dataset_name in EXPERIMENT_CONFIG['deployments']:
        print('\n\n############################################################')
        print(f'DEPLOYMENT: {dataset_name}')
        print('############################################################')
        routing_batches, dataset_stats = load_deployment_data(dataset_name, tokenizer)
        dataset_dir = os.path.join(output_dir, dataset_name.replace(' ', '_'))
        os.makedirs(dataset_dir, exist_ok=True)
        try:
            with open(os.path.join(dataset_dir, 'dataset_stats.json'), 'w', encoding='utf-8') as f:
                json.dump(dataset_stats, f, indent=2, default=str)
        except Exception:
            pass
        best_single_ft = None
        best_single_selection_time = 0.0
        if 'best_single_ft' in EXPERIMENT_CONFIG['methods']:
            print(f'\nSelecting Best Single FT for {dataset_name} on first incoming batch...', flush=True)
            best_single_ft, best_single_selection_time, best_single_scores_df = select_best_single_ft(bank_df=bank_df, first_batch=routing_batches[0], device=device)
            best_single_scores_df.to_csv(os.path.join(dataset_dir, 'best_single_ft_selection.csv'), index=False)
            print(f"Best Single FT: {best_single_ft['dataset_name']} (loss={best_single_ft['selection_loss']:.6f}, time={best_single_selection_time:.1f}s)", flush=True)
        first_batch_routing_model = None
        first_batch_routing_info = None
        if 'first_batch_routing' in EXPERIMENT_CONFIG['methods']:
            first_batch_routing_model, first_batch_routing_info, first_batch_scored_bank_df, first_batch_full_info = build_first_batch_routing_model(metadata_path=metadata_path, bank_df=bank_df, first_batch=routing_batches[0], device=device)
            first_batch_scored_bank_df.to_csv(os.path.join(dataset_dir, 'first_batch_routing_bank_losses.csv'), index=False)
            pd.DataFrame([first_batch_routing_info]).to_csv(os.path.join(dataset_dir, 'first_batch_routing_selection.csv'), index=False)
            print(f"First-batch routing model fixed for all {EXPERIMENT_CONFIG['num_batches']} batches.", flush=True)
        dataset_wide_results = []
        dataset_long_results = []
        for batch_id, batch in enumerate(routing_batches):
            wide_result, long_rows = run_one_batch(dataset_name=dataset_name, batch_id=batch_id, batch=batch, metadata_path=metadata_path, bank_df=bank_df, best_single_ft=best_single_ft, best_single_selection_time=best_single_selection_time, first_batch_routing_model=first_batch_routing_model, first_batch_routing_info=first_batch_routing_info, device=device)
            dataset_wide_results.append(wide_result)
            dataset_long_results.extend(long_rows)
            all_wide_results.append(wide_result)
            all_long_results.extend(long_rows)
            if EXPERIMENT_CONFIG['progressive_save']:
                pd.DataFrame(dataset_wide_results).to_csv(os.path.join(dataset_dir, 'all_batches_results.csv'), index=False)
                pd.DataFrame(dataset_long_results).to_csv(os.path.join(dataset_dir, 'all_method_results.csv'), index=False)
                save_global_results(all_wide_results, all_long_results, all_summaries, output_dir)
        if first_batch_routing_model is not None:
            clear_model(first_batch_routing_model, device)
            first_batch_routing_model = None
        summary_df = build_dataset_summary(dataset_name, dataset_long_results)
        summary_df.to_csv(os.path.join(dataset_dir, 'summary_results.csv'), index=False)
        all_summaries.append(summary_df)
        save_global_results(all_wide_results, all_long_results, all_summaries, output_dir)
        print(f'\nSummary for {dataset_name}:')
        print(summary_df[['method', 'loss_mean', 'loss_std', 'perplexity_mean', 'total_time_mean_sec']].to_string(index=False), flush=True)
    save_global_results(all_wide_results, all_long_results, all_summaries, output_dir)
    print('\n============================================================')
    print('EXPERIMENT COMPLETE')
    print('============================================================')
    print(f"all_batches_results.csv : {os.path.join(output_dir, 'all_batches_results.csv')}")
    print(f"summary_results.csv     : {os.path.join(output_dir, 'summary_results.csv')}")
    print(f"all_method_results.csv  : {os.path.join(output_dir, 'all_method_results.csv')}")
    print(f"method_comparison.csv   : {os.path.join(output_dir, 'method_comparison.csv')}")
    print(f"method_comparison.tex   : {os.path.join(output_dir, 'method_comparison.tex')}")
    print('============================================================')
if __name__ == '__main__':
    main()