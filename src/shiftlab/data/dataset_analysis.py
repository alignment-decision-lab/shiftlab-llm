from shiftlab.train.diagnostic_experiment import shift_measurement
from shiftlab.data.load_datasets import load_dataset_from_subconfig
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import os
import pandas as pd
from shiftlab.train.diagnostic_experiment import utils
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform



datasets_config = {
    "Wikipedia": {
    "dataset": {
        "type": "hf_text",
        "name": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
    "training": {
        "dataset_size": 1000,
        "batch_size": 16,
        "context_length": 512,
    },
},

"BookCorpus": {
    "dataset": {
        "type": "hf_text",
        "name": "Yuti/bookcorpus",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
    "training": {
        "dataset_size": 40000,
        "batch_size": 16,
        "context_length": 512,
    },
},

"Pile-CC": {
    "dataset": {
        "type": "hf_text",
        "name": "timaeus/pile-pile-cc",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
    "training": {
        "dataset_size": 1000,
        "batch_size": 16,
        "context_length": 512,
    },
},
    
"OWT": {
    "dataset": {
        "type": "hf_text",
        "name": "Skylion007/openwebtext",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
    "training": {
        "dataset_size": 1000,
        "batch_size": 16,
        "context_length": 512,
    },
},

"PG-19": {
    "dataset": {
        "type": "hf_text",
        "name": "emozilla/pg19",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
    "training": {
        "dataset_size": 1000,
        "batch_size": 16,
        "context_length": 512,
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
    "training": {
        "dataset_size": 40000,
        "batch_size": 16,
        "context_length": 512,
    },
},

"PubMed Central": {
    "dataset": {
        "type": "hf_text",
        "name": "datajuicer/the-pile-pubmed-central-refined-by-data-juicer",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
    "training": {
        "dataset_size": 1000,
        "batch_size": 16,
        "context_length": 512,
    },
},

"StackExchange": {
    "dataset": {
        "type": "hf_text",
        "name": "flax-sentence-embeddings/stackexchange_title_body_jsonl",
        "split": "train",
        "text_column": "texts",
        "streaming": True,
    },
    "training": {
        "dataset_size": 40000,
        "batch_size": 16,
        "context_length": 512,
    },
},

"ArXiv": {
    "dataset": {
        "type": "hf_text",
        "name": "timaeus/pile-arxiv",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
    "training": {
        "dataset_size": 1000,
        "batch_size": 16,
        "context_length": 512,
    },
},

"DM Mathematics": {
    "dataset": {
        "type": "hf_text",
        "name": "timaeus/pile-dm_mathematics",  
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
    "training": {
        "dataset_size": 40000,
        "batch_size": 16,
        "context_length": 512,
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
    "training": {
        "dataset_size": 1000,
        "batch_size": 16,
        "context_length": 512,
    },
},

"EuroParl": {
    "dataset": {
            "type": "translation",
            "config": "en-fr",
            "name": "Helsinki-NLP/europarl",
            "split": "train",
            "language": "en",
            "streaming": True,
        },
        "training": {
            "dataset_size": 40000,
            "batch_size": 16,
            "context_length": 512,
        },
},

"FreeLaw": {
    "dataset": {
        "type": "hf_text",
        "name": "timaeus/pile-freelaw",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
    "training": {
        "dataset_size": 1000,
        "batch_size": 16,
        "context_length": 512,
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
    "training": {
        "dataset_size": 1000,
        "batch_size": 16,
        "context_length": 512,
    },
},

"Hacker News": {
    "dataset": {
        "type": "hf_text",
        "name": "timaeus/pile-hackernews",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
    "training": {
        "dataset_size": 1000,
        "batch_size": 16,
        "context_length": 512,
    },
},

"Enron Emails": {
    "dataset": {
        "type": "hf_text",
        "name": "timaeus/pile-enron_emails",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
    "training": {
        "dataset_size": 1000,
        "batch_size": 16,
        "context_length": 512,
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
    "training": {
        "dataset_size": 600,
        "batch_size": 16,
        "context_length": 512,
    },
},

"USPTO": {
    "dataset": {
        "type": "hf_text",
        "name": "common-pile/uspto_filtered",
        "split": "train",
        "text_column": "text",
        "streaming": True,
    },
    "training": {
        "dataset_size": 1000,
        "batch_size": 16,
        "context_length": 512,
    },
},

}



def build_dataloaders(datasets_config, tokenizer, data_collator):
    """ Build dataloaders for each dataset in the datasets_config. """
    dataloaders = {}
    for dataset_name, dataset_config in datasets_config.items():
        print(f"\n Loading dataset {dataset_name}...")
        dataset = load_dataset_from_subconfig(
            dataset_config["dataset"],
            dataset_config["training"],
        )
        tokenized_dataset = utils.tokenize_and_group_dataset(dataset, tokenizer, dataset_config)
        dataloader = shift_measurement.create_deploy_dataloader(tokenized_dataset, data_collator, dataset_config)

        dataloaders[dataset_name] = dataloader
    return dataloaders



def compute_token_kl_matrices(dataloaders, vocab_size, max_tokens=512000, epsilon=1e-8):
    """ Compute symetric and asymetric token-KL matrices between datasets. The KL is computed as 1/2 * (KL(P||Q) + KL(Q||P)) where P and Q are the token distributions of the two datasets. The KL is computed for each pair of datasets and stored in a matrix."""
    dataset_names = list(dataloaders.keys())
    distributions = {}
    print("\n Computing token distributions for each dataset...")
    for name in dataset_names:
        dataloader = dataloaders[name]
        distribution, total_tokens = shift_measurement.compute_dataset_token_distribution_with_budget(
            dataloader,
            vocab_size,
            max_tokens,
            epsilon=epsilon,
        )
        distributions[name] = distribution
        print(f"Dataset {name}: {total_tokens:,} valid tokens used to compute distribution.")
    
    n_datasets = len(dataset_names)
    asymmetric_matrix = torch.zeros((n_datasets, n_datasets),dtype=torch.float64)

    for i, dataset_i in enumerate(dataset_names):
        for j, dataset_j in enumerate(dataset_names):
            kl_ij = shift_measurement.compute_kl(distributions[dataset_i], distributions[dataset_j])
            asymmetric_matrix[i, j] = kl_ij

    symetric_matrix = 0.5 * (asymmetric_matrix + asymmetric_matrix.T)
    asymetric_df = pd.DataFrame(asymmetric_matrix.numpy(), index=dataset_names, columns=dataset_names)
    symetric_df = pd.DataFrame(symetric_matrix.numpy(), index=dataset_names, columns=dataset_names)

    return symetric_df, asymetric_df

def plot_heatmap(symetric_df, output_path):
    """ Plot a heatmap of the symetric token-KL matrix. """
    fig, ax = plt.subplots(figsize=(12, 10))

    image = ax.imshow(symetric_df.values, interpolation='nearest', aspect='equal')
    # Dataset names
    ax.set_xticks(range(len(symetric_df.columns)))
    ax.set_yticks(range(len(symetric_df.index)))
    ax.set_xticklabels(symetric_df.columns, rotation=45, ha='right')
    ax.set_yticklabels(symetric_df.index)
    # Display values inside each cells
    for i in range(len(symetric_df.index)):
        for j in range(len(symetric_df.columns)):
            text = ax.text(j, i, f"{symetric_df.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8, color="black")
    cbar = ax.figure.colorbar(image)
    cbar.ax.set_ylabel("Token-KL")
    ax.set_title("Token-KL similarity between datasets")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)



def main():
    """ Main function to compute the token-KL matrix between datasets. """
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloaders = build_dataloaders(datasets_config, tokenizer, data_collator)
    vocab_size = tokenizer.vocab_size
    symetric_df, asymetric_df = compute_token_kl_matrices(dataloaders, vocab_size, max_tokens=512000, epsilon=1e-8)

    output_dir = "outputs/dataset_analysis"
    os.makedirs(output_dir, exist_ok=True)

    symetric_df.to_csv(
        os.path.join(
            output_dir,
            "symetric_token_kl_matrix.csv",
        )
    )
    asymetric_df.to_csv(
        os.path.join(
            output_dir,
            "asymetric_token_kl_matrix.csv",
        )
    )

    print(symetric_df)
    print(asymetric_df)
    plot_heatmap(symetric_df, os.path.join(output_dir, "symetric_token_kl_heatmap.png"))
    plot_heatmap(asymetric_df, os.path.join(output_dir, "asymetric_token_kl_heatmap.png"))

    # Show the reordered heatmap
    distance_matrix = symetric_df.values

    Z = linkage(
        squareform(distance_matrix),
        method="average",
    )

    order = leaves_list(Z)

    ordered_df = symetric_df.iloc[
        order,
        order,
    ]
    ordered_df.to_csv(
        os.path.join(
            output_dir,
            "symmetric_token_kl_matrix_reordered.csv",
        )
    )
    plot_heatmap(ordered_df, os.path.join(output_dir, "symetric_token_kl_heatmap_reordered.png"))
    print("Reordered symmetric token-KL matrix:")
    print(ordered_df)

if __name__ == "__main__":
    main()