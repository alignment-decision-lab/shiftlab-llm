from datasets import load_dataset, Dataset

import re

# -- WikiText --

def load_wikitext_dataset(config):
    dataset = load_dataset(config["dataset"]["name"], config["dataset"].get("config", None))

    dataset = dataset[config["dataset"]["split"]]

    if config["dataset"]["text_column"] != "text":
        dataset = dataset.rename_column(config["dataset"]["text_column"], "text")

    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 5)

    if config["training"]["dataset_size"] is not None:
        dataset = dataset.select(range(min(config["training"]["dataset_size"], len(dataset))))
    return dataset

# -- WikiSource --

def clean_wikisource_text(example):
    text = example["text"]

    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"Sommaire\s*:.*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    example["text"] = text.strip()

    return example


def load_wikisource_dataset(config):
    dataset = load_dataset(config["dataset"]["name"], config["dataset"].get("config", None))

    dataset = dataset[config["dataset"]["split"]]

    if config["dataset"]["text_column"] != "text":
        dataset = dataset.rename_column(config["dataset"]["text_column"], "text")

    dataset = dataset.map(clean_wikisource_text)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 5)

    if config["training"]["dataset_size"] is not None:
        dataset = dataset.select(range(min(config["training"]["dataset_size"], len(dataset))))
    return dataset

# -- HistText --

def clean_histtext_text(example):
    text = example["content"]

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    example["content"] = text.strip()

    return example


def load_histtext_dataset(config):
    dataset = load_dataset(
        config["dataset"]["name"],
        data_files=config["dataset"]["data_files"],
        split=config["dataset"]["split"]
    )

    if config["dataset"].get("cleaning") == "basic":
        dataset = dataset.map(clean_histtext_text)

    if config["dataset"]["text_column"] != "text":
        dataset = dataset.rename_column(
            config["dataset"]["text_column"],
            "text"
        )

    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 5)

    if config["training"]["dataset_size"] is not None:
        dataset = dataset.select(
            range(min(config["training"]["dataset_size"], len(dataset)))
        )

    return dataset

# -- PG-19 --

def clean_pg19_text(example):
    text = example["text"]

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    example["text"] = text.strip()

    return example


def load_pg19_dataset(config):
    dataset = load_dataset(
        config["dataset"]["name"],
        split=config["dataset"]["split"],
        streaming=config["dataset"].get("streaming", False)
    )

    if config["dataset"].get("cleaning") == "basic":
        dataset = dataset.map(clean_pg19_text)

    if config["dataset"]["text_column"] != "text":
        dataset = dataset.rename_column(config["dataset"]["text_column"], "text")

    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 5)

    dataset_size = config["training"].get("dataset_size")

    if dataset_size is None:
        dataset_size = config.get("diagnostic", {}).get("total_size")

    if config["dataset"].get("streaming", False):
        if dataset_size is not None:
            dataset = dataset.take(dataset_size)
        dataset = Dataset.from_list(list(dataset))

    else:
        if dataset_size is not None:
            dataset = dataset.select(range(min(dataset_size, len(dataset))))

    return dataset

# --Gallica--
def clean_gallica_text(example):
    text = example["text"]

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    example["text"] = text.strip()

    return example


def load_gallica_dataset(config):
    dataset = load_dataset(
    config["dataset"]["name"],
    config["dataset"].get("config", None),
    split=config["dataset"]["split"],
    streaming=config["dataset"].get("streaming", False)
    )

    if config["dataset"]["text_column"] != "text":
        dataset = dataset.rename_column(
            config["dataset"]["text_column"],
            "text"
        )

    if config["dataset"].get("cleaning") == "basic":
        dataset = dataset.map(clean_gallica_text)

    dataset = dataset.filter(
        lambda x: len(x["text"].strip()) > 5
    )

    dataset_size = config["training"].get("dataset_size")

    if dataset_size is None:
        dataset_size = config.get("diagnostic", {}).get("total_size")

    if config["dataset"].get("streaming", False):
        if dataset_size is not None:
            dataset = dataset.take(dataset_size)
        dataset = Dataset.from_list(list(dataset))

    else:
        if dataset_size is not None:
            dataset = dataset.select(range(min(dataset_size, len(dataset))))

    return dataset

# -- Any dataset from Hugging Face --

def normalize_text_column(example):
    text = example["text"]

    if isinstance(text, list):
        text = " ".join(map(str, text))

    example["text"] = str(text)
    return example

def load_translation_dataset(config):
    dataset = load_dataset(
        config["dataset"]["name"],
        config["dataset"].get("config", None),
        split=config["dataset"]["split"],
        streaming=config["dataset"].get("streaming", False),
    )

    lang = config["dataset"].get("language", "en")

    if config["dataset"].get("streaming", False):
        dataset_size = config["training"].get("dataset_size")
        dataset = dataset.take(dataset_size)
        dataset = Dataset.from_list(list(dataset))

    def extract_language(example):
        example["text"] = example["translation"][lang]
        return example

    dataset = dataset.map(extract_language)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 5)

    return dataset

def load_hf_text_dataset(config):
    dataset = load_dataset(
        config["dataset"]["name"],
        config["dataset"].get("config", None),
        split=config["dataset"]["split"],
        streaming=config["dataset"].get("streaming", False),
    )

    dataset_size = config["training"].get("dataset_size")
    dataset_offset = int(
        config["training"].get("dataset_offset", 0)
    )

    if dataset_offset < 0:
        raise ValueError(
            "dataset_offset must be greater than or equal to 0."
        )

    if config["dataset"].get("streaming", False):
        if dataset_size is None:
            raise ValueError(
                "dataset_size must be provided when streaming=True"
            )

        if dataset_offset > 0:
            dataset = dataset.skip(dataset_offset)

        dataset = dataset.take(dataset_size)

        # Convert the finite iterable into a standard Hugging Face Dataset.
        dataset = Dataset.from_list(list(dataset))

    else:
        if dataset_size is not None:
            start = min(dataset_offset, len(dataset))
            end = min(
                dataset_offset + dataset_size,
                len(dataset),
            )

            dataset = dataset.select(
                range(start, end)
            )

    if config["dataset"]["text_column"] != "text":
        dataset = dataset.rename_column(
            config["dataset"]["text_column"],
            "text",
        )

    dataset = dataset.map(normalize_text_column)

    dataset = dataset.filter(
        lambda x: len(x["text"].strip()) > 5
    )

    return dataset


# -- General dataset loader --

def load_dataset_from_config(config):
    dataset_type = config["dataset"]["type"]

    if dataset_type == "wikitext":
        return load_wikitext_dataset(config)

    elif dataset_type == "wikisource":
        return load_wikisource_dataset(config)

    elif dataset_type == "histtext":
        return load_histtext_dataset(config)

    elif dataset_type == "pg19":
        return load_pg19_dataset(config)
    
    elif dataset_type == "gallica":
        return load_gallica_dataset(config)
    
    elif dataset_type == "hf_text":
        return load_hf_text_dataset(config)

    elif dataset_type == "translation":
        return load_translation_dataset(config)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def load_dataset_from_subconfig(dataset_config, training_config):
    local_config = {
        "dataset": dataset_config,
        "training": training_config
    }
    return load_dataset_from_config(local_config)