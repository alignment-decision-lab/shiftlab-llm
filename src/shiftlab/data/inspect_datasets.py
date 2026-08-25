from datasets import load_dataset, get_dataset_config_names
import re


def clean_wikisource_text(example):
    text = example["text"]

    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"Sommaire\s*:.*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    example["text"] = text.strip()

    return example
def compute_average_length(dataset, text_column="text", clean=False, max_samples=10000):

    total_length = 0
    n = min(max_samples, len(dataset))

    for i in range(n):

        example = dataset[i]

        if clean:
            example = clean_wikisource_text(example)

        text = example[text_column]

        total_length += len(text.split())

    average = total_length / n

    return average


def inspect_examples(dataset, name, text_column="text", n_examples=3, clean=False):

    print("\n" + "=" * 100)
    print(f"DATASET : {name}")
    print("=" * 100)

    print(f"Nombre d'exemples : {len(dataset)}")
    print(f"Colonnes : {dataset.column_names}")

    for i in range(min(n_examples, len(dataset))):

        example = dataset[i]

        if clean:
            example = clean_wikisource_text(example)

        text = example[text_column]

        print("\n" + "-" * 80)
        print(f"Example {i}")
        print("-" * 80)

        print(text[:1500])
        print("\nLength :", len(text))

# =========================
# Heterogeneous datasets
# =========================

def try_load_and_inspect(dataset_id, name, split="train", config=None,
                         text_column="text", n_examples=3, streaming=False):

    print("\n" + "#" * 100)
    print(f"Trying dataset: {name}")
    print("#" * 100)

    try:
        if config is None:
            dataset = load_dataset(dataset_id, split=split, streaming=streaming)
        else:
            dataset = load_dataset(dataset_id, config, split=split, streaming=streaming)

        if streaming:
            dataset_small = list(dataset.take(1000))
            print(f"Loaded in streaming mode. Sample size: {len(dataset_small)}")
            print("Columns:", dataset_small[0].keys())

            for i, example in enumerate(dataset_small[:n_examples]):
                print("\n" + "-" * 80)
                print(f"Example {i}")
                print("-" * 80)
                print(example[text_column][:1500])
                print("\nLength:", len(example[text_column]))

        else:
            inspect_examples(
                dataset=dataset,
                name=name,
                text_column=text_column,
                n_examples=n_examples,
                clean=False
            )

            avg = compute_average_length(dataset, text_column=text_column)
            print(f"Average length {name}: {avg}")

    except Exception as e:
        print(f"FAILED to load {name}")
        print("Error:", e)
    


if __name__ == "__main__":

    # # =========================
    # # WikiText
    # # =========================

    # wikitext = load_dataset(
    #     "wikitext",
    #     "wikitext-2-raw-v1",
    #     split="train"
    # )

    # inspect_examples(
    #     dataset=wikitext,
    #     name="WikiText RAW (English)",
    #     text_column="text",
    #     n_examples=3,
    #     clean=False
    # )

    # # =========================
    # # Wikisource brut
    # # =========================

    # wikisource = load_dataset(
    #     "OpenLLM-France/wikisource",
    #     split="train"
    # )

    # inspect_examples(
    #     dataset=wikisource,
    #     name="Wikisource RAW (French)",
    #     text_column="text",
    #     n_examples=3,
    #     clean=False
    # )

    # # =========================
    # # Wikisource cleaned
    # # =========================

    # inspect_examples(
    #     dataset=wikisource,
    #     name="Wikisource CLEANED (French)",
    #     text_column="text",
    #     n_examples=3,
    #     clean=True
    # )
    # avg_wikitext = compute_average_length(wikitext)
    # avg_wikisource = compute_average_length(wikisource)
    # avg_wikisource_clean = compute_average_length(wikisource, clean=True)

    # print("\nAverage lengths:")
    # print("WikiText:", avg_wikitext)
    # print("Wikisource:", avg_wikisource)
    # print("Wikisource clean:", avg_wikisource_clean)

    # # =========================
    # # ProgressGym HistText
    # # =========================

    # histtext = load_dataset(
    #     "json",
    #     data_files="hf://datasets/PKU-Alignment/ProgressGym-HistText/C017/*.json",
    #     split="train"
    # )

    # inspect_examples(
    #     dataset=histtext,
    #     name="ProgressGym HistText C17",
    #     text_column="content",
    #     n_examples=3,
    #     clean=False
    # )

    # avg_histtext = compute_average_length(
    #     histtext,
    #     text_column="content"
    # )

    # print("HistText C17:", avg_histtext)

        # =========================
    # OSCAR / multilingual web
    # =========================

    # 
    #  
    
    #     # =========================
    # # FineWeb sample
    # # =========================

    # try_load_and_inspect(
    #     dataset_id="HuggingFaceFW/fineweb",
    #     name="FineWeb sample",
    #     split="train",
    #     text_column="text",
    #     streaming=True
    # )

    # # =========================
    # # SlimPajama
    # # =========================

    # try_load_and_inspect(
    #     dataset_id="DKYoon/SlimPajama-6B",
    #     name="SlimPajama 6B",
    #     split="train",
    #     text_column="text",
    #     streaming=True
    # )
    # # =========================
    # # RedPajama sample
    # # =========================

    # try_load_and_inspect(
    #     dataset_id="togethercomputer/RedPajama-Data-V2",
    #     name="RedPajama V2",
    #     split="train",
    #     text_column="raw_content",
    #     streaming=True
    # )

    # # =========================
    # # PG-19
    # # =========================

    # try_load_and_inspect(
    # dataset_id="emozilla/pg19",
    # name="PG-19",
    # split="train",
    # text_column="text",
    # streaming=True
    # )

    # try_load_and_inspect(
    # dataset_id="PleIAs/Post-OCR-Correction",
    # name="Post-OCR Correction",
    # config="french",
    # split="train",
    # text_column="text",
    # streaming=True
    # )

# ---------- OpenWebText2 (The Pile) ----------

    try_load_and_inspect(
    dataset_id="suolyer/pile_openwebtext2",
    name="OpenWebText2 (The Pile)",
    split="validation",
    text_column="text",
    streaming=True,
)