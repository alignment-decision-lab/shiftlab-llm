import argparse
import csv
import os
import random
import traceback
from statistics import mean, median

from datasets import load_dataset


DATASET_CANDIDATES = [
    # Direct / near-direct The Pile components
    {
        "component": "pile_cc",
        "display_name": "Pile-CC",
        "the_pile_component": "Pile-CC",
        "source_type": "direct_the_pile_subset",
        "fidelity": "exact_subset",
        "hf_path": "timaeus/pile-pile-cc",
        "hf_config": None,
        "split": "train",
        "text_candidates": ["text"],
        "expected": "100k subset from Pile source",
    },
    {
        "component": "pubmed_central",
        "display_name": "PubMed Central",
        "the_pile_component": "PubMed Central",
        "source_type": "alternative_same_source_family",
        "fidelity": "very_close",
        "hf_path": "ccdv/pubmed-summarization",
        "hf_config": "document",
        "split": "train",
        "text_candidates": ["article", "abstract", "text"],
        "expected": "PubMed/PubMed-like scientific biomedical articles",
    },
    {
        "component": "books3",
        "display_name": "Books3",
        "the_pile_component": "Books3",
        "source_type": "unavailable_or_restricted",
        "fidelity": "not_tested",
        "hf_path": "EleutherAI/pile",
        "hf_config": "books3",
        "split": "train",
        "text_candidates": ["text"],
        "expected": "likely unavailable through modern HF datasets",
    },
    {
        "component": "openwebtext2",
        "display_name": "OpenWebText2",
        "the_pile_component": "OpenWebText2",
        "source_type": "alternative_same_dataset_family",
        "fidelity": "very_close",
        "hf_path": "Skylion007/openwebtext",
        "hf_config": None,
        "split": "train",
        "text_candidates": ["text"],
        "expected": "OpenWebText alternative",
    },
    {
        "component": "arxiv",
        "display_name": "ArXiv",
        "the_pile_component": "ArXiv",
        "source_type": "alternative_same_domain",
        "fidelity": "same_domain",
        "hf_path": "gfissore/arxiv-abstracts-2021",
        "hf_config": None,
        "split": "train",
        "text_candidates": ["abstract", "text", "title"],
        "expected": "ArXiv abstracts alternative",
    },
    {
        "component": "github",
        "display_name": "Github",
        "the_pile_component": "Github",
        "source_type": "direct_the_pile_subset",
        "fidelity": "exact_subset",
        "hf_path": "timaeus/pile-github",
        "hf_config": None,
        "split": "train",
        "text_candidates": ["text", "code"],
        "expected": "100k subset from Pile source",
    },
    {
        "component": "freelaw",
        "display_name": "FreeLaw / Caselaw Access Project",
        "the_pile_component": "FreeLaw",
        "source_type": "alternative_same_source_family",
        "fidelity": "very_close",
        "hf_path": "common-pile/caselaw_access_project",
        "hf_config": None,
        "split": "train",
        "text_candidates": ["text", "casebody", "opinion", "body", "content"],
        "expected": "Caselaw Access Project alternative",
    },
    {
        "component": "stackexchange",
        "display_name": "Stack Exchange",
        "the_pile_component": "Stack Exchange",
        "source_type": "alternative_same_source_family",
        "fidelity": "very_close",
        "hf_path": "flax-sentence-embeddings/stackexchange_title_body_jsonl",
        "hf_config": None,
        "split": "train",
        "text_candidates": ["texts", "body", "text", "title"],
        "expected": "StackExchange title/body alternative",
    },
    {
        "component": "uspto",
        "display_name": "USPTO Backgrounds",
        "the_pile_component": "USPTO Backgrounds",
        "source_type": "original_pile_unavailable",
        "fidelity": "not_found_yet",
        "hf_path": "EleutherAI/pile",
        "hf_config": "uspto",
        "split": "train",
        "text_candidates": ["text"],
        "expected": "original Pile entry likely fails with pile.py",
    },
    {
        "component": "pubmed_abstracts",
        "display_name": "PubMed Abstracts",
        "the_pile_component": "PubMed Abstracts",
        "source_type": "direct_the_pile_subset",
        "fidelity": "exact_subset",
        "hf_path": "timaeus/pile-pubmed_abstracts",
        "hf_config": None,
        "split": "train",
        "text_candidates": ["text", "article", "abstract"],
        "expected": "100k subset from Pile source",
    },
    {
        "component": "gutenberg_pg_19",
        "display_name": "Gutenberg (PG-19)",
        "the_pile_component": "Gutenberg (PG-19)",
        "source_type": "alternative_same_dataset_family",
        "fidelity": "very_close",
        "hf_path": "emozilla/pg19",
        "hf_config": None,
        "split": "train",
        "text_candidates": ["text"],
        "expected": "PG-19 accessible parquet version",
    },
    {
        "component": "opensubtitles",
        "display_name": "OpenSubtitles",
        "the_pile_component": "OpenSubtitles",
        "source_type": "alternative_same_source_family",
        "fidelity": "very_close_but_parallel",
        "hf_path": "sentence-transformers/parallel-sentences-opensubtitles",
        "hf_config": None,
        "split": "train",
        "text_candidates": ["english", "sentence1", "text", "translation"],
        "expected": "OpenSubtitles-derived parallel sentences",
    },
    {
        "component": "wikipedia_en",
        "display_name": "Wikipedia (en)",
        "the_pile_component": "Wikipedia (en)",
        "source_type": "alternative_same_dataset_family",
        "fidelity": "very_close",
        "hf_path": "wikimedia/wikipedia",
        "hf_config": "20231101.en",
        "split": "train",
        "text_candidates": ["text"],
        "expected": "modern English Wikipedia dump",
    },
    {
        "component": "dm_mathematics",
        "display_name": "DM Mathematics",
        "the_pile_component": "DM Mathematics",
        "source_type": "direct_the_pile_subset",
        "fidelity": "exact_subset",
        "hf_path": "timaeus/pile-dm_mathematics",
        "hf_config": None,
        "split": "train",
        "text_candidates": ["text"],
        "expected": "100k subset from Pile source",
    },
    {
        "component": "ubuntu_irc",
        "display_name": "Ubuntu IRC",
        "the_pile_component": "Ubuntu IRC",
        "source_type": "alternative_same_domain",
        "fidelity": "same_domain",
        "hf_path": "sedthh/ubuntu_dialogue_qa_corpus",
        "hf_config": None,
        "split": "train",
        "text_candidates": ["text", "dialogue", "question", "answer"],
        "expected": "Ubuntu dialogue/chat alternative",
    },
    {
        "component": "bookcorpus2",
        "display_name": "BookCorpus2",
        "the_pile_component": "BookCorpus2",
        "source_type": "unavailable_or_restricted",
        "fidelity": "not_tested",
        "hf_path": "EleutherAI/pile",
        "hf_config": "bookcorpus2",
        "split": "train",
        "text_candidates": ["text"],
        "expected": "likely unavailable through modern HF datasets",
    },
    {
        "component": "europarl",
        "display_name": "EuroParl",
        "the_pile_component": "EuroParl",
        "source_type": "alternative_same_source_family",
        "fidelity": "very_close_but_parallel",
        "hf_path": "Helsinki-NLP/europarl",
        "hf_config": "en-fr",
        "split": "train",
        "text_candidates": ["translation", "text"],
        "expected": "EuroParl parallel corpus; English side can be extracted",
    },
    {
        "component": "hackernews",
        "display_name": "HackerNews",
        "the_pile_component": "HackerNews",
        "source_type": "alternative_same_source_family",
        "fidelity": "very_close",
        "hf_path": "typedef-ai/hacker-news-dataset",
        "hf_config": None,
        "split": "2025_comments",
        "text_candidates": ["text", "title"],
        "expected": "HackerNews comments split",
    },
    {
        "component": "youtubesubtitles",
        "display_name": "YoutubeSubtitles",
        "the_pile_component": "YoutubeSubtitles",
        "source_type": "original_pile_unavailable",
        "fidelity": "not_found_yet",
        "hf_path": "EleutherAI/pile",
        "hf_config": "youtube_subtitles",
        "split": "train",
        "text_candidates": ["text"],
        "expected": "original Pile entry likely fails with pile.py",
    },
    {
        "component": "philpapers",
        "display_name": "PhilPapers",
        "the_pile_component": "PhilPapers",
        "source_type": "original_pile_unavailable",
        "fidelity": "not_found_yet",
        "hf_path": "EleutherAI/pile",
        "hf_config": "philpapers",
        "split": "train",
        "text_candidates": ["text"],
        "expected": "original Pile entry likely fails with pile.py",
    },
    {
        "component": "nih_exporter",
        "display_name": "NIH ExPorter",
        "the_pile_component": "NIH ExPorter",
        "source_type": "original_pile_unavailable",
        "fidelity": "not_found_yet",
        "hf_path": "EleutherAI/pile",
        "hf_config": "nih_exporter",
        "split": "train",
        "text_candidates": ["text"],
        "expected": "original Pile entry likely fails with pile.py",
    },
    {
        "component": "enron_emails",
        "display_name": "Enron Emails",
        "the_pile_component": "Enron Emails",
        "source_type": "alternative_same_source_family",
        "fidelity": "very_close",
        "hf_path": "corbt/enron-emails",
        "hf_config": None,
        "split": "train",
        "text_candidates": ["body", "text", "message", "subject"],
        "expected": "Enron email corpus alternative",
    },
]


def short_text(text, max_chars=500):
    text = str(text).replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text[:max_chars] + "..." if len(text) > max_chars else text


def extract_text(sample, text_candidates):
    for key in text_candidates:
        if key in sample and sample[key] is not None:
            value = sample[key]

            if isinstance(value, dict):
                return " ".join(str(v) for v in value.values())

            if isinstance(value, list):
                return " ".join(str(v) for v in value)

            return str(value)

    # Fallback useful for StackExchange-like datasets
    if "texts" in sample and sample["texts"] is not None:
        value = sample["texts"]

        if isinstance(value, dict):
            return " ".join(str(v) for v in value.values())

        if isinstance(value, list):
            return " ".join(str(v) for v in value)

        return str(value)

    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 0:
            return value

    return ""


def ascii_ratio(text):
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def load_one_dataset(info, args):
    hf_path = info["hf_path"]
    hf_config = info["hf_config"]
    split = info["split"] if args.split is None else args.split

    kwargs = {
        "path": hf_path,
        "split": split,
        "streaming": args.streaming,
    }

    if hf_config is not None:
        kwargs["name"] = hf_config

    return load_dataset(**kwargs)


def inspect_dataset(info, args):
    print("\n" + "=" * 80)
    print(f"Inspecting: {info['display_name']} ({info['component']})")
    print(f"HF path: {info['hf_path']}")
    print(f"HF config: {info['hf_config']}")
    print("=" * 80)

    result = {
        "component": info["component"],
        "display_name": info["display_name"],
        "the_pile_component": info["the_pile_component"],
        "source_type": info["source_type"],
        "fidelity": info["fidelity"],
        "hf_path": info["hf_path"],
        "hf_config": info["hf_config"] or "",
        "split": info["split"],
        "expected": info["expected"],
        "status": "failed",
        "num_samples_loaded": "",
        "columns": "",
        "avg_chars": "",
        "median_chars": "",
        "min_chars": "",
        "max_chars": "",
        "avg_ascii_ratio": "",
        "example_1": "",
        "example_2": "",
        "example_3": "",
        "error": "",
    }

    try:
        dataset = load_one_dataset(info, args)

        if args.streaming:
            samples = list(dataset.take(args.num_samples))
        else:
            n = min(args.num_samples, len(dataset))
            dataset = dataset.select(range(n))
            samples = list(dataset)

        if not samples:
            raise ValueError("No samples loaded.")

        columns = list(samples[0].keys())
        texts = [extract_text(sample, info["text_candidates"]) for sample in samples]
        lengths = [len(t) for t in texts]
        ascii_scores = [ascii_ratio(t) for t in texts]

        result.update({
            "status": "success",
            "num_samples_loaded": len(samples),
            "columns": ", ".join(columns),
            "avg_chars": round(mean(lengths), 2),
            "median_chars": round(median(lengths), 2),
            "min_chars": min(lengths),
            "max_chars": max(lengths),
            "avg_ascii_ratio": round(mean(ascii_scores), 4),
        })

        random.seed(args.seed)
        example_indices = random.sample(
            range(len(samples)),
            k=min(args.num_examples, len(samples)),
        )

        print("Status: SUCCESS")
        print(f"Samples loaded: {result['num_samples_loaded']}")
        print(f"Columns: {columns}")
        print(f"Average length: {result['avg_chars']} chars")
        print(f"Median length: {result['median_chars']} chars")
        print(f"Min / Max length: {result['min_chars']} / {result['max_chars']} chars")
        print(f"Average ASCII ratio: {result['avg_ascii_ratio']}")

        for i, idx in enumerate(example_indices, start=1):
            example = short_text(texts[idx], args.max_example_chars)
            result[f"example_{i}"] = example

            print("\n" + "-" * 40)
            print(f"Example {i}")
            print("-" * 40)
            print(example)

    except Exception as e:
        result["error"] = str(e)
        print("Status: FAILED")
        print(f"Error: {e}")

        if args.verbose_errors:
            traceback.print_exc()

    return result


def save_results(results, output_csv):
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "component",
        "display_name",
        "the_pile_component",
        "source_type",
        "fidelity",
        "hf_path",
        "hf_config",
        "split",
        "expected",
        "status",
        "num_samples_loaded",
        "columns",
        "avg_chars",
        "median_chars",
        "min_chars",
        "max_chars",
        "avg_ascii_ratio",
        "example_1",
        "example_2",
        "example_3",
        "error",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print(f"CSV saved to: {output_csv}")
    print("=" * 80)


def print_final_summary(results):
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for r in results:
        symbol = "✅" if r["status"] == "success" else "❌"
        config = f"/{r['hf_config']}" if r["hf_config"] else ""
        print(f"{symbol} {r['display_name']} -> {r['hf_path']}{config}")

    print("\n" + "-" * 80)
    print(f"Successes: {sum(r['status'] == 'success' for r in results)}")
    print(f"Failures: {sum(r['status'] == 'failed' for r in results)}")
    print("-" * 80)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Override split for all datasets. If None, uses each dataset default.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples to inspect per dataset.",
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=2,
        help="Number of examples to print per dataset.",
    )

    parser.add_argument(
        "--max_example_chars",
        type=int,
        default=500,
        help="Maximum characters printed per example.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/the_pile/dataset_inventory.csv",
        help="Path to save CSV results.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for displayed examples.",
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode. Recommended.",
    )

    parser.add_argument(
        "--verbose_errors",
        action="store_true",
        help="Print full tracebacks.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("THE PILE-LIKE DATASET INSPECTION")
    print("=" * 80)
    print(f"Streaming: {args.streaming}")
    print(f"Samples per dataset: {args.num_samples}")

    results = []

    for info in DATASET_CANDIDATES:
        result = inspect_dataset(info, args)
        results.append(result)

    save_results(results, args.output_csv)
    print_final_summary(results)


if __name__ == "__main__":
    main()