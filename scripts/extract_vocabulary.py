import sys

sys.path.append("..")

import argparse
import pathlib
from collections import defaultdict
import json

import pandas as pd
import rdkit.Chem as Chem
from molminer.processor import GraphProcessor
from tqdm import tqdm


def compute_property_statistics(csv_path: str, save_path: str) -> None:
    """
    Computes basic statistical properties (mean, standard deviation, minimum, and maximum)
    for each numeric column in a CSV file and saves the results to a JSON file.

    This function is intended to support preprocessing tasks such as scaling and unscaling
    of numeric properties.

    Parameters:
    -----------
        csv_path : str
            Path to the input CSV file containing property data.
            The file must contain a 'smiles' column, which will be excluded
            from the analysis, and one or more numeric columns.
        save_path : str
            Path where the resulting JSON file with the computed statistics
            will be saved.

    Returns:
        None. The computed statistics are saved to the specified JSON file, and a
        confirmation message is printed to the console.
    """
    # load the dataset and drop the smiles column
    df = pd.read_csv(csv_path)
    numeric_df = df.drop(columns=["smiles"])

    # Compute mean, std, min, and max for each column
    stats = {}
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        stats[col] = {
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
        }

    # save as json file (indent for easy-readability)
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"-> Saved statistics to {save_path}.")


def extract_vocabulary(
    dataset_file: str,
    processing_logger: str,
    vocab_attachments_file: str,
    vocab_fragments_file: str,
    vocab_anchors_file: str,
) -> None:
    """
    Extracts fragment, attachment, and anchor vocabularies from a dataset of SMILES strings.

    This function processes a SMILES dataset using a molecular coarsification method to extract
    canonical fragments and their associated attachment and anchor information, which is then
    saved to separate CSV files for downstream use.

    Parameters:
    -----------
    dataset_file : str
        Path to a CSV file containing SMILES strings in the first column (with header).

    processing_logger : str
        Path to a text file where a summary of successful and failed molecules will be logged,
        including error types and counts.

    vocab_attachments_file : str
        Path to the output CSV file that stores all unique fragment-anchor combinations.

    vocab_fragments_file : str
        Path to the output CSV file containing the fragment vocabulary including:
        - 'canon_smiles': Canonical SMILES strings of fragments
        - 'anchors': List of local anchor atom indices
        - 'canon_anchors': List of canonical anchor atom indices
        - 'local_to_canon': Mapping from local to canonical indices

    vocab_anchors_file : str
        Path to the output CSV file storing unique anchor atom symbol sets extracted from
        fragment anchor sites.
    """

    processor = GraphProcessor()

    vocabulary = {
        "canon_smiles": [],
        "anchors": [],
        "canon_anchors": [],
        "local_to_canon": [],
    }

    num_success = 0
    processing_error_types = defaultdict(int)
    failed_files = []

    # this is for padding
    vocabulary["canon_smiles"].append("H")
    vocabulary["anchors"].append([0])
    vocabulary["canon_anchors"].append([0])
    vocabulary["local_to_canon"].append({0: 0})

    with open(dataset_file, "r") as f:
        for ctr, line in tqdm(enumerate(f.readlines())):
            # skip header
            if ctr > 0:
                mol_smiles = line.split(",")[0]  # grab the smiles (first column)

                molecule = Chem.MolFromSmiles(mol_smiles)
                try:
                    info = processor.coarsify_molecule(molecule)

                    for i, smiles in enumerate(info["nodes_vocab_smiles"]):
                        if smiles not in vocabulary["canon_smiles"]:
                            vocabulary["canon_smiles"].append(smiles)
                            vocabulary["anchors"].append(
                                info["possible_local_attachments"][i]
                            )
                            vocabulary["canon_anchors"].append(
                                info["canon_attachments"][i]
                            )
                            vocabulary["local_to_canon"].append(
                                info["local_to_canon"][i]
                            )

                    num_success += 1

                except Exception as e:
                    processing_error_types[e.__class__.__name__] += 1
                    failed_files.append((mol_smiles, e.__class__.__name__))

    vocab_fragments_df = pd.DataFrame(vocabulary)

    canon_anchors = vocab_fragments_df["canon_anchors"].to_numpy().tolist()
    token_smiles = vocab_fragments_df["canon_smiles"].to_numpy().tolist()

    k = 0
    with open(vocab_attachments_file, "w") as g:
        print(",smiles,anchors", file=g)
        print(f'{k},<empty>,"[]"', file=g)
        k += 1

        for i in range(len(token_smiles)):
            for j in range(len(canon_anchors[i])):
                print(f'{k},{token_smiles[i]},"{canon_anchors[i][j]}"', file=g)
                k += 1

    # print out a summary of the processing success/fails
    with open(processing_logger, "w") as f:
        print(
            f"Vocabulary processing of {num_success + len(failed_files)} samples completed.",
            file=f,
        )
        print(
            f"\t - {num_success} ({(100 * num_success / (num_success + len(failed_files))):.2f}%) Success",
            file=f,
        )
        print(
            f"\t - {len(failed_files)} ({(100 * len(failed_files) / (num_success + len(failed_files))):.2f}%) Fails",
            file=f,
        )
        for key in processing_error_types.keys():
            print(
                f"\t\t - {processing_error_types[key]} ({(100 * processing_error_types[key] / (num_success + len(failed_files))):.2f}%) <{key}> Fails",
                file=f,
            )

        print("\nFiles that failed processing (filepath, exception):", file=f)
        for failed_file, reason in failed_files:
            print(f"\t- {failed_file}, {reason}", file=f)

    vocab_fragments_df.to_csv(vocab_fragments_file)

    # types of attachments
    canon_smiles = vocab_fragments_df.canon_smiles.to_list()
    anchors = [item for item in vocab_fragments_df.anchors.to_list()]

    k = 0
    unique_symbols = []
    with open(vocab_anchors_file, "w") as g:
        print(",anchors", file=g)
        print("0,\"['h']\"", file=g)

        for i in range(1, len(canon_smiles)):
            mol = Chem.MolFromSmiles(canon_smiles[i])
            for j in range(len(anchors[i])):
                atom_symbols = [
                    mol.GetAtomWithIdx(idx).GetSymbol().lower() for idx in anchors[i][j]
                ]
                if atom_symbols not in unique_symbols:
                    k += 1
                    unique_symbols.append(atom_symbols)
                    print(f'{k},"{atom_symbols}"', file=g)
    return


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract attachment, fragment and anchor vocabularies "
        "from an SMILES dataset."
    )
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        help="Path to the dataset text file (e.g. ../data/zinc/augmented_zinc250k.txt)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    dataset_file = args.dataset.expanduser().resolve()
    out_dir = dataset_file.parent

    processing_logger = out_dir / f"logger_vocabextraction.txt"
    vocab_attachments_file = out_dir / "vocab_attachments.csv"
    vocab_fragments_file = out_dir / "vocab_fragments.csv"
    vocab_anchors_file = out_dir / "vocab_anchors.csv"
    stats_file = out_dir / "stats.json"

    extract_vocabulary(
        dataset_file,
        processing_logger,
        vocab_attachments_file,
        vocab_fragments_file,
        vocab_anchors_file,
    )

    compute_property_statistics(csv_path=dataset_file, save_path=stats_file)
