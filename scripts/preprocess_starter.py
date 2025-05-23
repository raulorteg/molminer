import sys

sys.path.append("..")

import argparse
import pathlib
import pickle

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from molminer.processor import GraphProcessor
from molminer.scalers import PropertyScaler
from tqdm import tqdm


def main(input_file: str, output_file: str, vocabulary: list, scaler: PropertyScaler):
    """
    Processes a CSV file containing SMILES strings and molecular properties.

    This function reads a CSV file where each row contains a SMILES string and associated
    molecular properties. It transforms SMILES into graph-based fragment masks, scales the
    molecular properties using a provided scaler, and saves the resulting (mask, scaled_properties)
    tuples to a pickle file.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file containing SMILES and molecular properties. The first line
        is assumed to be a header.

    output_file : str
        Path where the processed data will be saved as a pickle file.

    vocabulary : list
        A list of canonical SMILES strings representing the fragment vocabulary. This is used
        to create binary masks indicating the presence of each fragment.

    scaler : PropertyScaler
        An instance of the `PropertyScaler` class used to normalize the molecular properties.
        Each property value is scaled according to predefined statistics.

    Returns
    -------
    None
        The function saves the processed data to the specified `output_file` and does not return anything.
    """

    processor = GraphProcessor()
    processed_data = []

    with open(input_file, "r") as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines)):
            # skip the header line (assumed to be the first line in the CSV)
            if i > 0:
                smiles, *properties = line.strip().split(",")
                present_fragments = set(
                    processor.coarsify_molecule(Chem.MolFromSmiles(smiles))[
                        "nodes_vocab_smiles"
                    ]
                )
                mask = np.array(
                    [1 if item in present_fragments else 0 for item in vocabulary]
                )

                scaled_properties = np.array(
                    [
                        scaler.scale(value=float(properties[0]), property_name="logP"),
                        scaler.scale(value=float(properties[1]), property_name="qed"),
                        scaler.scale(value=float(properties[2]), property_name="SAS"),
                        scaler.scale(
                            value=float(properties[3]), property_name="FractionCSP3"
                        ),
                        scaler.scale(value=float(properties[4]), property_name="molWt"),
                        scaler.scale(value=float(properties[5]), property_name="TPSA"),
                        scaler.scale(value=float(properties[6]), property_name="MR"),
                        scaler.scale(value=float(properties[7]), property_name="hbd"),
                        scaler.scale(value=float(properties[8]), property_name="hba"),
                        scaler.scale(
                            value=float(properties[9]), property_name="num_rings"
                        ),
                        scaler.scale(
                            value=float(properties[10]),
                            property_name="num_rotable_bonds",
                        ),
                        scaler.scale(
                            value=float(properties[11]),
                            property_name="num_quiral_centers",
                        ),
                    ]
                )

                processed_data.append((mask, scaled_properties))

    # Save the processed data as a pickle file: list of (mask, scaled_properties)
    with open(output_file, "wb") as f:
        pickle.dump(processed_data, f)


def _parse_args() -> argparse.Namespace:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description="Convert CSV splits into pickle files ready for FragmentStarter training."
    )
    p.add_argument(
        "--data_dir",
        type=pathlib.Path,
        help="Folder that contains train.csv, val.csv, test.csv, "
        "vocab_fragments.csv and stats.json",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()

    # expected files at data_dir
    split_csvs = {
        "train": data_dir / "train.csv",
        "valid": data_dir / "valid.csv",
        "test": data_dir / "test.csv",
    }
    vocab_path = data_dir / "vocab_fragments.csv"
    stats_path = data_dir / "stats.json"

    missing = [
        p
        for p in list(split_csvs.values()) + [vocab_path, stats_path]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing expected file(s): {', '.join(str(m) for m in missing)}"
        )

    # load shared resources
    vocabulary = pd.read_csv(vocab_path, index_col=0)["canon_smiles"].tolist()
    scaler = PropertyScaler(stats_path=stats_path)

    # process each split
    for split_name, csv_path in split_csvs.items():
        pkl_path = data_dir / f"{split_name}_starter.pkl"
        print(f"[{split_name}] -> {pkl_path}")
        main(
            input_file=csv_path,
            output_file=pkl_path,
            vocabulary=vocabulary,
            scaler=scaler,
        )

    print("Done -> all starter pickles written to", data_dir)
