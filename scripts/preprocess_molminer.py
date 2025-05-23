import argparse
import ast
import gc
import itertools
import multiprocessing
import pathlib
import sys

import pandas as pd
import torch
from rdkit import RDLogger
from tqdm import tqdm

sys.path.append("..")

from functools import partial

from molminer.processor import Architect, GraphProcessor
from molminer.scalers import PropertyScaler
from molminer.utils import (
    init_worker,
    pickle_in_slices,
    read_smiles_generator,
    set_seed,
    worker_process,
)

# silence RDKit warnings
RDLogger.logger().setLevel(RDLogger.CRITICAL)


def _load_vocab(path: pathlib.Path, literal_cols: list[str]) -> pd.DataFrame:
    """Helper funciton to read a CSV and literal-eval selected columns."""
    df = pd.read_csv(path, index_col=0)
    for col in literal_cols:
        df[col] = df[col].map(ast.literal_eval)
    return df


def _parse_args() -> argparse.Namespace:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description="Convert CSV splits into pickle files ready for Molminer training."
    )
    p.add_argument(
        "--data_dir",
        type=pathlib.Path,
        help="Folder that contains train.csv, val.csv, test.csv, "
        "vocab_fragments.csv and stats.json",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )

    p.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="σ parameter passed to Architect for geometry (default: 2.0)",
    )

    p.add_argument(
        "--total_epochs",
        type=int,
        default=30,
        help="Number of epoch resamples to create (default: 30).",
    )
    p.add_argument(
        "--max_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help=(
            "Maximum parallel worker processes "
            f"(default: {multiprocessing.cpu_count()})."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.seed:
        set_seed(args.seed)
        print(f"- Using seed: {args.seed}")

    # resolve file locations
    data_dir: pathlib.Path = args.data_dir.expanduser().resolve()

    paths = {
        "train_csv": data_dir / "train.csv",
        "valid_csv": data_dir / "valid.csv",
        "test_csv": data_dir / "test.csv",
        "vocab_fragments": data_dir / "vocab_fragments.csv",
        "vocab_attachments": data_dir / "vocab_attachments.csv",
        "vocab_anchors": data_dir / "vocab_anchors.csv",
        "stats_json": data_dir / "stats.json",
    }

    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected file(s):\n  " + "\n  ".join(str(p) for p in missing)
        )

    # load shared resources
    vocab_frag_df = _load_vocab(
        paths["vocab_fragments"], ["anchors", "canon_anchors", "local_to_canon"]
    )
    vocab_attach_df = _load_vocab(paths["vocab_attachments"], ["anchors"])
    vocab_anchor_df = _load_vocab(paths["vocab_anchors"], ["anchors"])

    scaler = PropertyScaler(stats_path=paths["stats_json"])

    processor_tpl = GraphProcessor()
    architect_tpl = Architect(
        vocab_fragments_df=vocab_frag_df,
        vocab_attachments_df=vocab_attach_df,
        vocab_anchors_df=vocab_anchor_df,
        properties=torch.zeros(12),  # init full of zeros since its a template
        sigma=args.sigma,
    )

    # consider increasing this number for larger dataset to avoid saving all
    # into a single massive pickle file
    num_slices = 8

    # create ‘steps/’ output folder inside data_dir
    steps_root = data_dir / "steps"
    steps_root.mkdir(exist_ok=True)

    # create the validation, testing processed data (note this is not resampled)
    for split_prefix in ["valid", "test"]:
        worker_fn = partial(
            worker_process,
            processor_template=processor_tpl,
            architect_template=architect_tpl,
        )

        with multiprocessing.Pool(
            processes=min(multiprocessing.cpu_count(), args.max_workers),
            initializer=init_worker,
            initargs=(vocab_frag_df, vocab_attach_df, vocab_anchor_df),
        ) as pool:
            results = pool.imap_unordered(
                worker_fn,
                read_smiles_generator(
                    paths[f"{split_prefix}_csv"],
                    chunk_size=args.max_workers,
                    scaler=scaler,
                ),
            )
            combined_stories = list(itertools.chain.from_iterable(results))

        # path: <data_dir>/steps/<epoch>/<split_prefix>_slice_*.pkl
        epoch_dir = steps_root / split_prefix
        epoch_dir.mkdir(parents=True, exist_ok=True)

        prefix = epoch_dir / f"{split_prefix}_slice"
        pickle_in_slices(
            combined_stories,
            N=1,  # test and validation are smaller, so this is ok.
            file_prefix=str(prefix),
        )

        del combined_stories
        gc.collect()

    # epoch loop
    for epoch in tqdm(range(args.total_epochs), desc="Epoch"):
        worker_fn = partial(
            worker_process,
            processor_template=processor_tpl,
            architect_template=architect_tpl,
        )

        with multiprocessing.Pool(
            processes=min(multiprocessing.cpu_count(), args.max_workers),
            initializer=init_worker,
            initargs=(vocab_frag_df, vocab_attach_df, vocab_anchor_df),
        ) as pool:
            results = pool.imap_unordered(
                worker_fn,
                read_smiles_generator(
                    paths["train_csv"],
                    chunk_size=args.max_workers,
                    scaler=scaler,
                ),
            )
            combined_stories = list(itertools.chain.from_iterable(results))

        # path: <data_dir>/steps/<epoch>/train_slice_*.pkl
        epoch_dir = steps_root / f"{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        prefix = epoch_dir / "train_slice"
        pickle_in_slices(
            combined_stories,
            N=num_slices,
            file_prefix=str(prefix),
        )

        del combined_stories
        gc.collect()

    print("\n All epochs processed. Pickles written under:", steps_root)
