import sys

sys.path.append("..")
import argparse
import pathlib

import joblib
import pandas as pd
from molminer.scalers import PropertyScaler
from sklearn.mixture import GaussianMixture


def _parse_args() -> argparse.Namespace:
    """Command-line interface for training a property GMM."""
    p = argparse.ArgumentParser(
        description="Fit a Gaussian Mixture Model to the (scaled) "
        "property vectors in train.csv."
    )
    p.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Folder containing train.csv and stats.json",
    )
    p.add_argument(
        "--model_out",
        type=pathlib.Path,
        required=True,
        help="Where to save the trained model (e.g: <data_dir>/gmm_model.pkl)",
    )
    p.add_argument(
        "--n_components",
        type=int,
        default=8,
        help="Number of Gaussian components (default: 8)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for GMM initialisation (default: 42)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # resolve paths
    data_dir = args.data_dir.expanduser().resolve()
    train_csv = data_dir / "train.csv"
    stats_json = data_dir / "stats.json"
    model_out = args.model_out.expanduser().resolve()

    # sanity-check files
    for pth in (train_csv, stats_json):
        if not pth.exists():
            raise FileNotFoundError(pth)

    # load scaler + dataset
    scaler = PropertyScaler(stats_path=stats_json)

    df = pd.read_csv(train_csv)
    df = df.drop(columns=["smiles"])  # keep only property columns

    # scale properties in-place
    for col in df.columns:
        df[col] = df[col].map(lambda x: scaler.scale(value=x, property_name=col))

    # fit the model
    gmm = GaussianMixture(
        n_components=args.n_components,
        random_state=args.seed,
        covariance_type="full",
    ).fit(df)

    # save
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gmm, model_out)
    print(f"-> GMM trained and saved to {model_out}")
