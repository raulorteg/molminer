import sys

sys.path.append("..")

import argparse
import gc
import logging
import pathlib
import pickle

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm as tqdm
from molminer.models import FragmentStarter
from molminer.utils import collate_fn_starter, set_seed
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _parse_args() -> argparse.Namespace:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description="Training parameters for the Initial Frament Selector of Molminer"
    )

    p.add_argument(
        "--data_dir",
        type=pathlib.Path,
        help="Folder that contains train.csv, val.csv, test.csv, "
        "vocab_fragments.csv and stats.json",
    )

    p.add_argument(
        "--ckpt_dir",
        type=pathlib.Path,
        required=True,
        help="Root folder for checkpoints",
    )

    # model and tranining hyperparameters
    p.add_argument("--d_ff", type=int, default=512, help="Hidden layer dimension")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    p.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate")
    p.add_argument(
        "--lr_factor", type=float, default=0.5, help="Learning rate scheduler factor"
    )
    p.add_argument("--num_layers", type=int, default=3, help="Number of hidden layers")
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--exp_name",
        type=str,
        default="starter",
        help="String prefix used in naming the checkpoints and loggers. (Default: starter)",
    )
    p.add_argument(
        "--total_epochs", type=int, default=100, help="Total number of training epochs."
    )

    return p.parse_args()


def main():
    args = _parse_args()

    # expected input files
    data_dir = args.data_dir.expanduser().resolve()
    split_csvs = {
        "train": data_dir / "train_starter.pkl",
        "valid": data_dir / "valid_starter.pkl",
        "test": data_dir / "test_starter.pkl",
    }
    vocab_fragments = data_dir / "vocab_fragments.csv"
    stats_path = data_dir / "stats.json"

    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = ckpt_dir / f"best_{args.exp_name}.pth"
    last_ckpt = ckpt_dir / f"last_{args.exp_name}.pth"

    training_logger = data_dir / f"{args.exp_name}_trainlog.txt"

    # sanity-check that everything exists
    missing = [
        p
        for p in list(split_csvs.values()) + [vocab_fragments, stats_path]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing expected file(s): {', '.join(str(m) for m in missing)}"
        )

    # Setup logging
    logging.basicConfig(
        filename=training_logger,
        level=logging.INFO,
        filemode="w",
    )

    # load the fragments' vocabulary
    smiles_list = pd.read_csv(vocab_fragments, index_col=0)["canon_smiles"].tolist()
    len_vocab = len(smiles_list)

    d_model_in = 12
    num_properties = d_model_in
    d_model_out = len_vocab

    # set seed of all modules
    if args.seed is not None:
        set_seed(seed=args.seed)
        print(f"- Using seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"- Using device: {device}")

    model = FragmentStarter(
        d_model_in=d_model_in,
        d_model_out=d_model_out,
        d_ff=args.d_ff,
        dropout=args.dropout,
        num_layers=args.num_layers,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"- Total model parameters: {total_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=args.lr_factor
    )

    # load the training and validation data
    with open(split_csvs["valid"], "rb") as f:
        val_dataset = pickle.load(f)

    dataloader_val = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn_starter,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    del val_dataset
    gc.collect()

    with open(split_csvs["train"], "rb") as f:
        train_dataset = pickle.load(f)

    dataloader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn_starter,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )

    del train_dataset
    gc.collect()

    best_val_loss = float("inf")
    for epoch in tqdm(range(args.total_epochs)):
        # Initialize running stats
        running_sum = 0.0
        running_sum_squared = 0.0
        count = 0

        model.train()
        for labels, properties in dataloader_train:
            optimizer.zero_grad()

            logits = model(x=properties.to(device))
            loss = criterion(logits, labels.float().to(device))

            loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update stats
            running_sum += loss.item() * properties.size(0)
            running_sum_squared += (loss.item() ** 2) * properties.size(0)
            count += properties.size(0)

        # Compute mean and std after the epoch
        mean_train_loss = running_sum / count
        std_train_loss = ((running_sum_squared / count) - (mean_train_loss**2)) ** 0.5

        # Initialize running stats
        running_sum = 0.0
        running_sum_squared = 0.0
        count = 0

        model.eval()
        with torch.no_grad():
            for labels, properties in dataloader_val:
                logits = model(x=properties.to(device))
                loss = criterion(logits, labels.float().to(device))

                # Update stats
                running_sum += loss.item() * properties.size(0)
                running_sum_squared += (loss.item() ** 2) * properties.size(0)
                count += properties.size(0)

        # Compute mean and std after the epoch
        mean_val_loss = running_sum / count
        std_val_loss = ((running_sum_squared / count) - (mean_val_loss**2)) ** 0.5

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mean_train_loss": mean_train_loss,
                    "std_train_loss": std_train_loss,
                    "mean_val_loss": mean_val_loss,
                    "std_val_loss": std_val_loss,
                    "best_val_loss": best_val_loss,
                    "model_hyperparams": {
                        "fragment_vocab_size": len_vocab,
                        "num_layers": args.num_layers,
                        "d_ff": args.d_ff,
                        "num_properties": num_properties,
                        "dropout": args.dropout,
                    },
                },
                best_ckpt,
            )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_train_loss": mean_train_loss,
                "std_train_loss": std_train_loss,
                "mean_val_loss": mean_val_loss,
                "std_val_loss": std_val_loss,
                "best_val_loss": best_val_loss,
                "model_hyperparams": {
                    "fragment_vocab_size": len_vocab,
                    "num_layers": args.num_layers,
                    "d_ff": args.d_ff,
                    "num_properties": num_properties,
                    "dropout": args.dropout,
                },
            },
            last_ckpt,
        )

        logging.info(
            f"Epoch [{epoch + 1}/{args.total_epochs}], Mean train loss: {mean_train_loss:8f}, Std train loss: {std_train_loss:.8f}, Mean val loss: {mean_val_loss:.8f}, Std val loss: {std_val_loss:.8f}, Learning rate: {optimizer.param_groups[0]['lr']:.8f}, Best val loss: {best_val_loss:.8f}"
        )

        scheduler.step(mean_val_loss)


if __name__ == "__main__":
    main()
