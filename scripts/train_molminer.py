import ast
import gc
import sys

import pandas as pd
import torch
import torch.nn as nn
from rdkit import RDLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

# Suppress RDKit warnings
rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.CRITICAL)

sys.path.append("..")

import argparse
import logging
import pathlib

from molminer.models import MoleculeTransformer
from molminer.utils import (
    collate_fn,
    combine_pickles_incrementally,
    linear_lr_scheduler,
    set_seed,
)


def _parse_args() -> argparse.Namespace:
    """Command-line interface."""
    p = argparse.ArgumentParser(description="Training parameters for Molminer")

    p.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Folder that contains train.csv, val.csv, test.csv, "
        "vocab_fragments.csv and stats.json",
    )

    p.add_argument(
        "--ckpt_dir",
        type=pathlib.Path,
        required=True,
        help="Root folder for checkpoints",
    )

    p.add_argument(
        "--exp_name",
        type=str,
        default="molminer",
        help="String prefix used in naming the checkpoints and loggers. (Default: molminer)",
    )
    p.add_argument(
        "--device", type=str, default="cpu", help="Torch device to use in training"
    )

    # Model parameters
    p.add_argument(
        "--d_model",
        type=int,
        default=2048,
        help="Embedding size for the model. (fragments)",
    )
    p.add_argument(
        "--d_anchor", type=int, default=8, help="Embedding size for anchors."
    )
    p.add_argument("--n_heads", type=int, default=64, help="Number of attention heads.")
    p.add_argument(
        "--num_layers", type=int, default=8, help="Number of layers in the model."
    )
    p.add_argument(
        "--context_hidden_dim",
        type=int,
        default=8192,
        help="Hidden layer dimension for the Contextualizer network.",
    )
    p.add_argument(
        "--context_n_layers",
        type=int,
        default=2,
        help="NUmber of hidden layers in the Contextualizer network",
    )

    p.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="STD used in turning distances into attn scores",
    )
    p.add_argument(
        "--geom_init_value",
        type=float,
        default=1.0,
        help="Scaling factor weighting the geometry bias in the attention mechanism",
    )
    p.add_argument(
        "--geom_requires_grad",
        action="store_true",
        help="Geometry scaling factor is trainable",
    )
    p.add_argument("--d_ff", type=int, default=8192, help="Feed-forward layer size.")
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")

    # Training parameters
    p.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training."
    )
    p.add_argument(
        "--total_epochs", type=int, default=30, help="Total number of training epochs."
    )

    p.add_argument(
        "--peak_lr",
        type=float,
        default=5e-5,
        help="Value of the peak learning rate at warmup.",
    )
    p.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="Minimum value of the learning rate at the end of warmup.",
    )
    p.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.15,
        help="Ratio of the total number of epochs trained that are used in warmup.",
    )

    # Optional cutoff
    p.add_argument(
        "--clip_value", type=float, default=1.0, help="Gradient clipping value"
    )

    p.add_argument(
        "--fixedrollout",
        action="store_true",
        help="Set this flag to use a fixed rollout (no resampling)",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return p.parse_args()


def main():
    args = _parse_args()

    data_dir = args.data_dir.expanduser().resolve()

    # vocab + stats
    vocab_fragments = data_dir / "vocab_fragments.csv"
    vocab_attachments = data_dir / "vocab_attachments.csv"
    vocab_anchors = data_dir / "vocab_anchors.csv"

    # training-story slices
    steps_root = data_dir / "steps"
    val_rollouts = steps_root / "valid"
    test_rollouts = steps_root / "test"

    # checkpoint directory
    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"best_{args.exp_name}.pth"
    last_ckpt = ckpt_dir / f"last_{args.exp_name}.pth"

    # Setup logging
    training_logger = data_dir / f"{args.exp_name}_trainlog.txt"
    logging.basicConfig(
        filename=training_logger,
        level=logging.INFO,
        filemode="w",
    )

    num_properties = 12

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    print(f"- Using device: {device}")

    # set seed of all modules
    if args.seed is not None:
        set_seed(seed=args.seed)
        print(f"- Using seed: {args.seed}")

    # load the fragments' vocabulary
    vocab_fragments_df = pd.read_csv(vocab_fragments, index_col=0)
    for col in ["anchors", "canon_anchors", "local_to_canon"]:
        vocab_fragments_df[col] = vocab_fragments_df[col].map(ast.literal_eval)

    vocab_attachments_df = pd.read_csv(vocab_attachments, index_col=0)
    for col in ["anchors"]:
        vocab_attachments_df[col] = vocab_attachments_df[col].map(ast.literal_eval)

    vocab_anchors_df = pd.read_csv(vocab_anchors, index_col=0)
    for col in ["anchors"]:
        vocab_anchors_df[col] = vocab_anchors_df[col].map(ast.literal_eval)

    fragment_vocab_size = len(vocab_fragments_df)
    attachment_vocab_size = len(vocab_attachments_df)
    anchor_vocab_size = len(vocab_anchors_df)

    #####################################################

    # create the model instance
    model = MoleculeTransformer(
        fragment_vocab_size=fragment_vocab_size,
        attachment_vocab_size=attachment_vocab_size,
        anchor_vocab_size=anchor_vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        d_anchor=args.d_anchor,
        num_properties=num_properties,
        context_hidden_dim=args.context_hidden_dim,
        context_n_layers=args.context_n_layers,
        geom_requires_grad=args.geom_requires_grad,
        geom_init_value=args.geom_init_value,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"- Total model parameters: {total_params:,}")

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr)
    loss_fn = nn.CrossEntropyLoss()

    # load into memory the rollouts used for validation
    # Note: these do not need to be recomputed every epoch, since its the baseline
    print("\t - Loading validation rollouts")
    dataloader_val = DataLoader(
        combine_pickles_incrementally(directory_path=val_rollouts),
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    print("\t - Loading training rollouts (step-0)")
    dataset = combine_pickles_incrementally(directory_path=steps_root / "0")
    num_batches_per_epoch = int(round(len(dataset) / args.batch_size))
    total_num_batches = int(num_batches_per_epoch * args.total_epochs)

    # Preprocess the SMILES list
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    del dataset
    gc.collect()

    # initialize the best validation loss
    best_val_loss = float("inf")

    seen_batches = 0
    for epoch in range(args.total_epochs):
        if epoch > 0 and not args.fixedrollout:
            # clean up the previous dataloader from memory
            del dataloader
            gc.collect()

            print(f"\t - Loading training stories (step-{epoch})")
            # Preprocess the SMILES list
            dataloader = DataLoader(
                dataset=combine_pickles_incrementally(
                    directory_path=steps_root / str(epoch)
                ),
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=True,
                pin_memory=True,
                num_workers=0,
            )

        linear_lr_scheduler(
            optimizer,
            seen_batches=seen_batches,
            total_num_batches=total_num_batches,
            initial_lr=args.peak_lr,
            final_lr=args.min_lr,
            warmup_ratio=args.warmup_ratio,
        )

        # Initialize running stats
        running_sum = 0.0
        running_sum_squared = 0.0
        count = 0

        model.train()
        for batch in tqdm(dataloader):
            seen_batches += 1

            optimizer.zero_grad()

            # Forward pass
            logits = model(
                atom_ids=batch["nodes_vocab_index"].to(device),
                attn_mask=batch["pairwise_distances"].to(device),
                focal_attachment_order=batch["focal_attachment_order"].to(device),
                nodes_saturation=batch["nodes_saturation"].to(device),
                focal_attachment_type=batch["focal_attachment_type"].to(device),
                properties=batch["properties"].to(device),
                attn_mask_readout=batch["distances_to_hit"].to(device),
            )

            # Target: The next frament to predict
            target = batch["next_token"][:, -1].to(
                device
            )  # Target shape: (batch_size,)

            # Compute the loss
            loss = loss_fn(logits, target)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_value
            )  # Gradient clipping
            optimizer.step()

            # Update stats
            running_sum += loss.item() * args.batch_size
            running_sum_squared += (loss.item() ** 2) * args.batch_size
            count += args.batch_size

            linear_lr_scheduler(
                optimizer,
                seen_batches=seen_batches,
                total_num_batches=total_num_batches,
                initial_lr=args.peak_lr,
                final_lr=args.min_lr,
                warmup_ratio=args.warmup_ratio,
            )

        # Compute mean and std after the epoch
        mean_train_loss = running_sum / count
        std_train_loss = ((running_sum_squared / count) - (mean_train_loss**2)) ** 0.5

        # Validation
        # Initialize running stats
        running_sum = 0.0
        running_sum_squared = 0.0
        count = 0

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader_val):
                # Forward pass
                logits = model(
                    batch["nodes_vocab_index"].to(device),
                    attn_mask=batch["pairwise_distances"].to(device),
                    focal_attachment_order=batch["focal_attachment_order"].to(device),
                    nodes_saturation=batch["nodes_saturation"].to(device),
                    focal_attachment_type=batch["focal_attachment_type"].to(device),
                    properties=batch["properties"].to(device),
                    attn_mask_readout=batch["distances_to_hit"].to(device),
                )

                # Target: The next fragment to predict
                target = batch["next_token"][:, -1].to(
                    device
                )  # Target shape: (batch_size,)

                # Compute the loss
                loss = loss_fn(
                    logits, target
                )  # CrossEntropyLoss between predicted logits and target

                # Update stats
                running_sum += loss.item() * args.batch_size
                running_sum_squared += (loss.item() ** 2) * args.batch_size
                count += args.batch_size

            # Compute mean and std after the epoch
            mean_val_loss = running_sum / count
            std_val_loss = ((running_sum_squared / count) - (mean_val_loss**2)) ** 0.5

        # save the last checkpoint of the model
        torch.save(
            {
                "epoch": epoch + 1,
                "seen_batches": seen_batches,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_train_loss": mean_train_loss,
                "std_train_loss": std_train_loss,
                "mean_val_loss": mean_val_loss,
                "std_val_loss": std_val_loss,
                "best_val_loss": best_val_loss,
                "peak_lr": args.peak_lr,
                "min_lr": args.min_lr,
                "sigma": args.sigma,
                "model_hyperparams": {
                    "fragment_vocab_size": fragment_vocab_size,
                    "attachment_vocab_size": attachment_vocab_size,
                    "anchor_vocab_size": anchor_vocab_size,
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "num_layers": args.num_layers,
                    "d_ff": args.d_ff,
                    "num_properties": num_properties,
                    "dropout": args.dropout,
                    "d_anchor": args.d_anchor,
                    "context_hidden_dim": args.context_hidden_dim,
                    "context_n_layers": args.context_n_layers,
                    "geom_init_value": args.geom_init_value,
                    "geom_requires_grad": args.geom_requires_grad,
                },
            },
            last_ckpt,
        )

        # save the last checkpoint as the best model if it outperforms the best validation loss
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "seen_batches": seen_batches,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mean_train_loss": mean_train_loss,
                    "std_train_loss": std_train_loss,
                    "mean_val_loss": mean_val_loss,
                    "std_val_loss": std_val_loss,
                    "best_val_loss": best_val_loss,
                    "peak_lr": args.peak_lr,
                    "min_lr": args.min_lr,
                    "sigma": args.sigma,
                    "model_hyperparams": {
                        "fragment_vocab_size": fragment_vocab_size,
                        "attachment_vocab_size": attachment_vocab_size,
                        "anchor_vocab_size": anchor_vocab_size,
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "num_layers": args.num_layers,
                        "d_ff": args.d_ff,
                        "num_properties": num_properties,
                        "dropout": args.dropout,
                        "d_anchor": args.d_anchor,
                        "context_hidden_dim": args.context_hidden_dim,
                        "context_n_layers": args.context_n_layers,
                        "geom_init_value": args.geom_init_value,
                        "geom_requires_grad": args.geom_requires_grad,
                    },
                },
                best_ckpt,
            )

        # log training and validation metrics every epoch
        num_reprocessing = epoch if not args.fixedrollout else 0
        logging.info(
            f"Epoch [{epoch + 1}/{args.total_epochs}], Batches seen {seen_batches}, Mean train loss: {mean_train_loss:.4f}, Std train loss: {std_train_loss:.4f}, Mean val loss: {mean_val_loss:.4f}, Std val loss: {std_val_loss:.4f}, Repet. id: {num_reprocessing}, Learning rate: {optimizer.param_groups[0]['lr']:.6f}, Best val loss: {best_val_loss:.4f}"
        )


if __name__ == "__main__":
    main()
