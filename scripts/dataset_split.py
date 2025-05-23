import argparse
import pathlib
import random

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Randomly split a CSV dataset into train/valid/test files "
        "(80 / 10 / 10 by default)."
    )
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        help="Path to the dataset (e.g. ../data/zinc/augmented_zinc250k.txt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    src = args.dataset.expanduser().resolve()
    out_dir = src.parent

    train_path = out_dir / "train.csv"
    valid_path = out_dir / "valid.csv"
    test_path = out_dir / "test.csv"

    random.seed(args.seed)

    train_prob, valid_prob, test_prob = 0.8, 0.1, 0.1

    with (
        open(src, "r", encoding="utf-8") as infile,
        open(train_path, "w", encoding="utf-8") as train_f,
        open(valid_path, "w", encoding="utf-8") as valid_f,
        open(test_path, "w", encoding="utf-8") as test_f,
    ):
        header = infile.readline()
        for f in (train_f, valid_f, test_f):
            f.write(header)

        for line in tqdm(infile, desc="Splitting"):
            r = random.random()
            if r < train_prob:
                train_f.write(line)
            elif r < train_prob + valid_prob:
                valid_f.write(line)
            else:
                test_f.write(line)

    print(
        f"Done. Wrote:\n"
        f"\t - {train_path}\n"
        f"\t - {valid_path}\n"
        f"\t - {test_path}\n"
        f"(seed={args.seed}, ratios={train_prob}/{valid_prob}/{test_prob})"
    )


if __name__ == "__main__":
    main()
