import os
import random
import argparse

def split_jsonl(
    input_path: str,
    output_dir: str,
    train_ratio: float = 70,
    valid_ratio: float = 10,
    eval_ratio: float = 20,
    seed: int = None
) -> None:
    """
    Split a JSONL file into train/valid/eval subsets and write each to
    output_dir/{train,valid,eval}/data.jsonl.

    Args:
        input_path:   Path to the original .jsonl file.
        output_dir:   Directory under which train/, valid/, eval/ folders will be created.
        train_ratio:  Percentage of lines to put in the train split.
        valid_ratio:  Percentage of lines to put in the validation split.
        eval_ratio:   Percentage of lines to put in the eval split.
        seed:         Optional random seed for reproducibility.
    """
    # Check ratios
    total = train_ratio + valid_ratio + eval_ratio
    if total <= 0:
        raise ValueError("Sum of ratios must be positive.")
    
    # Read all lines
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # Shuffle
    if seed is not None:
        random.seed(seed)
    random.shuffle(lines)
    
    n = len(lines)
    n_train = int(n * train_ratio / total)
    n_valid = int(n * valid_ratio / total)
    # rest goes to eval
    # (ensures train+valid+eval = n)
    n_eval  = n - n_train - n_valid

    train_lines = lines[:n_train]
    valid_lines = lines[n_train:n_train + n_valid]
    eval_lines  = lines[n_train + n_valid:]
    
    # Prepare output directories
    for split in ('train', 'valid', 'eval'):
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
    
    # Write out each split
    paths_and_data = [
        (os.path.join(output_dir, 'train', 'data.jsonl'), train_lines),
        (os.path.join(output_dir, 'valid', 'data.jsonl'), valid_lines),
        (os.path.join(output_dir, 'eval',  'data.jsonl'), eval_lines),
    ]
    for path, subset in paths_and_data:
        with open(path, 'w') as out_f:
            out_f.writelines(subset)
        print(f"Wrote {len(subset):4d} lines to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a JSONL file into train/valid/eval directories."
    )
    parser.add_argument(
        "--input_path", "-i",
        required=True,
        help="Path to the source .jsonl file"
    )
    parser.add_argument(
        "--output_dir", "-o",
        required=True,
        help="Directory under which train/, valid/, eval/ will be created"
    )
    parser.add_argument(
        "--train_ratio", "-t",
        type=float, default=70.0,
        help="Percentage for the train split (default: 70)"
    )
    parser.add_argument(
        "--valid_ratio", "-v",
        type=float, default=10.0,
        help="Percentage for the valid split (default: 10)"
    )
    parser.add_argument(
        "--eval_ratio", "-e",
        type=float, default=20.0,
        help="Percentage for the eval split (default: 20)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int, default=None,
        help="Random seed for reproducible shuffling (optional)"
    )

    args = parser.parse_args()

    split_jsonl(
        input_path=args.input_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        eval_ratio=args.eval_ratio,
        seed=args.seed
    )

if __name__ == "__main__":
    main()