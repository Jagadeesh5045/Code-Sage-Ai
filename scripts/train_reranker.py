"""Train a cross-encoder re-ranker on code-query relevance pairs.

This script trains a CrossEncoder model that scores how relevant a code
chunk is to a developer's natural-language query.  Training data comes
from CodeSearchNet (positive pairs) with in-batch negatives.

Usage:
    python scripts/train_reranker.py
    python scripts/train_reranker.py --epochs 3 --max-samples 30000

The trained model is saved to  models/finetuned-reranker/  and will
be picked up automatically by core/reranker.py when present.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def main():
    parser = argparse.ArgumentParser(
        description="Train cross-encoder re-ranker on CodeSearchNet",
    )
    parser.add_argument("--base-model", default=config.RERANKER_MODEL,
                        help="Base cross-encoder model")
    parser.add_argument("--output-dir", default=config.FINETUNED_RERANKER_MODEL,
                        help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--max-samples", type=int, default=30000,
                        help="Max training samples")
    parser.add_argument("--languages", nargs="+",
                        default=["python", "javascript"],
                        help="CodeSearchNet languages")
    args = parser.parse_args()

    from datasets import load_dataset
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    print(f"Loading base cross-encoder: {args.base_model}")
    model = CrossEncoder(args.base_model, num_labels=1, max_length=512)

    # ── Load and prepare data ──────────────────────────────────────────
    print(f"Loading CodeSearchNet: {args.languages}")
    positives = []
    all_codes = []

    for lang in args.languages:
        try:
            ds = load_dataset(
                "code_search_net", lang, split="train", trust_remote_code=True,
            )
        except Exception as e:
            print(f"  Warning: could not load {lang}: {e}")
            continue

        for row in ds:
            if len(positives) >= args.max_samples:
                break
            docstring = row.get("func_documentation_string", "").strip()
            code = row.get("func_code_string", "").strip()
            if docstring and code:
                positives.append((docstring[:256], code[:512]))
                all_codes.append(code[:512])

    if not positives:
        print("No training data. Exiting.")
        sys.exit(1)

    print(f"Collected {len(positives)} positive pairs")

    # Build training set with positives (label=1) and hard negatives (label=0)
    train_examples = []
    for query, code in positives:
        # Positive
        train_examples.append(InputExample(texts=[query, code], label=1.0))
        # Random negative (different code for the same query)
        neg_code = random.choice(all_codes)
        train_examples.append(InputExample(texts=[query, neg_code], label=0.0))

    random.shuffle(train_examples)
    print(f"Total training examples (pos + neg): {len(train_examples)}")

    # ── Train ──────────────────────────────────────────────────────────
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=args.batch_size,
    )

    print(f"Training for {args.epochs} epochs (batch_size={args.batch_size})...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(train_dataloader)),
        show_progress_bar=True,
        output_path=args.output_dir,
    )

    print(f"\nTrained re-ranker saved to: {args.output_dir}")
    print("The application will automatically use it on next startup.")


if __name__ == "__main__":
    main()
