"""Fine-tune a Sentence-Transformer embedding model on CodeSearchNet.

Starting from `all-MiniLM-L6-v2`, this script fine-tunes on
code-query relevance pairs from the CodeSearchNet dataset so that
the resulting model produces higher-quality embeddings for code
search tasks.

Usage:
    python scripts/finetune_embeddings.py                    # defaults
    python scripts/finetune_embeddings.py --epochs 5
    python scripts/finetune_embeddings.py --languages python javascript

The fine-tuned model is saved to  models/finetuned-embedding/  and
will be picked up automatically by core/embedder.py when present.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Lazy imports so --help works without heavy dependencies
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune embedding model on CodeSearchNet",
    )
    parser.add_argument("--base-model", default=config.EMBEDDING_MODEL,
                        help="Base model name (default: all-MiniLM-L6-v2)")
    parser.add_argument("--output-dir", default=config.FINETUNED_EMBEDDING_MODEL,
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Max training samples to use")
    parser.add_argument("--languages", nargs="+",
                        default=["python", "javascript"],
                        help="CodeSearchNet languages to train on")
    args = parser.parse_args()

    from datasets import load_dataset
    from sentence_transformers import (
        SentenceTransformer,
        InputExample,
        losses,
    )
    from torch.utils.data import DataLoader

    print(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)

    # ── Load CodeSearchNet ─────────────────────────────────────────────
    print(f"Loading CodeSearchNet for languages: {args.languages}")
    train_examples = []

    for lang in args.languages:
        try:
            ds = load_dataset(
                "code_search_net", lang, split="train", trust_remote_code=True,
            )
        except Exception as e:
            print(f"  Warning: could not load {lang}: {e}")
            continue

        for row in ds:
            if len(train_examples) >= args.max_samples:
                break
            docstring = row.get("func_documentation_string", "").strip()
            code = row.get("func_code_string", "").strip()
            if docstring and code:
                # Positive pair: docstring ↔ code
                train_examples.append(
                    InputExample(texts=[docstring[:512], code[:512]], label=1.0)
                )

    if not train_examples:
        print("No training data found. Exiting.")
        sys.exit(1)

    print(f"Collected {len(train_examples)} training pairs")

    # ── Train ──────────────────────────────────────────────────────────
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=args.batch_size,
    )
    train_loss = losses.CosineSimilarityLoss(model)

    print(f"Training for {args.epochs} epochs (batch_size={args.batch_size})...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(train_dataloader)),
        show_progress_bar=True,
        output_path=args.output_dir,
    )

    print(f"\nFine-tuned model saved to: {args.output_dir}")
    print("The application will automatically use it on next startup.")


if __name__ == "__main__":
    main()
