"""Run the whole thing: load, split, train, measure, draw, export.

This file is complete. It calls the functions you wrote. If it crashes, the
problem is almost always in one of your four functions, not here.

Usage:
    python src/run.py                          the defaults
    python src/run.py --size 16                a smaller picture
    python src/run.py --gray                   throw the colour away
    python src/run.py --hidden 256 64          two hidden layers
    python src/run.py --epochs 60 --lr 0.003   train longer, more carefully

Every run prints one line at the end that you can paste straight into the
experiment table in your report.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from data import load_folder, split_train_test
from evaluate import (accuracy, confusion_matrix, plot_confusion, plot_history,
                      plot_worst, worst_examples)
from export_web import export
from train import build_model, save, train

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "clean"))
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--gray", action="store_true", help="grayscale instead of colour")
    parser.add_argument("--hidden", type=int, nargs="*", default=[128])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="run", help="name for the output files")
    args = parser.parse_args()

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"reading {args.data}")
    X, y, class_names, paths = load_folder(
        args.data, image_size=args.size, color=not args.gray
    )
    print(f"{len(X)} images, {len(class_names)} classes: {', '.join(class_names)}")
    for index, name in enumerate(class_names):
        print(f"  {name:24s} {int((y == index).sum())}")

    Xtr, ytr, ptr, Xte, yte, pte = split_train_test(
        X, y, paths, test_ratio=args.test_ratio, seed=args.seed
    )
    print(f"training on {len(Xtr)}, testing on {len(Xte)}")

    model = build_model(X.shape[1], len(class_names), hidden_sizes=tuple(args.hidden))
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"model has {n_parameters:,} parameters")

    history = train(
        model, Xtr, ytr, Xte, yte,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )

    train_accuracy = accuracy(model, Xtr, ytr)
    test_accuracy = accuracy(model, Xte, yte)
    print(f"\ntrain accuracy {train_accuracy:.1%}")
    print(f"test  accuracy {test_accuracy:.1%}")

    plot_history(history, out_dir / f"{args.tag}_curves.png")
    plot_confusion(
        confusion_matrix(model, Xte, yte, len(class_names)),
        class_names, out_dir / f"{args.tag}_confusion.png",
    )
    plot_worst(
        worst_examples(model, Xte, yte, pte, k=10),
        class_names, out_dir / f"{args.tag}_worst.png",
    )

    save(model, out_dir / f"{args.tag}_model.pt")
    export(model, class_names, args.size, not args.gray, Xte, out_dir=ROOT / "docs")

    settings = {
        "tag": args.tag,
        "size": args.size,
        "color": not args.gray,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "n_images": int(len(X)),
        "n_parameters": int(n_parameters),
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
    }
    (out_dir / f"{args.tag}_settings.json").write_text(json.dumps(settings, indent=2))

    print("\nOne line for your experiment table:")
    print(
        f"| {args.size}x{args.size} | "
        f"{'colour' if not args.gray else 'gray'} | "
        f"{args.hidden} | {n_parameters:,} | "
        f"{train_accuracy:.1%} | {test_accuracy:.1%} |"
    )


if __name__ == "__main__":
    main()
