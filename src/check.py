"""Check your own work. Run this before you submit.

    python src/check.py

It does not use any of the functions you wrote. It reads the files on disk and
the model you exported, and works everything out for itself. That is the point:
if it disagrees with the numbers your own code printed, one of the two is
wrong, and finding out which is part of the assignment.

Paste the whole output into your report.
"""

import base64
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

problems = []
warnings = []


def title(text):
    print(f"\n{text}\n" + "-" * len(text))


def complain(text):
    problems.append(text)
    print(f"  PROBLEM  {text}")


def warn(text):
    warnings.append(text)
    print(f"  warning  {text}")


def good(text):
    print(f"  ok       {text}")


def image_files(folder):
    return sorted(
        p for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


# ---------------------------------------------------------------------------


def check_folder(name, folder, least_per_class):
    title(f"{name}  ({folder})")
    folder = Path(folder)
    if not folder.exists():
        complain(f"{folder} does not exist")
        return None

    classes = sorted(d for d in folder.iterdir() if d.is_dir())
    if not classes:
        complain(f"{folder} has no class sub-folders in it")
        return None

    counts = {}
    for class_dir in classes:
        files = image_files(class_dir)
        counts[class_dir.name] = len(files)
        print(f"  {class_dir.name:28s} {len(files):5d} images")

    if len(counts) < 3:
        complain(f"only {len(counts)} classes. You need at least 3.")
    fewest = min(counts.values()) if counts else 0
    if fewest < least_per_class:
        complain(
            f'"{min(counts, key=counts.get)}" has only {fewest} images, '
            f"fewer than the {least_per_class} this assignment asks for"
        )
    most = max(counts.values()) if counts else 0
    if fewest and most > fewest * 3:
        warn(
            f"the biggest class has {most} images and the smallest has {fewest}. "
            "A model can score well just by always guessing the big class."
        )
    return counts


def check_for_repeats(folders):
    title("The same image appearing twice")
    seen = defaultdict(list)
    for folder in folders:
        for path in image_files(folder):
            try:
                seen[hashlib.md5(path.read_bytes()).hexdigest()].append(path)
            except Exception:
                complain(f"could not read {path}")

    repeats = {k: v for k, v in seen.items() if len(v) > 1}
    if not repeats:
        good("every image is different")
        return

    crossing = [
        group for group in repeats.values()
        if len({p.parent.parent.name for p in group}) > 1
        or len({p.parent.name for p in group}) > 1
    ]
    if crossing:
        complain(
            f"{len(crossing)} image(s) appear in more than one place, including "
            "across different classes or across your training and your own "
            "photos. Test accuracy measured on an image the model trained on "
            "means nothing."
        )
        for group in crossing[:3]:
            print("           " + "  ==  ".join(str(p.relative_to(ROOT)) for p in group))
    else:
        warn(f"{len(repeats)} image(s) appear twice inside the same class")


# ---------------------------------------------------------------------------


def load_web_model():
    title("The model you exported for the web page")
    web = ROOT / "docs"
    missing = [
        name for name in ("model.json", "weights.bin", "selftest.json", "labels.json")
        if not (web / name).exists()
    ]
    if missing:
        complain(f"docs/ is missing {', '.join(missing)}. Run src/run.py first.")
        return None

    model = json.loads((web / "model.json").read_text())
    weights = np.frombuffer((web / "weights.bin").read_bytes(), dtype=np.float32)

    if len(weights) != model["n_floats"]:
        complain(
            f"weights.bin holds {len(weights)} numbers but model.json expects "
            f"{model['n_floats']}. Export again."
        )
        return None

    shape = f'{model["input"]["size"]}x{model["input"]["size"]}'
    kind = "colour" if model["input"]["channels"] == 3 else "grayscale"
    good(f'{len(model["labels"])} classes: {", ".join(model["labels"])}')
    good(f"input {shape} {kind}, {model['n_floats']:,} parameters")
    print(f"           layers: " + " -> ".join(
        f'Linear({l["in"]}, {l["out"]})' if l["type"] == "linear" else "ReLU"
        for l in model["layers"]
    ))
    return model, weights


def forward(model, weights, x):
    """The same arithmetic app.js does in the browser, in numpy."""
    values = np.asarray(x, dtype=np.float32)
    for layer in model["layers"]:
        if layer["type"] == "linear":
            start, count = layer["w"]
            weight = weights[start:start + count].reshape(layer["out"], layer["in"])
            bias = weights[layer["b"][0]:layer["b"][0] + layer["out"]]
            values = weight @ values + bias
        elif layer["type"] == "relu":
            values = np.maximum(values, 0.0)
    return values


def prepare(path_or_image, model):
    size = model["input"]["size"]
    mode = "RGB" if model["input"]["channels"] == 3 else "L"
    image = path_or_image
    if not isinstance(image, Image.Image):
        image = Image.open(path_or_image)
    image = image.convert(mode).resize((size, size))
    return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0


def check_selftest(model, weights):
    title("Does the exported model give the answers Python gave?")
    cases = json.loads((ROOT / "docs" / "selftest.json").read_text())["cases"]
    worst = 0.0
    for case in cases:
        image = Image.open(io.BytesIO(base64.b64decode(case["png"])))
        got = forward(model, weights, prepare(image, model))
        worst = max(worst, float(np.abs(got - np.array(case["logits"])).max()))

    if worst < 0.02:
        good(f"yes, they agree (largest gap {worst:.4f})")
    else:
        complain(
            f"no, they differ by {worst:.3f}. The web page will be wrong too. "
            "Export again after training."
        )


def check_my_photos(model, weights, folder):
    title(f"Your own photos  ({folder})")
    folder = Path(folder)
    if not folder.exists():
        complain(
            f"{folder} does not exist. Problem 5 asks for at least 5 photos per "
            "class that you took yourself."
        )
        return

    labels = model["labels"]
    right = 0
    total = 0
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    confident_mistakes = []

    for class_dir in sorted(d for d in folder.iterdir() if d.is_dir()):
        if class_dir.name not in labels:
            complain(
                f'"{class_dir.name}" is not one of your classes. '
                f'Name the folders exactly: {", ".join(labels)}'
            )
            continue
        true_index = labels.index(class_dir.name)
        files = image_files(class_dir)
        if len(files) < 5:
            warn(f'"{class_dir.name}" has only {len(files)} of your own photos')

        for path in files:
            try:
                logits = forward(model, weights, prepare(path, model))
            except Exception as error:
                complain(f"could not read {path}: {error}")
                continue
            shifted = logits - logits.max()
            probabilities = np.exp(shifted) / np.exp(shifted).sum()
            guess = int(np.argmax(logits))

            total += 1
            matrix[true_index][guess] += 1
            if guess == true_index:
                right += 1
            else:
                confident_mistakes.append(
                    (float(probabilities[guess]), path.name,
                     labels[guess], class_dir.name)
                )

    if total == 0:
        complain("no readable photos found")
        return

    print(f"\n  accuracy on your own photos: {right}/{total} = {right / total:.1%}")
    print("\n  rows are what it really is, columns are what the model said")
    width = max(len(name) for name in labels) + 2
    print(" " * (width + 4) + "".join(f"{name[:8]:>9s}" for name in labels))
    for i, name in enumerate(labels):
        print(f"    {name:<{width}s}" + "".join(f"{int(v):9d}" for v in matrix[i]))

    if confident_mistakes:
        confident_mistakes.sort(reverse=True)
        print("\n  the mistakes it was most sure about:")
        for confidence, name, said, really in confident_mistakes[:5]:
            print(f"    {name:28s} said {said} ({confidence:.0%}), really {really}")


# ---------------------------------------------------------------------------


def main():
    print("MAS1004 Assignment 1, checking your work")

    check_folder("Training images", ROOT / "data" / "clean", least_per_class=50)
    check_folder("Your own photos", ROOT / "data" / "my_photos", least_per_class=5)
    check_for_repeats([ROOT / "data" / "clean", ROOT / "data" / "my_photos"])

    loaded = load_web_model()
    if loaded:
        model, weights = loaded
        check_selftest(model, weights)
        check_my_photos(model, weights, ROOT / "data" / "my_photos")

    title("Summary")
    if problems:
        print(f"  {len(problems)} thing(s) to fix:")
        for item in problems:
            print(f"    - {item}")
    else:
        print("  nothing is broken")
    if warnings:
        print(f"  {len(warnings)} thing(s) worth a sentence in your report:")
        for item in warnings:
            print(f"    - {item}")
    print()
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
