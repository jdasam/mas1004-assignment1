"""Tests for Problem 2. Run: pytest tests/test_data.py"""

import numpy as np

from conftest import CLASSES, PER_CLASS
from data import load_folder, split_train_test

TOTAL = len(CLASSES) * PER_CLASS


def test_shapes_and_types(image_folder):
    X, y, class_names, paths = load_folder(image_folder, image_size=16, color=True)

    assert X.dtype == np.float32, f"X should be float32, got {X.dtype}"
    assert y.dtype == np.int64, f"y should be int64, got {y.dtype}"
    assert X.ndim == 2, "X should be two dimensional: one row per image"
    assert X.shape == (TOTAL, 16 * 16 * 3)
    assert y.shape == (TOTAL,)
    assert len(paths) == TOTAL


def test_values_are_between_zero_and_one(image_folder):
    X, _, _, _ = load_folder(image_folder, image_size=16, color=True)
    assert X.min() >= 0.0 and X.max() <= 1.0, (
        "Pixel values must be scaled to 0..1. Divide by 255."
    )
    assert X.max() > 0.5, (
        "Every value is tiny. You probably divided by 255 twice."
    )


def test_class_names_are_sorted(image_folder):
    _, _, class_names, _ = load_folder(image_folder, image_size=16)
    assert class_names == sorted(CLASSES)


def test_labels_match_the_folder_each_image_came_from(image_folder):
    _, y, class_names, paths = load_folder(image_folder, image_size=16)
    for label, path in zip(y, paths):
        assert class_names[label] == path.parent.name


def test_broken_and_non_image_files_are_skipped(image_folder):
    X, _, _, paths = load_folder(image_folder, image_size=16)
    assert len(X) == TOTAL, (
        "broken.jpg and notes.txt must not end up in X. "
        f"Expected {TOTAL} rows, got {len(X)}."
    )
    names = [p.name for p in paths]
    assert "broken.jpg" not in names and "notes.txt" not in names


def test_grayscale_gives_one_channel(image_folder):
    X, _, _, _ = load_folder(image_folder, image_size=16, color=False)
    assert X.shape == (TOTAL, 16 * 16)


def test_same_result_every_time(image_folder):
    first = load_folder(image_folder, image_size=16)
    second = load_folder(image_folder, image_size=16)
    assert np.array_equal(first[0], second[0]), (
        "Two runs gave different arrays. Sort the files so the order is fixed."
    )
    assert np.array_equal(first[1], second[1])


def test_split_keeps_everything(image_folder):
    X, y, _, paths = load_folder(image_folder, image_size=16)
    Xtr, ytr, ptr, Xte, yte, pte = split_train_test(X, y, paths, test_ratio=0.25, seed=0)

    assert len(Xtr) == len(ytr) == len(ptr)
    assert len(Xte) == len(yte) == len(pte)
    assert len(Xtr) + len(Xte) == TOTAL, "The split lost or duplicated rows."


def test_split_has_no_image_on_both_sides(image_folder):
    X, y, _, paths = load_folder(image_folder, image_size=16)
    _, _, ptr, _, _, pte = split_train_test(X, y, paths, test_ratio=0.25, seed=0)

    shared = set(map(str, ptr)) & set(map(str, pte))
    assert not shared, (
        "These images are in both the training set and the test set: "
        f"{sorted(shared)[:3]}. Then the test accuracy is a lie."
    )


def test_every_class_appears_on_both_sides(image_folder):
    X, y, _, paths = load_folder(image_folder, image_size=16)
    _, ytr, _, _, yte, _ = split_train_test(X, y, paths, test_ratio=0.25, seed=0)

    assert set(np.unique(ytr)) == set(range(len(CLASSES)))
    assert set(np.unique(yte)) == set(range(len(CLASSES))), (
        "A class has no test images, so its accuracy cannot be measured."
    )


def test_split_is_repeatable(image_folder):
    X, y, _, paths = load_folder(image_folder, image_size=16)
    first = split_train_test(X, y, paths, test_ratio=0.25, seed=7)
    second = split_train_test(X, y, paths, test_ratio=0.25, seed=7)
    assert [str(p) for p in first[2]] == [str(p) for p in second[2]], (
        "The same seed gave two different splits."
    )
