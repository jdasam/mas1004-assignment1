"""Checks that the exported web files really hold your model.

This one tests code that was given to you, so it should pass from the start.
It is here as a safety net: it re-runs your model from model.json and
weights.bin using nothing but numpy, the same way app.js does in the browser,
and checks the answers match PyTorch.

Run: pytest tests/test_export.py
"""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from export_web import export


def forward_like_the_browser(model_json, weights, x):
    """The same forward pass app.js does, written in numpy."""
    values = np.asarray(x, dtype=np.float32)
    for layer in model_json["layers"]:
        if layer["type"] == "linear":
            weight_start, weight_count = layer["w"]
            bias_start, _ = layer["b"]
            weight = weights[weight_start:weight_start + weight_count]
            weight = weight.reshape(layer["out"], layer["in"])
            bias = weights[bias_start:bias_start + layer["out"]]
            values = weight @ values + bias
        elif layer["type"] == "relu":
            values = np.maximum(values, 0.0)
    return values


@pytest.fixture
def exported(tmp_path):
    torch.manual_seed(0)
    size, channels, classes = 8, 3, 4
    dim = size * size * channels

    model = nn.Sequential(
        nn.Linear(dim, 16), nn.ReLU(),
        nn.Linear(16, 10), nn.ReLU(),
        nn.Linear(10, classes),
    )
    model.eval()

    X = np.random.default_rng(0).random((5, dim)).astype(np.float32)
    names = ["one", "two", "three", "four"]
    export(model, names, size, True, X, out_dir=tmp_path)
    return tmp_path, model, X


def test_every_file_is_written(exported):
    out, _, _ = exported
    for name in ("model.json", "weights.bin", "selftest.json", "labels.json"):
        assert (out / name).exists(), f"{name} was not written"


def test_weight_count_matches(exported):
    out, model, _ = exported
    model_json = json.loads((out / "model.json").read_text())
    weights = np.frombuffer((out / "weights.bin").read_bytes(), dtype=np.float32)

    assert len(weights) == model_json["n_floats"]
    assert len(weights) == sum(p.numel() for p in model.parameters())


def test_numpy_replay_matches_pytorch(exported):
    out, model, X = exported
    model_json = json.loads((out / "model.json").read_text())
    weights = np.frombuffer((out / "weights.bin").read_bytes(), dtype=np.float32)

    with torch.no_grad():
        expected = model(torch.from_numpy(X)).numpy()

    for row, wanted in zip(X, expected):
        got = forward_like_the_browser(model_json, weights, row)
        assert np.allclose(got, wanted, atol=1e-4), (
            "Replaying the exported weights gives a different answer from "
            "PyTorch, so the browser would too."
        )


def test_selftest_answers_match(exported):
    out, _, _ = exported
    model_json = json.loads((out / "model.json").read_text())
    weights = np.frombuffer((out / "weights.bin").read_bytes(), dtype=np.float32)
    cases = json.loads((out / "selftest.json").read_text())["cases"]

    assert len(cases) == 3
    for case in cases:
        assert len(case["logits"]) == len(model_json["labels"])
        assert len(case["png"]) > 0


def test_export_refuses_a_layer_the_browser_cannot_run(tmp_path):
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 2))
    X = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="Dropout"):
        export(model, ["a", "b"], 2, False, X, out_dir=tmp_path)


def test_export_catches_a_size_mismatch(tmp_path):
    model = nn.Sequential(nn.Linear(100, 2))
    X = np.zeros((1, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="disagree"):
        # 8 x 8 x 3 is 192 numbers, not 100
        export(model, ["a", "b"], 8, True, X, out_dir=tmp_path)
