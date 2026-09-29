"""Tests for Problem 3. Run: pytest tests/test_train.py"""

import copy

import torch
import torch.nn as nn

from train import build_model, train


def test_model_is_a_sequential_of_linear_and_relu():
    model = build_model(12, 4, hidden_sizes=(8, 6))
    assert isinstance(model, nn.Sequential), (
        f"build_model must return nn.Sequential, got {type(model).__name__}"
    )
    kinds = [type(layer) for layer in model]
    assert kinds == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear], (
        "With two hidden sizes the layers must be "
        "Linear, ReLU, Linear, ReLU, Linear. "
        f"Got {[k.__name__ for k in kinds]}. The export script cannot handle "
        "anything else."
    )


def test_no_hidden_layers_means_one_linear():
    model = build_model(12, 4, hidden_sizes=())
    assert [type(layer) for layer in model] == [nn.Linear]


def test_sizes_line_up():
    model = build_model(12, 4, hidden_sizes=(8,))
    output = model(torch.zeros(5, 12))
    assert output.shape == (5, 4), (
        f"A batch of 5 should come out as (5, 4), got {tuple(output.shape)}"
    )


def test_there_is_no_softmax_at_the_end():
    model = build_model(12, 4, hidden_sizes=(8,))
    assert isinstance(model[-1], nn.Linear), (
        "The last layer must be nn.Linear. CrossEntropyLoss wants raw outputs, "
        "and the web page applies softmax itself."
    )
    with torch.no_grad():
        rows = model(torch.randn(3, 12))
    assert not torch.allclose(rows.sum(dim=1), torch.ones(3), atol=1e-3), (
        "The outputs add up to 1, so there is a softmax in the model. Remove it."
    )


def test_history_has_the_right_shape(blobs):
    X, y = blobs
    model = build_model(3, 3, hidden_sizes=(16,))
    history = train(model, X, y, X, y, epochs=4, lr=0.05, batch_size=16)

    for key in ("train_loss", "test_loss", "train_acc", "test_acc"):
        assert key in history, f"history is missing the key {key!r}"
        assert len(history[key]) == 4, (
            f"history[{key!r}] should have one number per epoch (4), "
            f"got {len(history[key])}"
        )
    for key in ("train_acc", "test_acc"):
        assert all(0.0 <= v <= 1.0 for v in history[key]), (
            f"history[{key!r}] must be between 0 and 1, not a percentage."
        )


def test_training_actually_changes_the_model(blobs):
    X, y = blobs
    model = build_model(3, 3, hidden_sizes=(16,))
    before = copy.deepcopy(model.state_dict())
    train(model, X, y, X, y, epochs=3, lr=0.05, batch_size=16)

    changed = any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
    assert changed, (
        "The parameters are exactly what they were before training. "
        "Did you forget optimiser.step(), or train a copy of the model?"
    )


def test_training_learns_an_easy_problem(blobs):
    X, y = blobs
    model = build_model(3, 3, hidden_sizes=(16,))
    history = train(model, X, y, X, y, epochs=40, lr=0.05, batch_size=16)

    assert history["train_loss"][-1] < history["train_loss"][0], (
        "The loss did not go down at all over 40 epochs."
    )
    assert history["train_acc"][-1] > 0.9, (
        "These are three well separated blobs. A working training loop reaches "
        f"over 90% on them. Yours reached {history['train_acc'][-1]:.0%}."
    )
