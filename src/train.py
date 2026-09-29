"""Build a model and train it. YOU write the bodies of these functions.

This is the same kind of model you saw in class: a few Linear layers with ReLU
between them. No convolutions this time. We get to those in week 10.

Run `pytest tests/test_train.py` after you fill them in.
"""

import torch
import torch.nn as nn


def build_model(input_dim, num_classes, hidden_sizes=(128,)):
    """Return an untrained model.

    Arguments
        input_dim    int, the D from load_folder. For 32x32 color this is 3072.
        num_classes  int, how many classes you have
        hidden_sizes tuple of int, one number per hidden layer.
                     (128,) means one hidden layer of 128 neurons.
                     (256, 64) means two hidden layers.
                     () means no hidden layer at all, just one Linear.

    Returns a torch.nn.Sequential

    The export script only understands two kinds of layer, nn.Linear and
    nn.ReLU, and it expects them to alternate: Linear, ReLU, Linear, ReLU, ...,
    Linear. The last layer must be Linear and must have num_classes outputs.
    Do not put a softmax at the end. The loss function adds it for you, and the
    web page adds it for you.

    If you use anything else, for example nn.Dropout or nn.BatchNorm1d, the
    export script will refuse to run and your web demo will not work.
    """
    raise NotImplementedError("Problem 3: fill in build_model")


def train(model, X_train, y_train, X_test, y_test,
          epochs=30, lr=0.01, batch_size=32):
    """Train the model and report what happened at every epoch.

    Arguments
        model        what build_model returned
        X_train      np.float32 (N, D)      y_train  np.int64 (N,)
        X_test       np.float32 (M, D)      y_test   np.int64 (M,)
        epochs       int, how many times to go through the training set
        lr           float, the learning rate
        batch_size   int, how many rows per step

    Returns a dict named history with four keys. Each value is a list of
    `epochs` numbers, one per epoch:
        "train_loss"  average loss over the training set
        "test_loss"   average loss over the test set
        "train_acc"   accuracy on the training set, between 0.0 and 1.0
        "test_acc"    accuracy on the test set, between 0.0 and 1.0

    The model must be trained in place, so that after this function returns,
    `model` is the trained one.

    Use nn.CrossEntropyLoss. It expects raw outputs, not probabilities, which is
    why build_model has no softmax at the end.

    Do not compute the gradient while you are measuring the test numbers, and
    remember that measuring is not learning: the test rows must never be used to
    update the parameters.
    """
    raise NotImplementedError("Problem 3: fill in train")


def save(model, path):
    """Save a trained model. This one is written for you."""
    torch.save(model, path)


def load(path):
    """Load a model saved by `save`. This one is written for you."""
    model = torch.load(path, weights_only=False)
    model.eval()
    return model
