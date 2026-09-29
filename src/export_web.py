"""Turn a trained model into the files the web page reads.

This file is complete. You do not need to change it.

It writes four things into docs/:
    model.json     the list of layers, the input settings, and your class names
    weights.bin    every number in the model, as raw float32
    selftest.json  three test images plus the answers Python gave for them
    labels.json    just your class names, so you can read them at a glance

The self test is the important one. When the page opens it runs those three
images through the JavaScript copy of your model and compares the result with
what Python got. If the two disagree, the badge at the top of the page turns
red, and that means the page is preparing images differently from the way you
prepared them during training. That is the single most common way this
assignment goes wrong.
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

FORMAT = "mas1004-mlp-v1"


def _describe_layers(model):
    """Walk the model and refuse anything the web page cannot run."""
    if not isinstance(model, nn.Sequential):
        raise TypeError(
            "build_model must return an nn.Sequential. "
            f"It returned {type(model).__name__}."
        )

    layers, blobs, offset = [], [], 0
    for position, module in enumerate(model):
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu().numpy().astype(np.float32)
            bias = module.bias.detach().cpu().numpy().astype(np.float32)
            n_in, n_out = weight.shape[1], weight.shape[0]

            layers.append({
                "type": "linear",
                "in": n_in,
                "out": n_out,
                "w": [offset, weight.size],
                "b": [offset + weight.size, bias.size],
            })
            blobs.append(weight.reshape(-1))  # row major, [out, in]
            blobs.append(bias)
            offset += weight.size + bias.size

        elif isinstance(module, nn.ReLU):
            layers.append({"type": "relu"})

        elif isinstance(module, nn.Flatten):
            continue  # harmless, the input is already flat

        else:
            raise TypeError(
                f"Layer {position} is {type(module).__name__}, which the web "
                "page cannot run. Use only nn.Linear and nn.ReLU."
            )

    if not layers or layers[-1]["type"] != "linear":
        raise TypeError("The last layer must be nn.Linear.")
    return layers, np.concatenate(blobs), offset


def _row_to_png(row, image_size, channels):
    """Turn one row of X back into a small PNG, and back into numbers again.

    We go through 8 bit pixels on purpose. The browser will only ever see 8 bit
    pixels, so the numbers Python compares against have to come from the same
    place, or the self test would fail for no good reason.
    """
    shape = (image_size, image_size, channels) if channels == 3 else (image_size, image_size)
    pixels = np.clip(row.reshape(shape) * 255.0, 0, 255).round().astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB" if channels == 3 else "L")

    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    requantised = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
    return base64.b64encode(buffer.getvalue()).decode("ascii"), requantised


def export(model, class_names, image_size, color, X_sample, out_dir="docs"):
    """Write model.json, weights.bin, selftest.json and labels.json.

    Arguments
        model        the trained nn.Sequential from build_model
        class_names  the list load_folder gave you, in the same order
        image_size   the image_size you passed to load_folder
        color        the color flag you passed to load_folder
        X_sample     a few rows of your test set, np.float32 (n, D), n >= 1.
                     Three of them end up in the self test.
        out_dir      where to write. Leave this alone.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    channels = 3 if color else 1

    layers, flat, total = _describe_layers(model)

    expected_dim = image_size * image_size * channels
    first_linear = next(layer for layer in layers if layer["type"] == "linear")
    if first_linear["in"] != expected_dim:
        raise ValueError(
            f"The model takes {first_linear['in']} numbers but a "
            f"{image_size}x{image_size} image with {channels} channel(s) is "
            f"{expected_dim} numbers. image_size, color and input_dim disagree."
        )
    if layers[-1]["out"] != len(class_names):
        raise ValueError(
            f"The model has {layers[-1]['out']} outputs but you gave "
            f"{len(class_names)} class names."
        )

    (out_path / "weights.bin").write_bytes(flat.tobytes())

    (out_path / "model.json").write_text(json.dumps({
        "format": FORMAT,
        "input": {"size": image_size, "channels": channels},
        "labels": list(class_names),
        "layers": layers,
        "n_floats": int(total),
    }, indent=2))

    (out_path / "labels.json").write_text(json.dumps(list(class_names), indent=2))

    model.eval()
    cases = []
    for row in np.asarray(X_sample, dtype=np.float32)[:3]:
        png_b64, requantised = _row_to_png(row, image_size, channels)
        with torch.no_grad():
            logits = model(torch.from_numpy(requantised[None, :])).numpy()[0]
        cases.append({
            "png": png_b64,
            "logits": [round(float(v), 4) for v in logits],
        })
    (out_path / "selftest.json").write_text(json.dumps({"cases": cases}))

    print(f"wrote {out_path}/model.json      {len(layers)} layers")
    print(f"wrote {out_path}/weights.bin     {total} numbers, "
          f"{total * 4 / 1024:.0f} KB")
    print(f"wrote {out_path}/selftest.json   {len(cases)} cases")
    print(f"wrote {out_path}/labels.json     {', '.join(class_names)}")
