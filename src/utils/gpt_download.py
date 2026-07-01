"""Download and load OpenAI's pretrained GPT-2 weights.

Adapted from the "Build a Large Language Model (From Scratch)" reference
utility. Downloads the TensorFlow checkpoint for a given GPT-2 size and returns
the ``(settings, params)`` pair, where ``params`` is the nested weight dict that
:func:`src.utils.gpt2_weights.load_weights_into_gpt` expects.

Requires ``tensorflow`` and ``tqdm`` (optional extras — not needed for training
your own model from scratch)::

    pip install tensorflow tqdm
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np


def download_and_load_gpt2(model_size: str, models_dir: str):
    """Download (if needed) and load GPT-2 weights for ``model_size``.

    Parameters
    ----------
    model_size:
        One of ``"124M"``, ``"355M"``, ``"774M"``, ``"1558M"``.
    models_dir:
        Directory to cache the downloaded checkpoint files in.

    Returns
    -------
    (settings, params):
        ``settings`` is the model's ``hparams.json`` as a dict; ``params`` is the
        nested weight dictionary (``wte``, ``wpe``, ``blocks``, ``g``, ``b``).
    """
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe",
    ]

    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        _download_file(file_url, file_path)

    # Load settings and parameters from the downloaded checkpoint.
    import tensorflow as tf  # imported lazily so torch-only workflows don't need TF

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = _load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return settings, params


def _download_file(url: str, destination: str) -> None:
    """Download ``url`` to ``destination`` with a tqdm progress bar."""
    try:
        from tqdm import tqdm
    except ImportError:  # progress bar is optional
        tqdm = None

    with urllib.request.urlopen(url) as response:
        file_size = int(response.headers.get("Content-Length", 0))
        if os.path.exists(destination) and os.path.getsize(destination) == file_size:
            print(f"File already exists and is up-to-date: {destination}")
            return

        block_size = 1024
        progress = None
        if tqdm is not None:
            progress = tqdm(
                total=file_size, unit="iB", unit_scale=True,
                desc=os.path.basename(url),
            )
        with open(destination, "wb") as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                if progress is not None:
                    progress.update(len(chunk))
        if progress is not None:
            progress.close()


def _load_gpt2_params_from_tf_ckpt(ckpt_path: str, settings: dict) -> dict:
    """Read a GPT-2 TF checkpoint into the nested ``params`` dictionary."""
    import tensorflow as tf

    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Names look like "model/h0/attn/c_attn/w"; route into the nested dict.
        variable_name_parts = name.split("/")[1:]  # drop the "model/" prefix
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        target_dict[variable_name_parts[-1]] = variable_array

    return params
