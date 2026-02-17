import textwrap
import typing
import types
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets


CORPUS_LENGTH = 100
CHECKPOINT = "google/gemma-3-1b-it"
BATCH_SIZE = 8
MAX_LENGTH = 512

torch.set_grad_enabled(False)


class GateTracker():
    """Hooks each MLP layer and streams activations to a single HDF5 file.

    Output layout (one file, datasets resized as batches arrive):
        activations.h5
            layer_000/
                input   (n_tokens, d_model)  float32
                output  (n_tokens, d_model)  float32
            layer_001/
                ...
    """

    def __init__(self, gemma_lm, output_path: str = "activations.h5"):
        self.gemma_lm = gemma_lm
        self.n_layers = len(gemma_lm.layers)
        self.d_model = gemma_lm.config.hidden_size

        self._file = h5py.File(output_path, "w")
        # Pre-create one resizable dataset per layer per signal.
        for i in range(self.n_layers):
            grp = self._file.create_group(f"layer_{i:03d}")
            for signal in ("input", "output"):
                grp.create_dataset(
                    signal,
                    shape=(0, self.d_model),
                    maxshape=(None, self.d_model),  # unlimited along axis 0
                    dtype="float32",
                    chunks=(512, self.d_model),     # one chunk ≈ 512 tokens
                )
        # Set by the main loop before each forward pass so hooks can mask padding.
        self.attention_mask: torch.Tensor | None = None
        self.hooks: list = []

    def close(self):
        "Flush and close the HDF5 file."
        self._file.close()

    def make_mlp_hook(self, layer_idx: int):
        """Create a closure over layer_idx. Each layer gets its own hook."""
        def hook(module_, inputs_, output_):
            # attention_mask: (batch, seq) — 1 for real tokens, 0 for padding.
            mask = self.attention_mask.flatten().bool().cpu()  # (batch*seq,)
            x = inputs_[0].flatten(0, 1).cpu().float()[mask].numpy()  # (real_tokens, d_model)
            y = output_.flatten(0, 1).cpu().float()[mask].numpy()
            for signal, arr in (("input", x), ("output", y)):
                ds = self._file[f"layer_{layer_idx:03d}/{signal}"]
                n_new = arr.shape[0]
                ds.resize(ds.shape[0] + n_new, axis=0)
                ds[-n_new:] = arr
            return output_
        return hook

    def register_hooks(self):
        "Register a forward hook on every MLP layer."
        for idx, layer in enumerate(self.gemma_lm.layers):
            handle = layer.mlp.register_forward_hook(self.make_mlp_hook(idx))
            self.hooks.append(handle)
        print(f"Added {len(self.hooks)} hooks.")

    def clear_hooks(self):
        "Remove all hooks registered by this tracker."
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def __del__(self):
        self.clear_hooks()
        self.close()

    def nuke_hooks(self):
        "Emergency button for when you lost pointers to the hooks. Use sparingly, as some models actually use hooks."
        self.clear_hooks()
        for layer in self.gemma_lm.layers:
            layer.mlp._forward_hooks.clear()


print(f"Loading {CHECKPOINT}...")
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
# Gemma has two different naming schemes depending on if there is a vision tower.
# This variable is a reference into the language model trunk in the AutoModel.
# It is a reference to a field inside of `model`
if CHECKPOINT == "google/gemma-3-1b-it":
    gemma_lm = model.model
else:
    gemma_lm = model.language_model

print("Loading NeelNanda/pile-10k...")
dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")
texts = dataset["text"][: CORPUS_LENGTH]
# Filter out very short texts
# texts = [t for t in texts if len(t) > 50]
print(f"  Using {len(texts)} texts")


def batch_iter(texts: list[str], batch_size: int):
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


tracker = GateTracker(gemma_lm, output_path="activations.h5")
tracker.register_hooks()

print("Evaluating model...")
model.eval()
for batch_texts in tqdm(batch_iter(texts, BATCH_SIZE), total=len(texts) // BATCH_SIZE):
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(model.device)
    tracker.attention_mask = inputs["attention_mask"]
    model(**inputs)   # activations are streamed to disk via hooks

tracker.close()
print("Done. Activations written to activations.h5")
