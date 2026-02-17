import pathlib
import textwrap
import typing
import types
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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
    """Hooks each MLP layer and streams gate activations to per-layer parquet files.

    Output layout:
        <output_dir>/layer_000.parquet
        <output_dir>/layer_001.parquet
        ...

    Each file is a single-column table:
        activation: fixed_size_list<float32>[n_neurons]   (one row = one token position)

    Row groups correspond to individual forward-pass batches, so the writer never
    holds more than one batch in memory at a time.
    """

    def __init__(self, gemma_lm, output_dir: str = "activations"):
        self.gemma_lm = gemma_lm
        self.n_layers = len(gemma_lm.layers)
        self.d_model = gemma_lm.config.hidden_size
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._schema = pa.schema([
            pa.field("activation", pa.list_(pa.float32(), self.d_model))
        ])
        # One ParquetWriter per layer, opened lazily on first write.
        self._writers: list[pq.ParquetWriter | None] = [None] * self.n_layers
        self.hooks: list = []

    # Writer helpers
    def _writer(self, layer_idx: int) -> pq.ParquetWriter:
        if self._writers[layer_idx] is None:
            path = self.output_dir / f"layer_{layer_idx:03d}.parquet"
            self._writers[layer_idx] = pq.ParquetWriter(str(path), self._schema)
        return self._writers[layer_idx]

    def close(self):
        "Flush and close all open parquet writers."
        for w in self._writers:
            if w is not None:
                w.close()
        self._writers = [None] * self.n_layers

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def make_mlp_hook(self, layer_idx: int):
        """Create a closure over layer_idx. Each layer gets its own hook."""
        def hook(module_, inputs_, output_):
            x = output_
            if x.dim() == 3:
                x = x.flatten(0, 1)             # (batch * seq, d_model)
            flat = pa.array(x.cpu().float().numpy().ravel(), type=pa.float32())
            col = pa.FixedSizeListArray.from_arrays(flat, self.d_model)
            self._writer(layer_idx).write_table(pa.table({"activation": col}))
            return output_
        return hook

    def register_hooks(self):
        "Register a forward hook on every MLP layer."
        for idx, layer in enumerate(self.gemma_lm.layers):
            handle = layer.mlp.register_forward_hook(self.make_mlp_hook(idx))
            self.hooks.append(handle)
        print(f"Added {len(self.hooks)} hooks → writing to {self.output_dir}/")

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


tracker = GateTracker(gemma_lm, output_dir="activations")
tracker.register_hooks()

model.eval()
for batch_texts in tqdm(batch_iter(texts, BATCH_SIZE), total=len(texts) // BATCH_SIZE):
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(model.device)
    model(**inputs)   # activations are streamed to disk via hooks

tracker.close()
print("Done. Parquet files written to activations/")
