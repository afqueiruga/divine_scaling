import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class ActivationDataset(Dataset):
    """Load scraped FFN activations from an HDF5 file.

    One sample = (x, ffn(x)) for a single token at a given layer.

    Args:
        path:   Path to the HDF5 file produced by `scrape_llm_activations.py`.
        layer:  Which layer to read (must have been recorded and written to the HDF5 file).
        device: Device to load the tensors onto (default cpu).
        dtype:  Type of the torch tensors (default float32).
    """

    def __init__(self, path: str, layer: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
        print("Loading layer", layer, "from", path)
        with h5py.File(path, "r") as f:
            grp = f[f"layer_{layer:03d}"]
            self.X = torch.from_numpy(grp["input"][:]).to(device, dtype=dtype)   # (n_tokens, d_model)
            self.Y = torch.from_numpy(grp["output"][:]).to(device, dtype=dtype)
        self.n_tokens, self.d_model = self.X.shape
        print("Loaded ", self.n_tokens, "tokens with d_model =", self.d_model)

    def __len__(self) -> int:
        return self.n_tokens

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def load_activations_plain(
    path: str,
    layer: int,
    train_frac: float = 0.9,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    n_data: int = -1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = ActivationDataset(path, layer, device=device, dtype=dtype)
    # TODO: Split into train/val
    return dataset.X[:n_data,...], dataset.Y[:n_data,...], dataset.X[:n_data,...], dataset.Y[:n_data,...]


# Run this as a script to test the loader
# python3 -m divine_scaling.problems_activations
if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_activations_plain(
        "data/activations.h5",
        layer=12,
        dtype=torch.float32,
    )
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
