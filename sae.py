"""
Sparse Autoencoder (SAE) in PyTorch.

Architecture:  x --[W_enc + b_enc]--> ReLU --> h --[W_dec]--> x_hat
Loss:          MSE(x, x_hat)  +  l1_coef * mean(|h|)

After every optimizer step the decoder columns are re-normalized to unit
norm (standard practice for SAEs — prevents the model from hiding sparsity
in column scaling).

Works transparently on CPU or CUDA; pass device='cuda' to train_sae.
"""
import torch
import torch.nn as nn
import numpy as np


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, dict_size: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        nn.init.orthogonal_(self.decoder.weight)  # columns start unit-norm

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        return self.decoder(h), h

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def _renorm_decoder(self):
        with torch.no_grad():
            w = self.decoder.weight          # shape (input_dim, dict_size)
            norms = w.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.copy_(w / norms)


def train_sae(X: np.ndarray, dict_size: int,
              l1_coef: float = 0.01,
              lr: float = 1e-3,
              n_epochs: int = 100,
              batch_size: int = 512,
              device: str = 'cpu',
              verbose: bool = True) -> SparseAutoencoder:
    """
    Train a sparse autoencoder on embedding matrix X (N × d).

    The returned model is in eval mode on `device`.
    """
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    N, d = X_t.shape

    model = SparseAutoencoder(d, dict_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=device)
        total_loss = recon_total = sparse_total = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            batch = X_t[perm[i : i + batch_size]]
            optimizer.zero_grad()
            x_hat, h = model(batch)
            recon  = ((batch - x_hat) ** 2).mean()
            sparse = h.abs().mean()
            loss   = recon + l1_coef * sparse
            loss.backward()
            optimizer.step()
            model._renorm_decoder()
            total_loss  += loss.item()
            recon_total += recon.item()
            sparse_total += sparse.item()
            n_batches += 1

        if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"    epoch {epoch+1:>4d}/{n_epochs}  "
                  f"loss={total_loss/n_batches:.4f}  "
                  f"recon={recon_total/n_batches:.4f}  "
                  f"sparse={sparse_total/n_batches:.4f}")

    model.eval()
    return model


def get_activations(model: SparseAutoencoder,
                    X: np.ndarray,
                    device: str = 'cpu',
                    batch_size: int = 2048) -> np.ndarray:
    """Run encoder on X, return hidden activations H (N × dict_size)."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    parts = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            h = model.encode(X_t[i : i + batch_size])
            parts.append(h.cpu().numpy())
    return np.vstack(parts)
