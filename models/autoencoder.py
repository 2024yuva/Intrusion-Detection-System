import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

class SimpleSeqAutoencoder(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, n_features: int, hidden: int = 32):
        if TORCH_AVAILABLE:
            super().__init__()
            self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
            self.decoder = nn.LSTM(input_size=hidden, hidden_size=n_features, batch_first=True)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        B, T, _ = x.shape
        dec_in = enc_out[:, -1:, :].repeat(1, T, 1)
        out, _ = self.decoder(dec_in)
        return out

def windowize(X, win=5, stride=1):
    if len(X) < win:
        return np.expand_dims(X[:win], 0)
    chunks = []
    for i in range(0, len(X) - win + 1, stride):
        chunks.append(X[i:i+win])
    return np.stack(chunks, axis=0)

def train_autoencoder(X, n_features, epochs=5, lr=1e-3, win=5):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSeqAutoencoder(n_features, hidden=32).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xw = windowize(X, win=win)
    Xw = torch.tensor(Xw, dtype=torch.float32, device=device)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        recon = model(Xw)
        loss = loss_fn(recon, Xw)
        loss.backward()
        opt.step()
    return {"state_dict": model.state_dict(), "win": win, "n_features": n_features}

def inference_autoencoder(params, X):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    win = params["win"]
    n_features = params["n_features"]
    model = SimpleSeqAutoencoder(n_features, hidden=32).to(device)
    model.load_state_dict(params["state_dict"])
    model.eval()
    Xw = windowize(X, win=win)
    Xw_t = torch.tensor(Xw, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon = model(Xw_t)
        mse = ((recon - Xw_t) ** 2).mean(dim=(1,2)).cpu().numpy()
    scores = np.repeat(mse, win)
    if scores.max() > 0:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    labels = (scores > 0.5).astype(int)
    return labels, scores
