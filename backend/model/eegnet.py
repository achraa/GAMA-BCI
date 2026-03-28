"""model/eegnet.py — EEGNet + Focal Loss (PyTorch)."""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet untuk P300 detection.
    Input : (batch, 1, n_channels, n_timepoints)
    Output: (batch, 1) probabilitas P300

    Referensi: Lawhern et al. 2018 — EEGNet: A Compact CNN for EEG-Based BCIs
    """
    def __init__(self, n_channels=16, n_timepoints=204, F1=8, D=2, dropout=0.5):
        super().__init__()
        F2 = F1 * D

        # Block 1: Temporal convolution
        self.block1_temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        # Block 1: Depthwise spatial convolution
        self.block1_spatial = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        # Block 2: Separable convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, kernel_size=(1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        self._flat_size = self._get_flat_size(n_channels, n_timepoints)
        self.classifier = nn.Sequential(
            nn.Linear(self._flat_size, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def _get_flat_size(self, n_ch, n_tp):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_ch, n_tp)
            x = self.block1_temporal(x)
            x = self.block1_spatial(x)
            x = self.block2(x)
            return x.numel()

    def forward(self, x):
        x = self.block1_temporal(x)
        x = self.block1_spatial(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.classifier(x))


class FocalLoss(nn.Module):
    """
    Focal Loss untuk class imbalance 1:11.
    FL = -α(1-p)^γ log(p)
    α=0.75 → lebih berat ke target (P300)
    γ=2.0  → fokus ke hard examples
    """
    def __init__(self, alpha=0.92, gamma=3.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred    = pred.clamp(1e-7, 1-1e-7).squeeze()
        target  = target.float()
        bce     = -(target * torch.log(pred) + (1-target) * torch.log(1-pred))
        p_t     = target * pred + (1-target) * (1-pred)
        alpha_t = target * self.alpha + (1-target) * (1-self.alpha)
        return (alpha_t * (1-p_t)**self.gamma * bce).mean()
