import torch
import torch.nn as nn
import torch.nn.functional as F


class Learnable3DLUT(nn.Module):
    """Image-adaptive 3D Lookup Table for photo enhancement.

    Learns multiple basis 3D LUTs and a lightweight CNN that predicts
    per-image blending weights.  The weighted combination of LUTs is
    applied via trilinear interpolation (F.grid_sample on a 3D volume).

    Args:
        lut_dim: Resolution of each LUT axis (e.g. 17).
        n_luts: Number of basis LUTs to learn.
    """

    def __init__(self, lut_dim: int = 17, n_luts: int = 3):
        super().__init__()
        self.lut_dim = lut_dim
        self.n_luts = n_luts

        identity = self._identity_lut()
        self.luts = nn.Parameter(
            identity.unsqueeze(0).repeat(n_luts, 1, 1, 1, 1)
            + torch.randn(n_luts, 3, lut_dim, lut_dim, lut_dim) * 0.001
        )

        self.weight_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, n_luts),
            nn.Softmax(dim=1),
        )

    # ------------------------------------------------------------------
    def _identity_lut(self) -> torch.Tensor:
        D = self.lut_dim
        lut = torch.zeros(3, D, D, D)
        for i in range(D):
            v = i / (D - 1)
            lut[0, :, :, i] = v
            lut[1, :, i, :] = v
            lut[2, i, :, :] = v
        return lut

    # ------------------------------------------------------------------
    def _apply_lut(self, lut: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """Apply one 3D LUT to an image via F.grid_sample (trilinear).

        lut : (3, D, D, D)
        img : (B, 3, H, W)  values in [0, 1]
        returns : (B, 3, H, W)
        """
        B, C, H, W = img.shape
        D = self.lut_dim

        grid = img.permute(0, 2, 3, 1) * 2 - 1          # (B, H, W, 3) in [-1,1]
        grid = grid[:, :, :, [2, 1, 0]]                   # reorder to (B, G, R) for grid_sample
        grid = grid.unsqueeze(1)                           # (B, 1, H, W, 3)

        lut_5d = lut.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, 3, D, D, D)

        out = F.grid_sample(
            lut_5d, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return out.squeeze(2)                              # (B, 3, H, W)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 3, H, W) in [0, 1]"""
        weights = self.weight_net(x)                       # (B, n_luts)

        outs = torch.stack(
            [self._apply_lut(self.luts[i], x) for i in range(self.n_luts)],
            dim=1,
        )                                                   # (B, n_luts, 3, H, W)

        w = weights.view(-1, self.n_luts, 1, 1, 1)
        result = (outs * w).sum(dim=1)                      # (B, 3, H, W)
        return result.clamp(0, 1)
