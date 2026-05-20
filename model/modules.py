"""Pluggable feature enhancement modules.

Modules:
    - SEBlock: Channel attention via global average pooling (Squeeze-and-Excitation).
    - SpectralConv1D: 1D convolution along the spectral (band) dimension.
    - DiagonalBandGate: Static hard top-k band selection at the network input.
"""

import math

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Performs: GAP -> FC(C, C//r) -> ReLU -> FC(C//r, C) -> Sigmoid -> channel-wise scaling.

    Args:
        channels: Number of input/output channels.
        reduction: Reduction ratio for the bottleneck (default 16).
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale


class SpectralConv1D(nn.Module):
    """1D convolution along the spectral (band) dimension.

    Learns local correlations between adjacent NIR bands (23nm spacing).
    Includes a residual connection.

    Args:
        num_channels: Number of channels to process.
        kernel_size: 1D convolution kernel size (default 3).
    """

    def __init__(self, num_channels=16, kernel_size=3):
        super().__init__()
        self.num_channels = num_channels
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=True)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        x_flat = x_flat.reshape(B * H * W, 1, C)
        x_flat = self.conv(x_flat)
        x_flat = x_flat.reshape(B, H * W, C).permute(0, 2, 1)
        x_out = x_flat.view(B, C, H, W)
        return self.relu(self.bn(x_out + x))


class DiagonalBandGate(nn.Module):
    """Static diagonal band-selection gate with straight-through hard top-k.

    Selects exactly ``k`` of ``num_bands`` input channels. The gate is a single
    learnable score vector (one scalar per band), so the same bands are kept for
    every image — this is a *selection* mechanism (read off the kept bands and
    physically drop the others at deployment), not the per-sample reweighting of
    an SE block.

    Score / soft-prob / mask::

        s_b   = theta_b + beta * prior_b
        g_b   = sigmoid(s_b / tau)              # soft keep-prob (gradient path)
        m     = top-k(s)                        # hard k-hot mask
        m_st  = m + g - g.detach()              # straight-through estimator

    During training the forward pass applies the HARD mask ``m`` (the backbone
    always sees exactly k bands, matching deployment), while gradients flow to
    ALL bands through ``g`` — so an unselected band can still earn its way in,
    instead of the gate locking onto its initialization.

    ``tau`` and ``beta`` are annealed by :meth:`set_progress` (call once per
    epoch with ``frac`` in [0, 1]):
        - ``tau``  : ``tau_start`` -> ``tau_end`` (cosine), soft -> near-hard.
        - ``beta`` : ``beta_start`` -> ``beta_end`` (linear over ``beta_anneal_frac``),
          prior-guided early -> data-driven late.

    The *deployment* selection (used by :meth:`selected_bands`) always evaluates
    the score at ``beta_end`` so it is independent of the current schedule state.

    Args:
        num_bands: Number of input channels (= encoder ``in_channels``).
        k: Number of bands to keep (1 <= k <= num_bands).
        prior: Optional length-``num_bands`` band-importance vector (HSI prior).
            ``None`` -> zeros (pure data-driven selection).
        tau_start, tau_end: Temperature annealing endpoints.
        beta_start, beta_end: Prior-strength annealing endpoints.
        beta_anneal_frac: Fraction of training over which beta reaches beta_end.
        prior_init_scale: Warm-start ``theta`` at ``prior_init_scale * prior``.
        theta_init_noise: Std of Gaussian added to theta init (breaks top-k ties
            when there is no prior / no warm-start).
    """

    def __init__(self, num_bands, k, prior=None,
                 tau_start=1.0, tau_end=0.05,
                 beta_start=1.0, beta_end=0.0,
                 beta_anneal_frac=0.5, prior_init_scale=0.0,
                 theta_init_noise=0.01):
        super().__init__()
        if not (1 <= k <= num_bands):
            raise ValueError(f"k must be in [1, {num_bands}], got {k}")

        self.num_bands = num_bands
        self.k = k
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_anneal_frac = float(beta_anneal_frac)

        if prior is None:
            prior_t = torch.zeros(num_bands, dtype=torch.float32)
        else:
            prior_t = torch.as_tensor(prior, dtype=torch.float32).flatten()
            if prior_t.numel() != num_bands:
                raise ValueError(
                    f"prior length {prior_t.numel()} != num_bands {num_bands}"
                )
        self.register_buffer("prior", prior_t)

        theta0 = prior_init_scale * prior_t.clone()
        if theta_init_noise > 0:
            theta0 = theta0 + theta_init_noise * torch.randn(num_bands)
        self.theta = nn.Parameter(theta0)

        # Schedule state; defaults to fully-annealed so a freshly loaded model
        # (e.g. in eval.py without any set_progress call) behaves as deployment.
        self.register_buffer("_tau", torch.tensor(self.tau_end))
        self.register_buffer("_beta", torch.tensor(self.beta_end))

    def set_progress(self, frac):
        """Update tau/beta for the current training progress (frac in [0, 1])."""
        frac = float(min(max(frac, 0.0), 1.0))
        cos = 0.5 * (1.0 + math.cos(math.pi * frac))  # 1 -> 0 as frac 0 -> 1
        tau = self.tau_end + (self.tau_start - self.tau_end) * cos
        if self.beta_anneal_frac <= 0:
            beta = self.beta_end
        else:
            t = min(frac / self.beta_anneal_frac, 1.0)
            beta = self.beta_start + (self.beta_end - self.beta_start) * t
        self._tau.fill_(float(tau))
        self._beta.fill_(float(beta))

    def _scores(self):
        return self.theta + self._beta * self.prior

    def _hard_mask(self, scores):
        idx = torch.topk(scores, self.k).indices
        mask = torch.zeros_like(scores)
        mask[idx] = 1.0
        return mask

    def forward(self, x):
        scores = self._scores()
        if self.training:
            tau = self._tau.clamp_min(1e-4)
            g = torch.sigmoid(scores / tau)
            mask = self._hard_mask(scores)
            mask = mask + g - g.detach()  # straight-through: forward=hard, grad=soft
        else:
            mask = self._hard_mask(scores)
        return x * mask.view(1, -1, 1, 1)

    @torch.no_grad()
    def selected_bands(self):
        """Return the sorted list of kept band indices at the deployment gate."""
        scores = self.theta + self.beta_end * self.prior
        return torch.topk(scores, self.k).indices.sort().values.tolist()

    def extra_repr(self):
        return (f"num_bands={self.num_bands}, k={self.k}, "
                f"tau={self.tau_start}->{self.tau_end}, "
                f"beta={self.beta_start}->{self.beta_end}")
