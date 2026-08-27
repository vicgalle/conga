"""Public API and numerical-contract checks."""
from __future__ import annotations

import torch

from conga import DTYPE, GaussianBlock, TargetPosterior, discover_gaussian_block


class _ScaleFiber(TargetPosterior):
    name = "test_scale_fiber"
    dim = 5

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        eta = value[..., 0]
        fiber = value[..., 1:]
        mean = eta.unsqueeze(-1) * torch.tensor(
            [1.0, -0.5, 0.25, 0.0], dtype=DTYPE
        )
        residual = (fiber - mean) * torch.exp(-eta.unsqueeze(-1))
        return -0.5 * eta.square() - 4.0 * eta - 0.5 * residual.square().sum(-1)

    def constrain(self, value: torch.Tensor):
        return {"eta": value[..., 0], "fiber": value[..., 1:]}


def test_discovers_and_factorizes_gaussian_fiber():
    target = _ScaleFiber()
    found = discover_gaussian_block(target, seed=3)
    assert found.g_idx.tolist() == [1, 2, 3, 4]

    block = GaussianBlock(target, found.g_idx, probe_etas=found.probe_etas)
    assert block.d_g == 4
    assert block.d_h == 1
    assert block.n_colors == 1

    etas = torch.tensor([[-0.4], [0.2], [0.8]], dtype=DTYPE)
    assert block.joint_identity_residual(
        etas, gen=torch.Generator().manual_seed(8)
    ) < 1e-10

    value, gradient = block.collapsed_vg(etas)
    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()
