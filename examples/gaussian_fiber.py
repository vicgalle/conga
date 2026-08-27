"""A minimal CONGA pass on a target with a heteroscedastic Gaussian fiber."""
from __future__ import annotations

import torch

from conga import DTYPE, GaussianBlock, TargetPosterior, discover_gaussian_block


class ScaleFiber(TargetPosterior):
    """One nonlinear scale coordinate above four conditionally Gaussian ones."""

    name = "scale_fiber"
    dim = 5

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        eta = value[..., 0]
        fiber = value[..., 1:]
        mean = eta.unsqueeze(-1) * torch.tensor(
            [1.0, -0.5, 0.25, 0.0], dtype=DTYPE
        )
        standardized = (fiber - mean) * torch.exp(-eta.unsqueeze(-1))
        return -0.5 * eta.square() - 4.0 * eta - 0.5 * standardized.square().sum(-1)

    def constrain(self, value: torch.Tensor):
        return {"eta": value[..., 0], "fiber": value[..., 1:]}


def main() -> None:
    target = ScaleFiber()
    found = discover_gaussian_block(target, seed=3)
    block = GaussianBlock(target, found.g_idx, probe_etas=found.probe_etas)

    eta = torch.tensor([[0.2]], dtype=DTYPE)
    mean, precision, _ = block.conditional_moments(eta)
    residual = block.joint_identity_residual(
        eta, gen=torch.Generator().manual_seed(7)
    )

    print(f"discovered fiber coordinates: {found.g_idx.tolist()}")
    print(f"core coordinates: {block.h_idx.tolist()}")
    print(f"probe colors: {block.n_colors}; bandwidth: {block.bandwidth}")
    print(f"conditional mean at eta=0.2: {mean[0].tolist()}")
    print(f"precision diagonal: {precision[0].diagonal().tolist()}")
    print(f"held-out factorization residual: {residual:.2e} nats")
    print("authorization: numerical; safe action: reparameterize or leave unchanged")


if __name__ == "__main__":
    main()
