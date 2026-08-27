"""CONGA: compile conditionally Gaussian geometry before inference."""
from .core import DTYPE, TargetPosterior
from .precond import StructureMetric, fit_structure_metric
from .sparse import SparsePrecisionBackend, SparsePrecisionFactor
from .spectral import GMRFSparseCholeskyBackend, GMRFSpectralBackend
from .structure import (
    GaussianBlock,
    SparseGaussianBlock,
    WhitenedTarget,
    discover_gaussian_block,
    rb_joint_covariance,
    whitening_cholesky,
)

__all__ = [
    "DTYPE",
    "TargetPosterior",
    "discover_gaussian_block",
    "GaussianBlock",
    "SparseGaussianBlock",
    "WhitenedTarget",
    "SparsePrecisionBackend",
    "SparsePrecisionFactor",
    "GMRFSpectralBackend",
    "GMRFSparseCholeskyBackend",
    "rb_joint_covariance",
    "whitening_cholesky",
    "fit_structure_metric",
    "StructureMetric",
]
