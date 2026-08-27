# CONGA

This is the initial release for **CONGA: Compiling Gaussian Fiber Geometry**.
It contains the compiler itself and the three experiments in Section 4.2:
graph-indexed Gaussian fields, the Stock--Watson stochastic-volatility model,
and the electricity-load dynamic linear model.

CONGA looks for a block of latent variables whose conditional density is
Gaussian. Once that structure has been established, it can collapse the block,
construct an exact conditional transport, and run inference on the smaller
non-Gaussian core. The distinction between *finding* a plausible block and
*authorizing* a transformation matters here: a numerical probe is useful
evidence, but it does not turn an approximate identity into an exact one. The
code consequently reports its residual checks and can leave the target alone
when a backend is not justified.

## A two-minute start

We use [uv](https://docs.astral.sh/uv/) to make the environment reproducible.
Python 3.12 is required.

```bash
uv sync --extra dev
uv run python examples/gaussian_fiber.py
uv run pytest -q
```

The example has one nonlinear scale coordinate and four conditionally Gaussian
coordinates. It asks the discovery front-end for a candidate fiber, builds its
conditional precision, and checks the resulting factorization on a held-out
point. A typical ending is:

```text
discovered fiber coordinates: [1, 2, 3, 4]
core coordinates: [0]
probe colors: 1; bandwidth: 0
held-out factorization residual: 0.00e+00 nats
authorization: numerical; safe action: reparameterize or leave unchanged
```

The main entry points live in `conga.structure`. For a target implementing
`log_prob`, the compact version of the same pass is:

```python
found = conga.discover_gaussian_block(target, seed=3)
block = conga.GaussianBlock(target, found.g_idx,
                            probe_etas=found.probe_etas)
value, gradient = block.collapsed_vg(eta)
```

`GaussianBlock` provides the dense and banded paths. `SparseGaussianBlock`
lowers a recovered general-graph precision to sparse Cholesky, while
`GMRFSpectralBackend` implements the shared-spectrum Brenier lowering for paths
and rectangular lattices. `conga.inference.Marginalized` is the small reference
sampler that runs on the collapsed core and reconstructs the Gaussian fiber.

## Reading the Section 4.2 results

The submitted outputs are under `results/paper`; the corresponding figures are
under `results/figures`. This command prints the main comparisons without
installing the heavier experiment dependencies:

```bash
uv run python -m experiments.summarize
```

The artifact map is deliberately direct:

| Paper case study | Driver | Locked output |
| --- | --- | --- |
| Graph-indexed fields | `experiments.graph_gmrf`, `.graph_gmrf_nuts`, `.graph_gmrf_flow` | `graph_gmrf_*.json` |
| Stock--Watson | `experiments.stock_watson` | `stock_watson.json` |
| Electricity-load DLM | `experiments.load_dlm` | `load_dlm_*.json` |

The locked files include seeds and protocol settings. Timings will naturally
move with hardware and library versions, so the useful reproduction targets
are the structural certificates, the qualitative sampler comparison, and the
failure/routing decisions rather than bitwise-identical wall-clock numbers.

## Short runs

Install the full experiment environment once:

```bash
uv sync --extra experiments --extra dev
```

Each driver has a `--quick` path. These are integration checks, not miniature
versions of the statistical study: twenty or forty retained draws are enough
to catch a broken transformation, but not enough to compare ESS or convergence.

```bash
# Discovery, sparse factorization, and graph routing
uv run python -m experiments.graph_gmrf --quick --repeats 2

# Collapsed sparse NUTS on an easy tree and the adversarial star
uv run python -m experiments.graph_gmrf_nuts --quick

# Learned flow, reusing the locked quadrature reference
uv run python -m experiments.graph_gmrf_flow --quick \
  --reference-source results/paper/graph_gmrf_nuts.json

# Stock--Watson contract plus short raw/collapsed NUTS runs
uv run python -m experiments.stock_watson \
  --quick --arms NUTS-raw NUTS-CONGA

# DLM contract and a short matched raw/collapsed comparison
uv run python -m experiments.load_dlm --mode check --quick
uv run python -m experiments.load_dlm --mode sample --quick
```

The WALNUTS arms can be added to the Stock--Watson command with
`WALNUTS-raw WALNUTS-CONGA`. They use the released `walnutpie` package through
the thin adapter in `experiments/walnuts.py`.

Fresh outputs are written to `results/runs`, separately from the locked paper
artifacts. Long sampling drivers checkpoint after every completed arm or cell;
pass `--resume` to continue an interrupted run.

## Full paper protocol

The defaults of the Stock--Watson and graph drivers are their paper settings.
The DLM sampling command spells out the third seed because its conservative CLI
default runs only two. The learned-flow experiment reads the quadrature
references written by the graph NUTS experiment, so run it afterwards.

```bash
uv run python -m experiments.graph_gmrf
uv run python -m experiments.graph_gmrf_nuts --resume
uv run python -m experiments.graph_gmrf_flow \
  --families tree lattice geometric expander star --resume
uv run python -m experiments.graph_gmrf_nuts --plot-only

uv run python -m experiments.stock_watson --resume

uv run python -m experiments.load_dlm --mode check
uv run python -m experiments.load_dlm \
  --mode sample --seeds 0 1 2 --resume
uv run python -m experiments.load_dlm --mode full
```

These are real benchmark runs. Stock--Watson uses 8 chains for four sampler
arms over three seeds. The full load model builds a 14-region, 8,760-hour state
space and is principally an exact-collapse accounting run; it is much larger
than the matched sampling case. A many-core machine with ample memory is the
right place for the complete suite.

## Repository guide

- `src/conga/`: discovery, validation, transport backends, and inference.
- `examples/`: a small model that exposes the compiler contract.
- `experiments/`: self-contained Section 4.2 models and drivers.
- `data/`: the Stock--Watson series and its source provenance.
- `results/paper/`: immutable JSON outputs used in the paper.
- `results/figures/`: submitted figures generated from those outputs.
- `tests/`: exactness and integration checks for dense, sparse, and spectral
  paths.

The Stock--Watson JSON records the original data and model URLs, commit-pinned
in its `_provenance` field. The graph and load experiments generate their
synthetic observations from the seeds recorded by their drivers.

This release intentionally omits the broader PosteriorDB benchmark harness and
the experiments outside Section 4.2. Nothing in `src/conga` imports that
infrastructure.

## License

The release is available under the MIT license. The optional samplers and
scientific Python libraries are installed as dependencies and retain their own
licenses.
