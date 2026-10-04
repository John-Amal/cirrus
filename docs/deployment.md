# Deployment

The trained model runs as a public HTTP service at
[cirrus-3s24.onrender.com/docs](https://cirrus-3s24.onrender.com/docs).

This note records how it gets there and why it is built the way it is. Most
of the design was forced by a 512 MB memory limit, and the constraints turned
out to be clarifying rather than annoying.

## What is served

The exported artifact is a **parameter predictor**, not a sampler: fields in,
the three parameters of a censored shifted gamma per grid cell out. The
head's draws come from `Gamma.rsample`, which is not a traceable graph
operation — but exporting a sampler would have been the wrong interface
regardless. A consumer that receives shape, scale and shift has the entire
predictive distribution and can take any quantile, probability or mean from
it. A consumer that receives 24 draws has whatever someone else decided 24
was good for.

Those derived quantities are computed **in closed form** by the service.
Training needed sample-based CRPS only because the incomplete gamma function
has no implemented derivative with respect to its shape parameter, so a
likelihood could not be backpropagated through. Inference has no gradients,
so the exact expressions are available. The sampling was a workaround for
autograd, not a property of the distribution.

The censored mean is the one worth stating, since it is not `shape * scale`:
everything the shift pushes below zero piles up at zero, so

```
E[Y] = shape * scale * SF(-shift; shape + 1, scale) + shift * SF(-shift; shape, scale)
```

All four closed forms were checked against Monte Carlo across three parameter
regimes before being trusted.

## Export

```bash
cirrus export --arm twcrps_p90 --onnx --quantise
```

The command verifies rather than claims: every format is run against PyTorch
on the same input and the maximum difference reported, with a non-zero exit
if anything diverges beyond 1e-4. A silent divergence in an export is the bug
that only appears in production.

| format | max difference | ms/batch | used |
| --- | --- | --- | --- |
| TorchScript | 0.00e+00 | 9.9 | locally |
| ONNX | — | — | in deployment |
| dynamic int8 | 2.1e-01 | 11.1 | **no** |

**Quantisation was measured and rejected.** It is both less accurate — 2.1e-01
against a shape parameter of order 0.02 — and *slower* than float32. At 5M
parameters over 128 tokens the matrix multiplications are too small for int8
to pay for its own overhead, and Apple's Accelerate already optimises float32
well. Quantisation is a technique for large models on constrained hardware;
this is a small model on capable hardware.

## Why ONNX in deployment

Hugging Face Spaces now requires a paid plan for Docker, so the service runs
on Render's free tier: 512 MB of RAM, spin-down after 15 minutes idle, 30–60
second cold starts.

PyTorch alone occupies roughly 300 MB resident before the model loads. That
is uncomfortably close to the limit, and an out-of-memory kill mid-request is
a worse demo than no demo.

The service was therefore made runtime-agnostic: a `Predictor` protocol with
TorchScript and ONNX implementations, chosen by file extension, with **torch
imported only inside the TorchScript one**. Everything else in the service is
numpy and scipy. Serving ONNX means torch never enters the process, taking
the image from about 1.5 GB to under 300 MB and resident memory well under
150 MB.

`/health` and `/info` report which runtime is live, so the claim is checkable
rather than assumed.

## The bundle

Model weights do not live in git. They are published to the Hub as a
**bundle**, because the model alone is not usable:

| file | why it is needed |
| --- | --- |
| `twcrps_p90.onnx` | the model |
| `twcrps_p90.pt` | TorchScript, for local runs |
| `normalisation.json` | the service normalises inputs with these |
| `thresholds.json` | exceedance probabilities are relative to these |
| `bundle.json` | channel order and provenance |

Shipping the model without the statistics would not fail — it would answer
confidently with numbers derived from the wrong normalisation. They travel
together for that reason.

```bash
python scripts/publish_serving_bundle.py --arm twcrps_p90 --push
```

## Deploying

```bash
docker build -t cirrus .                    # local, TorchScript, needs torch
docker run -p 8000:8000 cirrus
```

For Render: a Web Service with the Docker runtime and Dockerfile path
`./Dockerfile.render`. That image installs its dependencies explicitly with
`--no-deps`, because `pyproject.toml` lists torch as a core requirement —
correct for training, wrong for serving. It downloads the bundle at **build**
time rather than at startup: a container that starts and then fails to fetch
would report healthy and serve errors.

## What went wrong, and what it taught

Four failures, none of them in the model code.

**`.gitignore` excluded the package.** The rule `serve/` has no leading
slash, so git applied it at every depth and `src/cirrus/serve/` was never
committed. The install succeeded and the import failed. This was the *second*
occurrence of the same pattern — `data/` had excluded `src/cirrus/data/` in
Phase 1. **Anchor directory ignores with a leading slash**, and check
`git ls-files` for a new package rather than trusting `git status`.

**The Docker layer cache served a stale commit.** `pip install
git+https://...` is byte-identical between builds, so Docker reused a layer
built before the fix. The instruction text is not evidence about the remote's
contents. Clearing the build cache fixes it; pinning the install to a commit
prevents it.

**A "torch-free" image still needed xarray.** `api.py` imports `IngestSpec`
from `ingest.py`, which imports xarray at module level for the download
machinery the service never uses. Importing one name from a module imports
all of its dependencies. The expedient fix was to install xarray; the correct
one is to read the channel order from `bundle.json` and drop the dependency.

**Quantisation needed a backend that was not compiled in.** `NoQEngine` from
deep inside the conversion. Now detected up front and reported rather than
raised, since it is a property of the installation and not of the model.

The theme across all four, and across several CI failures earlier in the
project: **the environment that builds and the environment that develops
diverge in ways that local testing cannot reveal.** Local green is not green.
