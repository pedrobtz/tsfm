## R CMD check results

Local check on 2026-08-14:

0 errors | 0 warnings | 0 notes

Checked with R 4.6.1 on macOS 26.5.2 (aarch64-apple-darwin23) from the built
source package, with vignettes enabled and all declared Imports and Suggests
installed.

The same commit is green on GitHub Actions across five configurations:
macOS release, Windows release, and Ubuntu on R devel, release, and oldrel-1.

A second local check with `_R_CHECK_DEPENDS_ONLY_=true` also reports
0 errors | 0 warnings | 0 notes, confirming the package installs and its tests
run with only the declared Imports present.

## Test scope

* The deterministic suite is network-free and passes on every configuration
  above. Nothing in it downloads weights, contacts the Hub, or requires Python.
* `torch` is an Import, but the LibTorch runtime it needs is downloaded
  separately by `torch::install_torch()`. Tests that execute tensors skip when
  that runtime is absent, which is the situation on a CRAN check machine. Four
  of the five CI configurations run without it precisely to keep that path
  exercised; one installs it so the native code is not merely skipped.
* Numerical parity against the pinned TimesFM reference, and every test that
  loads the 925 MB checkpoint, are opt-in behind
  `TSFM_RUN_CHECKPOINT_TEST=true`. They were run locally against the Hub cache
  and pass: five golden fixtures on CPU within the recorded `atol`/`rtol`
  budget, contract conformance against the real handle, and the four documented
  user workflows end to end.
* Vignettes evaluate no chunks. Their outputs were produced by running the code
  against the real checkpoint and pasted in, so no vignette build downloads
  anything.

## Release status

This is a development version, not yet a CRAN submission. `Version` remains
`0.0.0.9000`.

TimesFM 2.5 is the first supported native architecture: it passes both release
gates, contract conformance and numerical parity against the pinned reference
implementation. TTM remains a registered scaffold whose forward pass aborts by
design, and Chronos-2 is unregistered and rejected before any network or tensor
work. Known limitations are listed in the README.
