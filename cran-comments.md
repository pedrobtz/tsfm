## R CMD check results

Local check on 2026-08-13:

0 errors | 0 warnings | 0 notes

The package was checked with R 4.5.2 on macOS Sequoia 15.7.7 using
the built source package, with vignettes enabled and all declared Imports and
Suggests installed.

## Test scope

* The deterministic, network-free test suite passed.
* Native TimesFM parity skipped because its golden fixtures now exist but the
  full native forward implementation is Stage 3 work. TTM parity skipped
  because that architecture remains deferred. Neither is claimed as supported.
* The native TimesFM operator-feasibility fixture ran on CPU and matched the
  pinned official reference output; this is not an end-to-end model parity
  claim.
* The opt-in 925 MB checkpoint test was run separately against the local Hub
  cache and loaded all 232 tensors. Normal checks use synthetic safetensors and
  never access the network.
* The stub exercised plain-R forecasting, the hardhat fit/predict bridge,
  parsnip `fit()`/`predict()`, and fable conversion.

## Release status

This is a development baseline, not a CRAN submission candidate. No real
foundation model is supported yet. TimesFM 2.5 must pass both contract
conformance and pinned numerical parity before `0.1.0`.
