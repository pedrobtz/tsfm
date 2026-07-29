## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## Test environments

* local: not applicable (see note below)
* GitHub Actions: ubuntu-latest (R release)

## Notes for the maintainer (remove before submitting)

Before the first CRAN submission, confirm the following in a full
`R CMD check --as-cran` on a machine with the Suggests installed:

* Downloads and Hub access happen only on first user call, never at check time.
  All tests that touch the Hub, torch weights, or the network use
  `skip_on_cran()` / `skip_if_not_installed()`; parity tests run off committed
  golden fixtures.
* `torch` and `safetensors` are declared in Imports for the target design but
  are exercised by the native architectures; if a native port has not yet
  landed at submission time, verify there is no "unused import" NOTE or move
  them to Suggests for that release.
* Examples that require a model download are wrapped in `\dontrun{}`.
* The package never redistributes model weights; each model's weight licence is
  surfaced via `tsfm_capabilities()`.
