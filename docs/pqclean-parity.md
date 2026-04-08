# PQClean Parity Status

## Objective

- [x] Keep the original `fn-dsa` solver path for classic `fn-dsa` compatibility.
- [x] Keep the PQClean/Tidecoin solver path separate from the classic path.
- [x] Keep PQClean C confined to the library test/reference surface only.
- [x] Achieve working Rust PQClean/Tidecoin deterministic keygen behavior for Falcon-512 and Falcon-1024.

## Current State

- [x] The imported PQClean C oracle is available locally under `tide-fn-dsa-kgen/tests/pqclean_ref/...`.
- [x] The Rust PQClean/Tidecoin path is wired through `keygen_from_seed_pqclean()` and `keygen_from_stream_key_tidecoin()`.
- [x] The Rust Falcon-1024 PQClean solver mismatch that originally motivated this plan has been fixed.
- [x] Falcon-512 PQClean deterministic keygen is working.
- [x] Falcon-1024 PQClean deterministic keygen is working.
- [x] Classic FN-DSA code has been cleaned back out of the PQClean work.
- [x] PQClean debug helpers have been moved out of the production solver file into a dedicated test-only module.

## What Was Fixed

- [x] The Falcon-1024 depth-1 PQClean solver path was corrected so Rust and the PQClean reference no longer diverge in the failing seeded path.
- [x] The bad scalar classic FN-DSA slowdown was fixed separately by restoring classic files and removing accidental workspace corruption from the classic solver path.
- [x] The imported PQClean oracle/debug layer was stabilized enough to use as a local reference surface.
- [x] The PQClean reference wrapper surface was reduced to the live API needed by the current parity module.
- [x] The fallback PQClean test shim (`SHAKE256x4`) was moved out of the crate root implementation surface.
- [x] The remaining PQClean `debug_*` helpers were moved from `tide-fn-dsa-kgen/src/pqclean_ntru.rs` to `tide-fn-dsa-kgen/src/pqclean_ntru_debug.rs`.

## Current Layout

### Production PQClean Surface

- [x] `tide-fn-dsa-kgen/src/pqclean_compat.rs`
- [x] `tide-fn-dsa-kgen/src/pqclean_float.rs`
- [x] `tide-fn-dsa-kgen/src/pqclean_ntru.rs`

### Test / Reference PQClean Surface

- [x] `tide-fn-dsa-kgen/src/pqclean_ntru_debug.rs`
- [x] `tide-fn-dsa-kgen/src/pqclean_ref.rs`
- [x] `tide-fn-dsa-kgen/src/pqclean_test_shims.rs`
- [x] `tide-fn-dsa-kgen/tests/pqclean_ref/...`
- [x] The active Rust/C wrapper contract is now limited to:
  - [x] final deterministic seeded keygen for Falcon-512
  - [x] final deterministic seeded keygen for Falcon-1024
  - [x] Falcon-1024 seeded `f/g/F` extraction
  - [x] Falcon-1024 `depth1_input`
  - [x] Falcon-1024 `depth1_prebabai`
  - [x] Falcon-1024 `depth1_components`

### Intentional Separation

- [x] No classic FN-DSA core file carries PQClean debug plumbing anymore.
- [x] `pqclean_ntru.rs` contains production solver logic only.
- [x] PQClean debug helpers are compiled only under `#[cfg(all(test, feature = "pqclean-ref"))]`.
- [x] PQClean C is not part of release or production builds.

## Test Coverage Present Today

- [x] Tidecoin stream-block derivation vectors pass.
- [x] Tidecoin deterministic keygen hash vectors pass.
- [x] Native and PQClean deterministic seeded keygen remain intentionally distinct.
- [x] The imported FLR self-test that was previously failing in `kgen` now passes under `pqclean-ref`.
- [x] Dedicated feature-gated PQClean parity tests now exist in `tide-fn-dsa-kgen/src/pqclean_parity_tests.rs`.
- [x] The gated parity module asserts Rust-vs-local-oracle parity for:
  - [x] Falcon-1024 `depth1_input`
  - [x] Falcon-1024 `depth1_prebabai`
  - [x] Falcon-1024 `depth1_components`
  - [x] Falcon-512 final deterministic seeded keygen
  - [x] Falcon-1024 final deterministic seeded keygen
- [x] `cargo check -p tide-fn-dsa-kgen --tests --features pqclean-ref` passes cleanly.

Concrete commands last verified:

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo check -p tide-fn-dsa-kgen --tests --features pqclean-ref
cargo test -p tide-fn-dsa-kgen --lib deterministic_stream_block_matches_tidecoin_pqhd_vectors -- --nocapture
cargo test -p tide-fn-dsa-kgen --lib deterministic_keygen_from_stream_matches_tidecoin_hash_vectors -- --nocapture
cargo test -p tide-fn-dsa-kgen --lib deterministic_native_and_pqclean_seeded_keygen_are_distinct -- --nocapture
cargo test -p tide-fn-dsa-kgen --lib pqclean_float::flr::tests::test_spec --features pqclean-ref -- --nocapture
cargo test -p tide-fn-dsa-kgen --lib pqclean_parity --features pqclean-ref -- --nocapture
```

## Important Change In Status

- [x] This is no longer a "find the first divergence" incident plan.
- [x] The main parity bug that originally broke Falcon-1024 has already been fixed.
- [x] The remaining work is maintenance and hardening, not emergency solver repair.

## What Is Still Worth Doing

- [x] Explicit Rust-vs-oracle parity assertions now live in a dedicated feature-gated test module.
- [x] Those parity checks live in the right place: they only compile and run under `feature = "pqclean-ref"`.
- [x] The unused PQClean reference wrapper entrypoints were removed from `pqclean_ref.rs` and the local C oracle wrappers.
- [x] The intended public contract for `keygen_from_seed_pqclean()` and `keygen_from_stream_key_tidecoin()` is now documented directly in crate-level and method-level docs.

## Risks That Still Exist

- [x] The imported PQClean sources were pruned down to the three retained Falcon-1024 depth-1 debug helpers still needed by the active parity module.
- [x] Some deep parity helpers now live only in the dedicated test module and are intentionally not exercised by default test runs.
- [x] The detailed checkpoint machinery is now asserted automatically when `pqclean-ref` parity tests are run.

## Bottom Line

- [x] Rust PQClean/Tidecoin keygen is working for Falcon-512 and Falcon-1024.
- [x] Classic FN-DSA compatibility has been separated back out.
- [x] PQClean C remains test-only.
- [x] PQClean debug machinery is isolated to PQClean test/reference files.
- [x] Stronger local guarantees now come from the dedicated `pqclean-ref` parity test module, not from more solver surgery.
