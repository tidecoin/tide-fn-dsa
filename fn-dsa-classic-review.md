# FN-DSA Classic Review and Progress

Reference baseline before PQClean porting began:

- `89ef9ada2aaa3fb74e9157b055c8c582c0afe2ac`

Commit originally reviewed:

- `9a66ccbd9b50ed38492311679c994456b59954ad` (`pqclean port - phase 1`)

## Current Status

The classic FN-DSA surface has now been rolled back and isolated from the PQClean port.

What is true now relative to baseline `89ef9ad`:

- all classic `tide-fn-dsa-kgen` core files are back to baseline
- all classic `tide-fn-dsa-sign` shared files are back to baseline
- the only classic-path file intentionally still different is `tide-fn-dsa-sign/src/flr_native.rs`
- the remaining non-baseline code is confined to PQClean-specific modules and small bridge wiring in `tide-fn-dsa-kgen/src/lib.rs`

## Rolled Back

These classic files were restored to the pre-port baseline shape:

- `tide-fn-dsa-kgen/src/fxp.rs`
- `tide-fn-dsa-kgen/src/gauss.rs`
- `tide-fn-dsa-kgen/src/mp31.rs`
- `tide-fn-dsa-kgen/src/ntru.rs`
- `tide-fn-dsa-kgen/src/ntru_avx2.rs`
- `tide-fn-dsa-kgen/src/poly.rs`
- `tide-fn-dsa-kgen/src/poly_avx2.rs`
- `tide-fn-dsa-kgen/src/vect.rs`
- `tide-fn-dsa-kgen/src/vect_avx2.rs`
- `tide-fn-dsa-kgen/src/zint31.rs`
- `tide-fn-dsa-kgen/src/zint31_avx2.rs`
- `tide-fn-dsa-sign/src/flr.rs`
- `tide-fn-dsa-sign/src/flr_emu.rs`
- `tide-fn-dsa-sign/src/poly.rs`

Verification against baseline showed these no longer differ.

## Intentionally Kept

### `tide-fn-dsa-sign/src/flr_native.rs`

This file remains intentionally different from baseline.

Reason:

- keep the `div_emu` / `sqrt_emu` architecture gating fix
- `div_emu` and `sqrt_emu` now only activate on `riscv64`
- this was an independent correctness / feature-shape fix, not PQClean debug spillover

### `tide-fn-dsa-kgen/src/lib.rs`

This file remains different from baseline only for the minimal PQClean bridge:

- PQClean modules are declared at [lib.rs:119](/home/yaroslav/dev/tidecoin/tide-fn-dsa/tide-fn-dsa-kgen/src/lib.rs:119)
- the test-only `flr` alias is declared at [lib.rs:130](/home/yaroslav/dev/tidecoin/tide-fn-dsa/tide-fn-dsa-kgen/src/lib.rs:130)
- the root test module only re-exports `SHAKE256x4`; the shim implementation itself was moved out to [pqclean_test_shims.rs](/home/yaroslav/dev/tidecoin/tide-fn-dsa/tide-fn-dsa-kgen/src/pqclean_test_shims.rs)

Important distinction:

- the earlier crate-root PQClean debug API was removed
- the earlier classic-solver debug entry point in `ntru.rs` was removed by restoring baseline
- the large PQClean parity-debug bulk was removed from the classic crate root

## PQClean Isolation Work Completed

The remaining PQClean code is now isolated to:

- `tide-fn-dsa-kgen/src/pqclean_compat.rs`
- `tide-fn-dsa-kgen/src/pqclean_float.rs`
- `tide-fn-dsa-kgen/src/pqclean_ntru.rs`
- `tide-fn-dsa-kgen/src/pqclean_ntru_debug.rs`
- `tide-fn-dsa-kgen/src/pqclean_ref.rs`
- `tide-fn-dsa-kgen/tests/pqclean_ref/...`
- `tide-fn-dsa-kgen/build.rs`

Cleanup done in that layer:

- removed PQClean candidate-tracing and candidate-debug artifacts from `pqclean_compat.rs`
- removed production dependence on debug-stage solver dispatch in `pqclean_ntru.rs`
- moved PQClean parity-debug helpers out of `pqclean_ntru.rs` into the dedicated test-only module `pqclean_ntru_debug.rs`
- `pqclean_ntru.rs` now contains production solver code only; the debug helper module is included only under `#[cfg(all(test, feature = "pqclean-ref"))]`
- isolated sign-path imports used by PQClean float code so default workspace clippy is clean
- localized the PQClean reference debug type inside `pqclean_ref.rs` instead of exposing it through the classic crate root
- marked the remaining PQClean reference/debug files as explicitly test/reference-only so `clippy` and test-target checks stay clean without leaking that surface into classic code

## Clippy Status

Current command:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Current result:

- passes cleanly

This is a real improvement over the earlier state where `clippy` was failing with dozens of errors caused by:

- crate-root PQClean debug plumbing
- unused parity helpers
- imported sign internals leaking into the wrong lint surface

## Verification

Commands run successfully after the rollback and isolation:

```bash
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo check -p tide-fn-dsa-kgen --tests --features pqclean-ref
cargo test -p tide-fn-dsa-kgen --lib test_keygen_ref -- --nocapture
cargo test -p tide-fn-dsa-kgen --lib test_keygen_self -- --nocapture
```

Observed classic test timings:

- `test_keygen_ref`: test body finished in `0.11s`
- `test_keygen_self`: test body finished in `0.09s`

This confirms the classic FN-DSA path is no longer stuck behind the earlier PQClean-plumbing contamination.

## Remaining Diff vs Baseline

After the rollback, the intended non-baseline files are:

- `Cargo.toml`
- `README.md`
- `pqclean-parity.md`
- `tide-fn-dsa-kgen/Cargo.toml`
- `tide-fn-dsa-kgen/build.rs`
- `tide-fn-dsa-kgen/src/lib.rs`
- `tide-fn-dsa-kgen/src/pqclean_compat.rs`
- `tide-fn-dsa-kgen/src/pqclean_float.rs`
- `tide-fn-dsa-kgen/src/pqclean_ntru.rs`
- `tide-fn-dsa-kgen/src/pqclean_ntru_debug.rs`
- `tide-fn-dsa-kgen/src/pqclean_ref.rs`
- `tide-fn-dsa-kgen/tests/pqclean_ref/...`
- `tide-fn-dsa-sign/src/flr_native.rs`

That is the correct remaining shape:

- PQClean lives in PQClean files
- PQClean parity/debug helpers live in test-only PQClean files
- classic FN-DSA core files are no longer carrying PQClean debugging or loose plumbing

## Open Follow-Up

The remaining `lib.rs` test-surface dependency is limited to a re-export only. The fallback `SHAKE256x4` implementation now lives in the dedicated PQClean test utility at [pqclean_test_shims.rs](/home/yaroslav/dev/tidecoin/tide-fn-dsa/tide-fn-dsa-kgen/src/pqclean_test_shims.rs).

## Bottom Line

The original concern was correct: PQClean porting had spilled into the classic FN-DSA surface.

That spillover has now been rolled back.

Current status is:

- classic FN-DSA core restored
- PQClean logic isolated
- workspace clippy clean
- classic keygen tests fast and passing
- remaining `SHAKE256x4` shim moved out of `lib.rs` implementation scope
