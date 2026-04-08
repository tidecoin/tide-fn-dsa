# PQClean Security Hardening Plan

## Objective

- [x] Audit and harden the Rust PQClean/Tidecoin key-generation path in `tide-fn-dsa-kgen`.
- [x] Make the security posture explicit: zeroization, primitive usage, floating-point policy, and constant-time expectations.
- [x] Separate compatibility goals from hardening goals so PQClean parity work and security work do not get conflated.

## Current Security Assessment

### Confirmed Findings

- [x] The Rust PQClean path currently uses floating-point / FFT machinery.
  - `tide-fn-dsa-kgen/src/pqclean_float.rs`
  - `tide-fn-dsa-kgen/src/pqclean_compat.rs`
  - `tide-fn-dsa-kgen/src/pqclean_ntru.rs`
- [x] On mainstream targets, `FLR` is backed by native `f64`.
  - `tide-fn-dsa-sign/src/flr.rs`
  - `tide-fn-dsa-sign/src/flr_native.rs`
- [x] The top-level PQClean wrapper uses `Zeroizing` for fixed scratch buffers.
- [x] The deeper PQClean solver previously allocated multiple heap `Vec`s without explicit zeroization.
- [x] The current PQClean solver now wraps its secret-bearing heap scratch in `Zeroizing`.
- [x] The PQClean keygen path is not strict constant-time.
  - It includes rejection loops.
  - It includes secret-dependent early exits.
  - It still includes algorithm-level rejection behavior and secret-dependent acceptance decisions.
- [x] The PQClean path no longer uses `flr_to_f64()` conversions to drive branch decisions.
- [x] The PQClean path no longer relies on secret-dependent short-circuit boolean evaluation in the reviewed compatibility/solver checks.

### Consequences

- [x] The current PQClean Rust path does **not** satisfy a "no floating point" requirement.
- [x] The current PQClean Rust path does **not** satisfy a "strict constant-time keygen" requirement.
- [x] The current PQClean Rust path is closer to "compatibility-oriented Falcon keygen with some CT-conscious components" than to a hardened constant-time implementation.

## Scope

### In Scope

- [ ] `tide-fn-dsa-kgen/src/pqclean_compat.rs`
- [ ] `tide-fn-dsa-kgen/src/pqclean_ntru.rs`
- [ ] `tide-fn-dsa-kgen/src/pqclean_float.rs`
- [ ] shared FLR backend implications from:
  - `tide-fn-dsa-sign/src/flr.rs`
  - `tide-fn-dsa-sign/src/flr_native.rs`
  - `tide-fn-dsa-sign/src/flr_emu.rs`

### Out of Scope

- [ ] Parity correctness against PQClean, except where security fixes could break parity and require revalidation.
- [ ] Classic `fn-dsa` path changes unrelated to the PQClean/Tidecoin solver.

## Hardening Tracks

### Track 1: Zeroization

- [x] Inventory every secret-bearing heap allocation in `pqclean_ntru.rs`.
- [x] Replace plain `Vec` usage with zeroizing equivalents where practical.
- [x] Ensure all secret polynomial, CRT, and FLR-domain temporaries are scrubbed on all exits.
- [x] Review failure paths and early returns to ensure scrubbing still happens.

Priority targets:

- [x] `Fd` / `Gd`
- [x] `FGt`, `Ft`, `Gt`
- [x] `fx`, `gx`, `Fp`, `Gp`
- [x] `fg_plain`
- [x] `cap_f`, `cap_g`
- [x] `flr_f`, `flr_g`
- [x] `num`, `den`
- [x] cloned FFT-domain temporaries like `kf`, `kg`

### Track 2: Floating-Point Policy

- [x] Decide the actual security requirement for PQClean/Tidecoin keygen:
  - [x] "floating point allowed if parity-compatible"
  - [ ] "must avoid native floating point"
  - [ ] "must avoid all floating point entirely"
- [x] Decide whether native floating point is acceptable for this path.
  - [x] accepted because it matches the project's classic FN-DSA signing posture
  - [ ] force emulated FLR backend for PQClean path
  - [ ] design a separate integer-only solver
- [x] Document the chosen policy in crate docs and implementation notes.

Current fact:

- [x] The implementation currently depends on FLR/FFT operations and native `f64` on common architectures.
- [x] Native floating point is currently accepted for the PQClean key-generation path because that matches the project's classic signing posture.

### Track 3: Constant-Time Posture

- [x] Enumerate secret-dependent branches in the PQClean path.
- [x] Separate unavoidable Falcon keygen rejection behavior from avoidable data-dependent branches.
- [x] Reduce avoidable secret-dependent float checks where possible.
- [x] Remove reviewed secret-dependent short-circuit boolean evaluation where practical.
- [x] Document clearly what constant-time guarantees are and are not being claimed.

Current fact:

- [x] Strict constant-time behavior is not currently achieved.
- [x] The remaining variable-time behavior is primarily the Falcon/PQClean algorithm structure itself, not incidental `f64` branch conversions in the reviewed PQClean path.

### Track 4: Primitive Review

- [ ] Reconfirm that the seeded expansion and Gaussian sampler match intended PQClean/Falcon behavior.
- [ ] Reconfirm use of `SHAKE256`, CDT table, modular arithmetic, and encoding logic.
- [ ] Verify no accidental primitive substitutions were introduced during porting.

Current fact:

- [x] Primitive choices appear aligned with the intended PQClean/Falcon design.

### Track 5: Contract Clarity

- [x] Document `keygen_from_seed_pqclean()` as a compatibility-oriented deterministic seeded API.
- [x] Document `keygen_from_stream_key_tidecoin()` as a Tidecoin retry-schedule wrapper over the PQClean-compatible seeded API.
- [x] Document that these APIs preserve the original seeded mapping and rejection behavior, and do not imply a separate PQClean signing or verification runtime.

### Track 6: Parity Coverage

- [x] Keep explicit local Rust-vs-oracle parity tests behind `feature = "pqclean-ref"`.
- [x] Cover Falcon-512 and Falcon-1024 deterministic seeded keygen against the local oracle.
- [x] Cover Falcon-1024 `depth1_input`, `depth1_prebabai`, and `depth1_components` checkpoints.
- [x] Keep the parity suite as a gated developer check, not a mandatory default CI path.

## Suggested Order

1. [x] Fix zeroization gaps first.
2. [x] Decide and document the floating-point policy.
3. [x] Based on that decision, accept the current FLR model for the PQClean path and document its limits.
4. [ ] Tighten constant-time posture where practical without breaking parity.
5. [x] Re-run parity tests after every meaningful change.

Status:

- [x] The low-risk constant-time tightening pass is complete.
- [ ] Any further constant-time tightening would require deeper algorithm-structure changes and is optional.

## Validation

- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo check --workspace`
- [x] `cargo check -p tide-fn-dsa-kgen --tests --features pqclean-ref`
- [x] `cargo test -p tide-fn-dsa-kgen --lib pqclean_parity --features pqclean-ref -- --nocapture`
- [x] add targeted tests or review notes for zeroization-sensitive code paths where useful

Validated tightening details:

- [x] PQClean branch comparisons no longer depend on `flr_to_f64()`
- [x] reviewed short-circuit boolean checks in the PQClean path were rewritten to unconditional evaluation plus a single combined branch

## Exit Criteria

- [x] All known secret-bearing PQClean solver temporaries are explicitly scrubbed or intentionally justified.
- [x] Floating-point policy is explicit and reflected in code/docs.
- [x] Constant-time claims are accurate and not overstated.
- [x] PQClean parity still passes after hardening changes.
- [x] The repository has a documented, reviewable security position for the PQClean/Tidecoin Rust path.
- [x] The required hardening track for the current accepted security posture is complete.
