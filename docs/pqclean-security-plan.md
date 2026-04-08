# PQClean Security Hardening and Integer-Only Migration Plan

## Objective

- [x] Audit and harden the Rust PQClean/Tidecoin key-generation path in `tide-fn-dsa-kgen`.
- [x] Make the security posture explicit: zeroization, primitive usage, floating-point policy, and constant-time expectations.
- [x] Separate compatibility goals from hardening goals so PQClean parity work and security work do not get conflated.
- [x] Remove hardware floating-point execution from the PQClean/Tidecoin key-generation path while preserving byte-exact PQClean parity.

## Current Security Assessment

### Confirmed Findings

- [x] The Rust PQClean path uses Falcon-style FLR / FFT machinery.
  - `tide-fn-dsa-kgen/src/pqclean_float.rs`
  - `tide-fn-dsa-kgen/src/pqclean_compat.rs`
  - `tide-fn-dsa-kgen/src/pqclean_ntru.rs`
- [x] The PQClean path now routes those operations through a PQClean-local integer-only numeric layer:
  - `tide-fn-dsa-kgen/src/pqclean_flr.rs`
  - `tide-fn-dsa-kgen/src/pqclean_flr_emu.rs`
  - `tide-fn-dsa-kgen/src/pqclean_poly.rs`
- [x] The PQClean production path no longer imports the sign-crate numeric backend by path.
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

- [x] The current PQClean Rust path does satisfy the agreed "no hardware floating point" requirement.
- [x] The current PQClean Rust path does **not** satisfy a "strict constant-time keygen" requirement.
- [x] The current PQClean Rust path is an integer-only, compatibility-oriented Falcon keygen with some CT-conscious components, not a hardened constant-time implementation.

### New Implementation Constraint

- [x] Classic/native FN-DSA key generation already uses integer/fixed-point arithmetic.
- [x] The PQClean/Tidecoin compatibility key-generation path must also stop relying on hardware floating point.
- [x] Exact PQClean parity remains mandatory.
- [x] This does **not** imply that the PQClean path should be rewritten onto the classic `FXR` engine.
- [x] The migration target is an integer-only implementation of the Falcon/PQClean FLR semantics, not a semantic redesign of the PQClean solver.
- [x] The PQClean path should have an explicit local numeric layer so native-FP behavior cannot leak back in indirectly through shared imports.

Clarification:

- [x] Classic `FXR` and Falcon/PQClean `FLR` are different numeric models.
- [x] The current migration candidate is the existing integer-only emulated `FLR` backend in `tide-fn-dsa-sign/src/flr_emu.rs`.
- [x] `flr_emu` is being considered because it already targets the project's FLR semantics; it is **not** assumed to be PQClean-compatible until parity is revalidated.
- [x] Therefore, the safe plan is:
  - [x] use `flr_emu` as the first backend candidate
  - [x] validate it only against the local PQClean oracle/parity suite
  - [x] fork or specialize it for PQClean only if parity shows that direct reuse is insufficient

## Scope

### In Scope

- [ ] `tide-fn-dsa-kgen/src/pqclean_compat.rs`
- [ ] `tide-fn-dsa-kgen/src/pqclean_ntru.rs`
- [x] `tide-fn-dsa-kgen/src/pqclean_float.rs`
- [x] `tide-fn-dsa-kgen/src/pqclean_flr.rs`
- [x] `tide-fn-dsa-kgen/src/pqclean_flr_emu.rs`
- [x] `tide-fn-dsa-kgen/src/pqclean_poly.rs`
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
  - [ ] "floating point allowed if parity-compatible"
  - [x] "must avoid native floating point"
  - [ ] "must avoid all floating point entirely"
- [x] Decide whether native floating point is acceptable for this path.
  - [ ] accepted because it matches the project's classic FN-DSA signing posture
  - [x] remove hardware floating-point execution from the PQClean path
  - [x] preserve the Falcon/PQClean FLR semantics with an integer-only implementation
- [x] Document the chosen policy in crate docs and implementation notes.

Current fact:

- [x] The implementation still depends on FLR/FFT operations.
- [x] Hardware `f64` execution has now been removed from the PQClean/Tidecoin compatibility path.

### Track 2A: Integer-Only FLR Migration

- [x] Define the migration target precisely:
  - [x] no hardware `f64` execution in the PQClean key-generation path
  - [x] keep exact PQClean/Falcon FLR rounding semantics
  - [x] do not alter public deterministic mapping
- [x] Isolate the PQClean numeric backend from the sign-crate native FLR backend.
- [ ] Decide the implementation shape:
  - [ ] force the existing emulated FLR backend for PQClean code paths
  - [x] isolate a PQClean-local integer-only FLR layer derived from the existing emulated backend
- [x] Keep the classic/native FN-DSA path unchanged.
- [x] Keep the PQClean parity harness green after every migration step.

Implementation note:

- [x] Reusing classic `FXR` directly is **not** the default plan.
- [x] The safe plan is to preserve Falcon/PQClean FLR semantics with an integer-only backend, because parity is mandatory.
- [x] The implementation base should still be `flr_emu`.
- [x] The intended shape is now a PQClean-local integer-only numeric layer derived from that backend.
- [x] This stronger isolation is preferred because it avoids accidental fallback to native-FP behavior through shared sign-side imports.

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
- [x] Reconfirm that the integer-only FLR migration does not change the effective Falcon arithmetic semantics.

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
2. [x] Decide and document that native floating point is no longer acceptable for the PQClean path.
3. [x] Isolate the PQClean numeric backend so it no longer depends on hardware-FP execution.
4. [x] Migrate PQClean FLR usage to an integer-only backend while preserving exact FLR semantics.
5. [x] Re-run the gated local parity suite after each migration step.
6. [x] Reconfirm deterministic keygen parity for Falcon-512 and Falcon-1024 end-to-end.
7. [x] Reconfirm that the classic/native FN-DSA key-generation path is untouched.
8. [ ] Tighten constant-time posture further only if still useful after the integer-only migration.

Status:

- [x] The low-risk constant-time tightening pass is complete.
- [x] The integer-only migration is complete.
- [ ] Any further constant-time tightening beyond that migration remains optional.

## Concrete Implementation Phases

### Phase 1: Backend Split

- [x] Remove the assumption that PQClean keygen should share the production sign-side native FLR backend.
- [x] Introduce an explicit PQClean FLR abstraction boundary in `tide-fn-dsa-kgen`.
- [x] Make it impossible for the PQClean path to silently route into native `f64` through shared imports.
- [x] Stop importing the sign-side numeric layer by path for the PQClean production path.

### Phase 2: Integer-Only FLR Wiring

- [x] Create a PQClean-local integer-only FLR layer from the existing emulated backend.
- [x] Route PQClean FLR operations through that local layer.
- [x] Keep any adjustments minimal and parity-driven.
- [x] Keep existing FLR operation ordering intact.
- [x] Preserve FFT helper behavior and rounding points exactly.

### Phase 3: Solver Revalidation

- [x] Revalidate Falcon-1024 depth-1 checkpoints:
  - [x] `depth1_input`
  - [x] `depth1_prebabai`
  - [x] `depth1_components`
- [x] Revalidate deterministic seeded keygen parity:
  - [x] Falcon-512
  - [x] Falcon-1024

### Phase 4: Contract and Documentation

- [x] Update user-facing wording so PQClean/Tidecoin deterministic keygen is described as integer-only in implementation, but still compatibility-oriented in behavior.
- [x] Ensure README claims about integer-only key generation distinguish classic/native keygen from PQClean/Tidecoin keygen until the migration is complete.

## Validation

- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo check --workspace`
- [x] `cargo check -p tide-fn-dsa-kgen --tests --features pqclean-ref`
- [x] `cargo test -p tide-fn-dsa-kgen --lib pqclean_parity --features pqclean-ref -- --nocapture`
- [x] add targeted tests or review notes for zeroization-sensitive code paths where useful

Validated tightening details:

- [x] PQClean branch comparisons no longer depend on `flr_to_f64()`
- [x] reviewed short-circuit boolean checks in the PQClean path were rewritten to unconditional evaluation plus a single combined branch

Required migration validation:

- [x] `cargo check --workspace`
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo check -p tide-fn-dsa-kgen --tests --features pqclean-ref`
- [x] `cargo test -p tide-fn-dsa-kgen --lib pqclean_parity --features pqclean-ref -- --nocapture`
- [x] targeted deterministic PQClean/Tidecoin keygen tests for Falcon-512 and Falcon-1024

## Exit Criteria

- [x] All known secret-bearing PQClean solver temporaries are explicitly scrubbed or intentionally justified.
- [x] Floating-point policy is explicit and reflected in code/docs.
- [x] Constant-time claims are accurate and not overstated.
- [x] PQClean parity still passes after hardening changes.
- [x] The repository has a documented, reviewable security position for the PQClean/Tidecoin Rust path.
- [x] The PQClean/Tidecoin compatibility key-generation path no longer executes hardware floating-point operations.
- [x] Exact PQClean parity still holds after the integer-only migration.
- [x] The required hardening track for the current accepted security posture is complete.
