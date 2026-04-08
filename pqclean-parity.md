# PQClean Parity Plan

## Objective

- [ ] Make the Rust PQClean/Tidecoin solver match PQClean exactly, bit for bit, for Falcon-512 and Falcon-1024.
- [ ] Preserve the original `fn-dsa` solver path for `fn-dsa` compatibility.
- [ ] Keep PQClean oracle code in the library test surface only; do not compile C in production.

## Current Facts

- [x] The imported PQClean C reference path is available inside this repository's test surface.
- [x] The reference Falcon-1024 deterministic seeded keygen path matches the Tidecoin vector.
- [x] The Rust Falcon-1024 PQClean-compatible path still diverges from PQClean.
- [x] The surviving mismatch is centered in the depth-1 specialized Falcon-1024 solver path.
- [x] At least one debug mismatch is a layout mismatch in the debug surface, not an arithmetic mismatch.
- [x] The surviving `depth1_prebabai` mismatch is a real value mismatch, not just transposition.
- [x] The current Rust implementation is still an adaptation of PQClean logic, not a line-faithful port in the critical depth-1 path.

## Confirmed Problem Statement

- [ ] `keygen_from_seed_pqclean()` must not claim PQClean/Tidecoin compatibility until Rust output matches PQClean exactly.
- [ ] Falcon-1024 currently fails that requirement for at least some valid seeds.
- [ ] The root issue is in the Rust solver implementation, not in the reference oracle anymore.

## Constraints

- [ ] No C compilation in release or production code paths.
- [ ] PQClean C may exist only as a test oracle and instrumentation surface in this repository.
- [ ] Do not rely on `rust-tidecoin` or node-side tests for core library validation.
- [ ] Do not collapse solver behaviors together; keep original `fn-dsa` compatibility separate from PQClean/Tidecoin compatibility.
- [ ] Avoid broad monkey patches; port PQClean rigorously and keep test coverage local to this repository.

## Scope Split

### Production Solver Surface

- [ ] Preserve original `fn-dsa` solver behavior for `fn-dsa` compatibility.
- [ ] Introduce and maintain a separate Rust PQClean/Tidecoin solver path for Falcon-512 and Falcon-1024.

### Test Oracle Surface

- [ ] Keep imported PQClean C wrappers under `tide-fn-dsa-kgen/tests/`.
- [ ] Expose narrow debug checkpoints from the C oracle:
  - [ ] `depth1_input`
  - [ ] `depth1_prebabai`
  - [ ] `depth1_components`
  - [ ] final deterministic seeded keygen
- [ ] Keep the oracle test-only and excluded from production builds.

## Phase 1: Stabilize the Oracle

- [ ] Verify all imported PQClean test wrappers use correct scratch sizes and no aliasing.
- [ ] Confirm the Falcon-1024 oracle path is stable under repeated runs.
- [ ] Keep one deterministic seeded vector test per scheme that proves the oracle path itself is correct.
- [ ] Remove or isolate any debug helper behavior that can corrupt the oracle signal.

## Phase 2: Normalize the Debug Surface

- [ ] Separate layout-parity checks from value-parity checks.
- [ ] Make debug exports explicit about memory layout:
  - [ ] coefficient-major
  - [ ] word-major
  - [ ] transposed views where needed
- [ ] Rewrite mismatching debug tests so they fail only on real semantic divergence.
- [ ] Document expected shape and indexing for each checkpoint buffer.

## Phase 3: Port Falcon-1024 Depth-1 Solver Literally

- [ ] Stop adapting the generic Rust NTRU code for the PQClean depth-1 path.
- [ ] Port PQClean's Falcon-1024 `solve_NTRU_binary_depth1` into Rust line by line.
- [ ] Preserve PQClean buffer layout, mutation order, and temporary storage behavior.
- [ ] Port helper routines used by the depth-1 path exactly where needed instead of reinterpreting them.
- [ ] Avoid "equivalent" refactors until parity is achieved.

## Phase 4: Validate Intermediate Parity

- [ ] Add direct parity tests at each critical checkpoint for Falcon-1024:
  - [ ] depth-1 input
  - [ ] pre-Babai `ft/gt`
  - [ ] post-Babai intermediate values
  - [ ] reconstructed outputs
- [ ] Run those tests against deterministic seeds that currently reproduce failure.
- [ ] Prove the first divergence point with an exact failing index before changing further logic.
- [ ] Repeat until there is no checkpoint divergence.

## Phase 5: Final Seeded Keygen Parity

- [ ] Add final deterministic seeded keygen tests for Falcon-1024 against the local PQClean oracle.
- [ ] Confirm byte-perfect parity for:
  - [ ] public key encoding
  - [ ] secret key encoding
  - [ ] hash/vector targets already known from Tidecoin
- [ ] Ensure the test count is intentionally small and deterministic so failures are fast to reproduce.

## Phase 6: Falcon-512 Review

- [ ] Audit Falcon-512 with the same checkpoint structure.
- [ ] Confirm whether Falcon-512 already matches exactly or only appears correct by luck.
- [ ] If Falcon-512 differs, repeat the same literal-port approach there.

## Phase 7: API and Naming Cleanup

- [ ] Keep the original solver exposed for `fn-dsa` compatibility.
- [ ] Keep PQClean/Tidecoin behavior behind explicit API naming that does not mislead users.
- [ ] Remove any claim of PQClean compatibility from paths that are not byte-faithful.
- [ ] Document which solver each public entrypoint uses.

## Exit Criteria

- [ ] Falcon-512 Rust PQClean path matches the local PQClean oracle bit for bit.
- [ ] Falcon-1024 Rust PQClean path matches the local PQClean oracle bit for bit.
- [ ] Deterministic seeded keygen parity tests pass quickly and reproducibly inside this repository.
- [ ] No production code depends on C.
- [ ] Original `fn-dsa` solver compatibility is preserved.
- [ ] PQClean/Tidecoin compatibility claims are true and test-backed.

## Known Risks

- [ ] Hidden layout mismatches can look like arithmetic mismatches unless debug surfaces are normalized first.
- [ ] Refactoring before parity is reached can erase evidence of the real divergence point.
- [ ] Reusing generic Rust helpers where PQClean has specialized depth-1 logic can reintroduce subtle divergence.
- [ ] Slow end-to-end tests can hide the fact that a single checkpoint test would expose the bug much faster.

## Implementation Rules

- [ ] Change one parity checkpoint at a time.
- [ ] Do not broaden the surface area while the first divergence point is still unknown.
- [ ] Keep tests local to `tide-fn-dsa-kgen`.
- [ ] Prefer exactness over elegance until parity is complete.
- [ ] After parity is achieved, do a cleanup pass without changing semantics.

## Immediate Next Steps

- [ ] Lock down the oracle and checkpoint tests as the authoritative baseline.
- [ ] Normalize the depth-1 debug surface so layout mismatches are no longer ambiguous.
- [ ] Re-port Falcon-1024 `solve_NTRU_binary_depth1` literally from PQClean into Rust.
- [ ] Re-run checkpoint parity tests until the first real divergence disappears.
- [ ] Re-run final deterministic seeded keygen parity tests for Falcon-1024.
