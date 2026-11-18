# TLSNotary benchmark for Artemis

Implements `zkml-commitments-paper/tlsn-eval-plan.md`: a TLSNotary-style predicate proof over a committed
response of `n` bytes, where the commitment is either recomputed in the circuit (SHA-256 as TLSNotary
ships it, or Poseidon2) or linked with Artemis (Horner evaluation in the circuit, one opening outside).

```
circuits/               Noir packages (one source tree for both backends), N is set per run
pcs/                    external commitment: Pedersen + zero-knowledge IPA over BN254 G1, hiding KZG
provekit-artemis.patch  the Artemis hook for worldfnd/provekit @ 893f6d58 (applied by ./run.sh setup)
provekit/               patched ProveKit checkout (not tracked)
bench.py, run.sh        sweep driver, results as CSV
plot.py                 plot + tables (results/summary.md)
results/                results.csv, logs, sweep_prover.{pdf,png}, summary.md
```

## Quick start

```bash
./run.sh setup     # clone + patch + build ProveKit (~10 min), build pcs
./run.sh test      # 9 soundness tests of the link, 4 tests of the pcs crate
./run.sh small     # age check + sweep 520 B .. 64 KiB, all rows; peak ~20 GiB RAM
./run.sh plot
```

Needs `nargo` and `bb` on `PATH` for the Barretenberg rows (`NARGO=`, `BB=` override), Python 3 with
matplotlib, and the Rust toolchain pinned by ProveKit (`nightly-2026-03-04`, picked up from
`rust-toolchain.toml`).

On AWS: `doe-suite-config/designs/tlsn.yml` (suite `tlsn`, role `setup-tlsn`, host type `tlsn_small` =
r6i.8xlarge). The instances clone the `tlsn-bench`
branch; each job is one `bench.py ... --out results/results.csv --clean-build` call for one backend and size.

## Backends and rows

| Row | Circuit | ProveKit | Barretenberg |
|---|---|---|---|
| No commitment | `nocom` | yes | yes |
| SHA-256 | `sha256` (`sha256_var` over the bytes) | yes | yes |
| Keccak-256 | `keccak256`: explicit Noir code (ProveKit's `csp-benchmarks/keccak_lib`, u32 split lanes); `keccak256_bb`: `noir-lang/keccak256` over the `keccakf1600` black box | `keccak256` (no Keccak black box), up to 16 KiB | both, the plot takes the faster one (`keccak256` up to 16 KiB) |
| BLAKE3 | `blake3`: explicit Noir code (`circuits/vendor/blake3_noir`, written for this benchmark); `blake3_bb`: `std::hash::blake3` | `blake3` (no BLAKE3 black box) | both, the plot takes the faster one; the black box takes at most 1024 bytes |
| Poseidon2 | `poseidon2_t16` (TACEO width 16, rate 15) and `poseidon2_native` (width-4 black box, rate 3); the plot takes the faster one per backend | yes | yes |
| Artemis | `artemis` + hook + IPA opening | **implemented, sound** | `artemis_estimate` + KZG opening: **cost estimate only** |
| Lunar | n/a | not applicable (no KZG columns, no pairing) | not implemented |

All rows take `payload: [u8; N]`, so every row pays for the same byte range checks, and the Poseidon2 and
Artemis rows pack 31 bytes per field element inside the proof system.

- **ProveKit** (`worldfnd/provekit` @ `893f6d58`, Noir `v1.0.0-beta.26`, Spartan + WHIR over the BN254
  scalar field, Skyscraper hash, hiding witness commitment). Timed in-process.
- **Barretenberg** (`bb`, UltraHonk with ZK, default verifier target; local `nargo`). Timed as the
  wall-clock of `bb prove` / `bb verify` minus the median wall-clock of the same command on the `empty`
  circuit; both raw values are in the CSV (`prover_raw_s`, `baseline_prove_s`).

## The Artemis hook in ProveKit

The Noir source of the Artemis row contains only the predicate. `artemis-bench prepare --artemis` marks the
private parameters `payload: [u8; N]` and `mask: Field` by name (`ArtemisConfig`) and adds, after ACIR
compilation (`provekit/r1cs-compiler/src/artemis.rs`):

- a challenge `beta` (`WitnessBuilder::Challenge`, drawn after the first witness block `w1` is committed),
- the Horner recurrence over the packed payload `u_0..u_{k-1}` (31 bytes each; packing is a linear
  combination and costs no constraint) and the mask: `k + 1` multiplication constraints,
- `rho = beta * (u(beta) + mask * beta^k)` as a witness in `w2` that the prover reveals.

Protocol:

1. The external commitments are `c = <u, G> + r H` (from the TLSNotary session in a deployment) and
   `c_mu = mask * G_k + r_mu H` (from the prover). `SHA-256(c, c_mu)` becomes the session id of ProveKit's
   domain separator, so `beta` depends on both.
2. The prover commits `w1` (payload bytes and mask), draws `beta`, commits `w2`, and sends `rho` as a
   transcript message right after the second commitment. The joint WHIR opening binds `rho` to its `w2`
   position with the same linear form that already binds the challenges (`exposed_offsets` in
   `WhirR1CSScheme`).
3. The verifier checks the ProveKit proof, reads `beta` and `rho`, and checks an IPA opening of `c + c_mu`
   at `beta` to `rho / beta`.

Points that the soundness of the link depends on, and how the code handles them:

- **Payload committed before `beta`.** ProveKit's witness splitter balances `w1`/`w2` and only pins the
  dependencies of direct challenge consumers to `w1`. The trailing factor `beta` in `rho` makes every
  packed element feed a product with `beta`; the compiler then checks after the split that every payload
  byte and the mask sit in `w1` and aborts otherwise.
- **Gaussian elimination.** ProveKit's optimizer may pivot a witness away, which would redefine a payload
  byte through later columns. Payload, mask and `rho` columns are excluded from pivoting
  (`optimize_r1cs_protected`).
- **`rho` in the transcript before the binding point** of the joint opening is drawn.

Tests (`provekit/tooling/artemis-bench/tests/soundness.rs`): honest proof and `rho / beta = p(beta)`,
commitment to another payload, wrong mask commitment, proof replayed under other commitments (different
`beta`), tampered `rho`, plain verification of a session-bound proof, public payload refused.

Not covered: a security proof for this instantiation. Section 6.6 of the plan (the theorem's assumptions
for Spartan + WHIR with a two-block commitment) is still open; the code only matches the protocol shape.

## What is measured

- **Prover time**: proof generation from a solved ACIR witness. ProveKit Artemis adds the IPA opening
  (`opening_s`). The commitment `c` to the payload is timed (`commit_s`) and not counted, as in the zkML
  evaluation. Excluded: Noir compilation, key generation, ACVM witness generation (`witgen_s`), generator
  setup.
- **Verifier time**: ProveKit: mean of 5 in-process repetitions; Artemis includes the IPA check (one MSM of
  size `k`). Barretenberg: wall-clock of `bb verify` minus the empty-circuit baseline, so small values are
  noisy.
- **Proof size**: ProveKit transcript + hints + public inputs; Artemis adds `c_mu` and the opening
  (compressed points). Barretenberg: size of the `proof` file.
- **Peak memory**: peak RSS of the proving process (ProveKit: read after proving, the process also loads
  the prover key; Barretenberg: `wait4` on `bb prove`).
- **Constraints**: ProveKit R1CS constraints after optimization; Barretenberg `circuit_size` from `bb gates`.

`external` rows in the CSV are the standalone commitment schemes (`pcs_bench`), used to complete the
Barretenberg estimate.

## Results (Apple M5 Pro, 48 GB, 5 runs, median; `results/summary.md` has all tables)

Prover time in seconds:

| Payload | ProveKit no com. | SHA-256 | Poseidon2 | Artemis | bb no com. | SHA-256 | Poseidon2 | Artemis (estimate) |
|---|---|---|---|---|---|---|---|---|
| 520 B | 0.023 | 0.120 | 0.025 | 0.027 | 0.010 | 0.207 | 0.019 | 0.012 |
| 16 KiB | 0.048 | 1.62 | 0.065 | 0.066 | 0.100 | 5.19 | 0.190 | 0.117 |
| 64 KiB | 0.107 | 7.20 | 0.184 | 0.151 | 0.293 | 40.1 | 0.605 | 0.349 |

- **Against SHA-256 (what TLSNotary ships)** the gain grows with the payload: 4.5x at 520 B, 25x at 16 KiB,
  48x at 64 KiB on ProveKit (115x on Barretenberg, estimate), and SHA-256 needs 16 GiB of prover memory at
  64 KiB against 0.2 GiB.
- **Keccak-256 and BLAKE3** (the other two hashes TLSNotary supports). BLAKE3 costs the same as SHA-256
  on ProveKit (1.63 s at 16 KiB, 6.3 s and 10 GiB at 64 KiB; 41x Artemis) and 16.9 s at 64 KiB on
  Barretenberg. Keccak-256 is the most expensive: 12.4 s and 14 GiB at 16 KiB on ProveKit (188x Artemis,
  10.7 M constraints, about 158 k per permutation without a black box), 10.4 s at 16 KiB and 39.5 s at
  64 KiB on Barretenberg with the black box. In the age check Keccak-256 costs 214 ms on ProveKit against
  25 ms for Artemis; BLAKE3 39 ms.
- **Against Poseidon2** the prover-time gain on ProveKit is small: none up to 16 KiB, 1.2x at 64 KiB. The
  Artemis circuit has 1.03x the constraints of the no-commitment circuit (Poseidon2: 2.2x), but the byte
  range checks that every row pays dominate, and Artemis proves at 1.4x the no-commitment time because of
  the second witness block and the IPA opening. The verifier is where
  Artemis is ahead of Poseidon2 on ProveKit (21 ms against 61 ms at 64 KiB).
- **Age check as shipped (10 committed bytes)**: no measurable difference between Artemis, Poseidon2 and no
  commitment (25 ms on ProveKit); SHA-256 costs 38 ms (ProveKit) and 151 ms against 46 ms (Barretenberg).

## Caveats

1. **Upstream binding is not covered** (plan, Section 6.1). TLSNotary authenticates SHA-256 hashes; neither
   a Pedersen nor a Poseidon2 commitment comes out of TLSNotary today. The numbers cover the downstream
   predicate proof only, for the Artemis and the Poseidon2 rows alike.
2. **Barretenberg Artemis is an estimate and not sound**: the evaluation point is a public input.
3. **Keccak-256 above 16 KiB without a black box** (ProveKit, and the explicit variant on Barretenberg)
   is skipped locally: about 50 GiB at 64 KiB by extrapolation. `bench.py --all-sizes` lifts the cap. The
   expected digests come from `pcs/src/bin/digest.rs` (`sha3`, `blake3` crates), so every run checks the
   two explicit circuits against a reference implementation.
4. **Not built**: the whole-response predicates (plan, workload 3, including the go/no-go measurement) and
   the upstream cost (plan, step 7).
5. KZG uses a locally sampled trapdoor (timing only).
