# TLSNotary benchmark for Artemis

Implements `zkml-commitments-paper/tlsn-eval-plan.md`: a TLSNotary-style predicate proof over a committed
response of `n` bytes, where the commitment is either recomputed in the circuit (SHA-256 as TLSNotary
ships it, or Poseidon2) or linked with Artemis (Horner evaluation in the circuit, one opening outside).

```
circuits/               Noir packages (one source tree for both backends), N is set per run
pcs/                    external commitment: Pedersen + zero-knowledge IPA over BN254 G1, hiding KZG,
                        LegoSNARK CP_link (lego.rs, lego_bench)
provekit-artemis.patch  the Artemis hook for worldfnd/provekit @ 893f6d58 (applied by ./run.sh setup)
provekit/               patched ProveKit checkout (not tracked)
barretenberg-dump.patch wire-column dump for Barretenberg v5.0.0-nightly.20260324 (./run.sh setup-bb-dump)
aztec-packages/         patched Barretenberg checkout (not tracked)
bench.py, run.sh        sweep driver, results as CSV
plot.py                 plot + tables (results/summary.md)
results/                results.csv, lego_layout.json, logs, sweep_prover.{pdf,png}, summary.md
```

## Quick start

```bash
./run.sh setup     # clone + patch + build ProveKit (~10 min), build pcs
./run.sh test      # 9 soundness tests of the link, 6 tests of the pcs crate
./run.sh small     # age check + sweep 520 B .. 64 KiB, all rows; peak ~20 GiB RAM
./run.sh large     # 83.6 KiB .. 1 MiB without SHA-256; peak ~20 GiB RAM
./run.sh setup-bb-dump   # clone + patch + build Barretenberg (~5 min), only for the LegoSNARK rows
./run.sh lego            # LegoSNARK rows on Barretenberg, all sizes; peak ~4 GiB RAM
./run.sh plot
```

Needs `nargo` and `bb` on `PATH` for the Barretenberg rows (`NARGO=`, `BB=` override), Python 3 with
matplotlib, and the Rust toolchain pinned by ProveKit (`nightly-2026-03-04`, picked up from
`rust-toolchain.toml`). `./run.sh setup-bb-dump` needs `cmake`, `ninja` and a C++20 clang (Apple clang 21
works); without the patched binary the LegoSNARK rows are skipped.

On AWS: `doe-suite-config/designs/tlsn.yml` (suite `tlsn`, role `setup-tlsn`, host types `tlsn_small` =
r6i.8xlarge and `tlsn_large` = r6i.16xlarge for SHA-256 above 64 KiB). The instances clone the `tlsn-bench`
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
| LegoSNARK | `nocom` + CP_link on the wire commitments | not applicable (no KZG columns, no pairing) | `lego_estimate`: **cost estimate**, link run on the wire columns of a real proof (below) |
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

## LegoSNARK CP_link on Barretenberg

UltraHonk commits to each wire column as `W = sum_i w_i [tau^i]_1` (the row values as monomial
coefficients); with ZK, three random values sit in the last three rows of each wire. LegoSNARK's CP_link
(Kiltz-Wee QA-NIZK for linear subspaces, `pcs/src/lego.rs`, same maths as `src/utils/lego_cplink.rs` of the
zkML evaluation) proves that the Pedersen commitment `c = r h_0 + sum_j u_j h_{1+j}` and the wire
commitments in the proof are the image of one vector `(r, wire cells)` under a public matrix `M`. The
proof is one G1 element, `pi = sum_cells w_cell P_cell + r P_0`, where the key `P = k^T M` has one point
per cell of every linked wire; the verifier computes one multi-pairing of (linked wires + 2) pairings.

The row `lego_estimate` uses the `nocom` circuit unchanged. `c` commits to the payload packed 31 bytes per
element, as in the Artemis row; the packing is linear, so it is part of `M` (byte `j` contributes
`256^(j mod 31) h_{1 + j div 31}`) and costs no gate. The uniqueness of the packing rests on the byte
range checks of the circuit.

**The cost is measured on the wire columns of a real proof.** `barretenberg-dump.patch` (for the tag of
the local `bb`, `v5.0.0-nightly.20260324`; `./run.sh setup-bb-dump`) makes `bb prove` write, when
`BB_DUMP_WIRES=<dir>` is set, the four wire columns it committed to with their masking values and
commitments (`wires.bin`) and the builder variable held by every cell (`cells.bin`). Apart from that it
only touches build files (a `BB_LITE` build of the CLI on macOS needs the `ipc` module). For every size `bench.py` proves once with the patched binary, checks that proof with the stock
`bb verify`, and runs `pcs/target/release/lego_bench`, which

1. finds every cell that holds a payload byte and the smallest set of wires that covers all bytes,
2. recomputes all four wire commitments from the dumped values with the points of `~/.bb-crs/bn254_g1.dat`
   and finds them in the proof file (bytes 384, 512, 640 and 1024; the run aborts otherwise),
3. runs setup, Pedersen commitment, link prover and link verifier on those values, with the key built on
   Barretenberg's points and the verifier fed the commitments of that proof (an honest link is accepted,
   a commitment to another payload is rejected).

The dumps stay in `.build/<row>_<bytes>/dump/` (`wires.bin`, `cells.bin`, `proof`, `payload.bin`,
`lego.json`). Both files are little endian. `wires.bin`: `"BBWIRES1"`, u64 dyadic size `n`, u64 active
rows, u64 masked rows (3); per wire (`w_l, w_r, w_o, w_4`) the commitment `x, y` and the masking values
of rows `n-3 .. n-1` (32 bytes each); then per wire the values of the active rows (32 bytes each,
canonical integers). `cells.bin`: `"BBCELLS1"`, u64 active rows, u64 number of variables; per row four
u32, the copy-class representative of the variable in `w_l, w_r, w_o, w_4`; then that representative for
every builder variable. `load_dump` in `pcs/src/bin/lego_bench.rs` reads both.

What the dump shows for `nocom` (all sizes; `results/lego_layout.json`, tables in `results/summary.md`):

- Payload byte `j` is builder variable `j`. Each byte sits in one cell of `w_l`, in a row of its own whose
  other three wires hold the zero variable (byte 0 also appears once in each other wire). **One linked
  wire, `w_l`; three pairings.**
- The range check adds a sorted copy of the bytes, four per row, so `w_l` holds `N + N/4` byte-sized
  values. The MSM runs over the non-zero cells of `w_l`: 1.25 N cells minus the zero bytes, all of at most
  8 bits except the three masking values (20 461 cells at 16 KiB, 1 305 745 at 1 MiB; 99.6% to 100% small).
  arkworks' MSM puts 8-bit scalars into 255 buckets, which is why the link prover is cheap: 2.4 ms at
  16 KiB, 3.1 ms at 64 KiB, 28.6 ms at 1 MiB.
- LegoSNARK adds no gate, so the circuit never crosses a power of two that `nocom` does not cross. The
  Artemis estimate adds about 0.35 gates per byte, nearly all of them for the packing (the circuit below,
  which packs and has no Horner evaluation, has 3 or 4 gates fewer), and has twice the dyadic size of
  `nocom` at 83.6 KiB, 89.4 KiB and 185.8 KiB.

`lego_packed_estimate` is the variant that packs inside the circuit (`circuits/lego_packed`) and links the
cells of the packed elements. It is the weaker baseline: 1.28x the gates, the packed elements end up in
two wires at most sizes (four pairings), and its overhead over no commitment (17 ms at 16 KiB, 70 ms at
64 KiB, 703 ms at 1 MiB) is that of the Artemis estimate. It is in the CSV and in `summary.md`, not in
the plot.

Per-circuit setup (not counted): a trapdoor `k` and the key `P`, one G1 point per cell of `w_l` up to the
last used row plus the masking rows, and the Pedersen key: 1.4 MB and 0.15 s at 16 KiB, 5.4 MB and 0.6 s
at 64 KiB, 86 MB and 10 s at 1 MiB. The key depends on the circuit (the position of the payload cells), so
every predicate and payload size needs its own trusted setup, in addition to the KZG reference string.

## What is measured

- **Prover time**: proof generation from a solved ACIR witness. ProveKit Artemis adds the IPA opening
  (`opening_s`). The commitment `c` to the payload is timed (`commit_s`) and not counted, as in the zkML
  evaluation. Excluded: Noir compilation, key generation, ACVM witness generation (`witgen_s`), generator
  setup.
- **Verifier time**: ProveKit: mean of 5 in-process repetitions; Artemis includes the IPA check (one MSM of
  size `k`). Barretenberg: wall-clock of `bb verify` minus the empty-circuit baseline, so small values are
  noisy.
- **Proof size**: ProveKit transcript + hints + public inputs; Artemis adds `c_mu` and the opening
  (compressed points). Barretenberg: size of the `proof` file; LegoSNARK adds 32 bytes.
- **LegoSNARK row** (`lego_estimate`): prover = `bb prove` of the `nocom` circuit + the link prover
  (`opening_s`, in-process); verifier = `bb verify` + the multi-pairing (5 repetitions); memory = the larger
  of `bb prove` and the link prover process. The Pedersen commitment is `commit_s` and not counted. The
  CSV rows contain a `bb prove` run of their own session; because the circuit is the `nocom` circuit and
  `bb prove` drifts by several percent between sessions, the plot and the tables use the `nocom` row plus
  the link prover.
- **Peak memory**: peak RSS of the proving process (ProveKit: read after proving, the process also loads
  the prover key; Barretenberg: `wait4` on `bb prove`).
- **Constraints**: ProveKit R1CS constraints after optimization; Barretenberg `circuit_size` from `bb gates`.

`external` rows in the CSV are the standalone commitment schemes (`pcs_bench`), used to complete the
Barretenberg Artemis estimate.

## Results (Apple M5 Pro, 48 GB, 5 runs, median; `results/summary.md` has all tables)

Prover time in seconds:

| Payload | ProveKit no com. | SHA-256 | Poseidon2 | Artemis | bb no com. | SHA-256 | Poseidon2 | Artemis (estimate) | LegoSNARK (estimate) |
|---|---|---|---|---|---|---|---|---|---|
| 520 B | 0.023 | 0.120 | 0.025 | 0.027 | 0.010 | 0.207 | 0.019 | 0.012 | 0.011 |
| 16 KiB | 0.048 | 1.62 | 0.065 | 0.066 | 0.100 | 5.19 | 0.190 | 0.117 | 0.102 |
| 64 KiB | 0.107 | 7.20 | 0.184 | 0.151 | 0.293 | 40.1 | 0.605 | 0.349 | 0.297 |
| 1 MiB | 1.18 | not run | 2.30 | 1.64 | 3.61 | not run | 7.55 | 4.32 | 3.64 |

- **Against SHA-256 (what TLSNotary ships)** the gain grows with the payload: 4.5x at 520 B, 25x at 16 KiB,
  48x at 64 KiB on ProveKit (115x on Barretenberg, estimate), and SHA-256 needs 16 GiB of prover memory at
  64 KiB against 0.2 GiB.
- **Keccak-256 and BLAKE3** (the other two hashes TLSNotary supports). BLAKE3 costs the same as SHA-256
  on ProveKit (1.63 s at 16 KiB, 6.3 s and 10 GiB at 64 KiB; 41x Artemis) and 16.9 s at 64 KiB on
  Barretenberg. Keccak-256 is the most expensive: 12.4 s and 14 GiB at 16 KiB on ProveKit (188x Artemis,
  10.7 M constraints, about 158 k per permutation without a black box), 10.4 s at 16 KiB and 39.5 s at
  64 KiB on Barretenberg with the black box. In the age check Keccak-256 costs 214 ms on ProveKit against
  25 ms for Artemis; BLAKE3 39 ms.
- **Against Poseidon2** the prover-time gain on ProveKit is small: none below 10 kB, 1.2x to 1.7x above. The
  Artemis circuit has 1.03x the constraints of the no-commitment circuit (Poseidon2: 2.2x), but the byte
  range checks that every row pays dominate, and Artemis proves at 1.4x the no-commitment time because of
  the second witness block and the IPA opening (0.34 s of the 1.64 s at 1 MiB). The verifier is where
  Artemis is ahead of Poseidon2 on ProveKit (21 ms against 61 ms at 64 KiB).
- **LegoSNARK on Barretenberg** is cheaper for the prover than the Artemis estimate at every size. Prover
  overhead over no commitment:

  | Payload | LegoSNARK CP_link | Artemis (estimate) |
  |---|---|---|
  | 16 KiB | 2.4 ms (2.4%) | 17.0 ms (17%) |
  | 64 KiB | 3.1 ms (1.1%) | 55 ms (19%) |
  | 1 MiB | 28.6 ms (0.8%) | 709 ms (20%) |

  The link adds no gate and its MSM runs over byte-sized scalars; the Artemis estimate pays for packing
  the bytes in the circuit (0.35 gates per byte) plus a KZG opening over full-size scalars. The link verifier costs
  0.4 ms (three pairings) and the proof grows by 32 bytes. The price is a trusted setup per circuit with
  a key of 82 bytes per payload byte (86 MB at 1 MiB), and the scheme needs KZG wire commitments, so it
  does not exist on ProveKit.

  The same on AWS (r6i.8xlarge, doe-suite suite `tlsn-lego_1794953544`, design `tlsn-lego`), with the other
  Barretenberg rows from suite `tlsn_1794941917`; figure and tables in `results/aws/`:

  | Payload | No commitment | LegoSNARK CP_link overhead | Artemis (estimate) overhead |
  |---|---|---|---|
  | 16 KiB | 0.163 s | 5.5 ms (3.4%) | 31.5 ms (19%) |
  | 64 KiB | 0.469 s | 8.5 ms (1.8%) | 97 ms (21%) |
  | 186 KiB | 1.037 s | 16.2 ms (1.6%) | 668 ms (64%) |
  | 1 MiB | 6.347 s | 123 ms (1.9%) | 1301 ms (20%) |

  The LegoSNARK prover time is always the no-commitment time plus the link prover. The CSV rows of
  `lego_estimate` hold the sum with a `bb prove` run of their own, which differs from the `nocom` median by
  run-to-run noise of a few percent in either direction (at 520 B and 256 KiB on AWS it came out below
  `nocom`); plot, summary and the combined table therefore use `nocom` + link.

  ```bash
  R=../doe-suite-results
  python3 plot.py $R/tlsn_1794941917 $R/tlsn-lego_1794953544 --out results/aws --backends bb --machine "AWS r6i.8xlarge"
  ```
- **Age check as shipped (10 committed bytes)**: no measurable difference between Artemis, Poseidon2 and no
  commitment (25 ms on ProveKit); SHA-256 costs 38 ms (ProveKit) and 151 ms against 46 ms (Barretenberg).

## Caveats

1. **Upstream binding is not covered** (plan, Section 6.1). TLSNotary authenticates SHA-256 hashes; neither
   a Pedersen nor a Poseidon2 commitment comes out of TLSNotary today. The numbers cover the downstream
   predicate proof only, for the Artemis and the Poseidon2 rows alike.
2. **Barretenberg Artemis is an estimate and not sound**: the evaluation point is a public input.
3. **SHA-256 above 64 KiB does not fit a 48 GB laptop** (about 0.25 to 0.3 GiB of prover RAM per kB of
   payload on both backends). `./run.sh large-sha` is for a large-memory machine.
4. **Keccak-256 above 16 KiB without a black box** (ProveKit, and the explicit variant on Barretenberg)
   is skipped locally: about 50 GiB at 64 KiB by extrapolation. `bench.py --all-sizes` lifts the cap. The
   expected digests come from `pcs/src/bin/digest.rs` (`sha3`, `blake3` crates), so every run checks the
   two explicit circuits against a reference implementation.
5. **Not built**: the whole-response predicates (plan, workload 3, including the go/no-go measurement) and
   the upstream cost (plan, step 7).
6. KZG uses a locally sampled trapdoor (timing only).
7. **Barretenberg LegoSNARK is a cost estimate, not an integrated prover.** Real: the circuit, the wire
   values, their position, the masking values, the reference-string points, the wire commitments (taken
   from a proof that the stock `bb verify` accepts) and the link itself, which verifies against those
   commitments. Not real: the link prover is a separate process that reads a dump written by a patched
   `bb` instead of running inside `bb prove` (its time is added to the wall-clock of the stock binary),
   the trapdoor of the link is sampled locally, and the transcript of the `bb` proof does not bind `c`
   and `pi`. There is no security proof for this instantiation; CP_link needs the wire commitment to be
   extractable and the linked cells to be fixed by the circuit, which holds for the layout that the
   verification key commits to. The layout was read from Barretenberg `v5.0.0-nightly.20260324` and can
   change with the version.
8. The LegoSNARK MSM time depends on the MSM implementation's handling of small scalars (arkworks 0.6
   buckets 8-bit and 16-bit scalars separately). The same MSM with full-size random scalars takes 10 ms
   for the 20 461 cells at 16 KiB and 354 ms for the 1 305 745 cells at 1 MiB (`pcs_bench kzg <cells>`,
   `commit_s`), against 2.4 ms and 28.6 ms on the real values.
