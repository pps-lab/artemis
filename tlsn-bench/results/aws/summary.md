# TLSNotary / Artemis benchmark: summary (AWS r6i.8xlarge)

Medians over the runs in the CSV. Prover time excludes ACVM witness generation, key generation and setup; the Artemis rows include the IPA opening (ProveKit) or the KZG opening (Barretenberg estimate). The LegoSNARK row is the no-commitment row plus the CP_link prover on the wire columns of a real Barretenberg proof (same circuit; the CSV rows hold the sum with a `bb prove` run of their own session).

## Barretenberg (UltraHonk, KZG): Prover time [s]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon | Artemis (cost estimate) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|---|
| 520 B | 0.019 | 0.340 | 0.717 | 0.195 | 0.140 | 0.022 | 0.021 |
| 2.3 KiB | 0.044 | 1.079 | 2.642 | 1.147 | 0.378 | 0.054 | 0.046 |
| 3.9 KiB | 0.066 | 1.517 | 4.514 | 1.970 | 0.612 | 0.078 | 0.069 |
| 9.8 KiB | 0.115 | 4.680 | 10.087 | 4.366 | 1.254 | 0.139 | 0.119 |
| 16.0 KiB | 0.163 | 9.225 | 17.130 | 7.549 | 2.137 | 0.194 | 0.168 |
| 64.0 KiB | 0.469 | 71.018 | 65.765 | 28.572 | 8.002 | 0.566 | 0.477 |

## Barretenberg (UltraHonk, KZG): Constraints (ProveKit R1CS) / gates (bb circuit size)

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon | Artemis (cost estimate) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|---|
| 520 B | 731 | 39659 | 74067 | 20701 | 9299 | 919 | 731 |
| 2.3 KiB | 3081 | 158149 | 323967 | 193699 | 41209 | 3936 | 3081 |
| 3.9 KiB | 5081 | 260255 | 538127 | 317217 | 68009 | 6504 | 5081 |
| 9.8 KiB | 12581 | 644144 | 1323847 | 791500 | 167109 | 16133 | 12581 |
| 16.0 KiB | 20561 | 1052522 | 2163051 | 1290322 | 273351 | 26378 | 20561 |
| 64.0 KiB | 82001 | 4189226 | 8609263 | 5162242 | 1091308 | 105260 | 82001 |

## Barretenberg (UltraHonk, KZG): Verifier time [ms]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon | Artemis (cost estimate) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|---|
| 520 B | 0.0 | 0.2 | 0.2 | 0.3 | 0.1 | 1.8 | 1.1 |
| 2.3 KiB | 0.2 | 0.2 | 0.5 | 0.3 | 0.1 | 1.9 | 1.2 |
| 3.9 KiB | 0.2 | 0.3 | 0.3 | 0.3 | 0.2 | 1.9 | 1.3 |
| 9.8 KiB | 0.2 | 0.3 | 0.3 | 0.4 | 0.1 | 2.0 | 1.5 |
| 16.0 KiB | 0.1 | 0.3 | 0.3 | 0.3 | 0.2 | 1.9 | 1.6 |
| 64.0 KiB | 0.4 | 0.6 | 0.6 | 0.6 | 0.2 | 2.1 | 1.6 |

## Barretenberg (UltraHonk, KZG): Proof size [kB]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon | Artemis (cost estimate) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|---|
| 520 B | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 2.3 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 3.9 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 9.8 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 16.0 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 64.0 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |

## Barretenberg (UltraHonk, KZG): Prover peak memory [GiB]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon | Artemis (cost estimate) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|---|
| 520 B | 0.04 | 0.12 | 0.17 | 0.08 | 0.06 | 0.04 | 0.04 |
| 2.3 KiB | 0.05 | 0.29 | 0.53 | 0.34 | 0.12 | 0.05 | 0.05 |
| 3.9 KiB | 0.05 | 0.43 | 0.90 | 0.55 | 0.16 | 0.05 | 0.05 |
| 9.8 KiB | 0.06 | 1.10 | 2.00 | 1.31 | 0.31 | 0.07 | 0.06 |
| 16.0 KiB | 0.08 | 2.37 | 3.28 | 2.12 | 0.48 | 0.09 | 0.08 |
| 64.0 KiB | 0.17 | 36.35 | 12.89 | 8.18 | 1.76 | 0.20 | 0.17 |

## Barretenberg: LegoSNARK CP_link, no-commitment circuit, byte cells linked (the plotted row)

Link prover, Pedersen commitment and link verifier are in-process medians. Overhead columns: prover time over no commitment. LegoSNARK: the link prover (the circuit is the no-commitment circuit) or, for the packed variant, the difference of the `bb prove` medians of the two LegoSNARK circuits (measured back to back) plus the link prover. Artemis: `artemis_estimate` + KZG opening - `nocom` (measured back to back).

| Payload | Gates | Linked wires | Non-zero cells in the MSM | of at most 16 bits | Link prover [ms] | LegoSNARK overhead [ms] | Artemis estimate overhead [ms] | Pedersen commit [ms] (not counted) | Link verifier [ms] | Setup [s] | Keys [MB] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 520 B | 731 | w_l | 699 | 99.57% | 1.5 | 1.5 | 2.7 | 2.6 | 1.1 | 0.01 | 0.0 |
| 2.3 KiB | 3081 | w_l | 3044 | 99.90% | 2.6 | 2.6 | 10.3 | 2.3 | 1.1 | 0.04 | 0.2 |
| 3.9 KiB | 5081 | w_l | 5024 | 99.94% | 2.8 | 2.8 | 11.9 | 2.0 | 1.1 | 0.06 | 0.3 |
| 9.8 KiB | 12581 | w_l | 12491 | 99.98% | 4.2 | 4.2 | 23.9 | 4.0 | 1.1 | 0.15 | 0.8 |
| 16.0 KiB | 20561 | w_l | 20461 | 99.99% | 5.5 | 5.5 | 31.5 | 4.0 | 1.1 | 0.23 | 1.4 |
| 64.0 KiB | 82001 | w_l | 81651 | 100.00% | 8.5 | 8.5 | 96.8 | 6.4 | 1.1 | 0.91 | 5.4 |

## Barretenberg: LegoSNARK CP_link, variant that packs in the circuit, packed cells linked

Link prover, Pedersen commitment and link verifier are in-process medians. Overhead columns: prover time over no commitment. LegoSNARK: the link prover (the circuit is the no-commitment circuit) or, for the packed variant, the difference of the `bb prove` medians of the two LegoSNARK circuits (measured back to back) plus the link prover. Artemis: `artemis_estimate` + KZG opening - `nocom` (measured back to back).

| Payload | Gates | Linked wires | Non-zero cells in the MSM | of at most 16 bits | Link prover [ms] | LegoSNARK overhead [ms] | Artemis estimate overhead [ms] | Pedersen commit [ms] (not counted) | Link verifier [ms] | Setup [s] | Keys [MB] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 520 B | 915 | w_l, w_o | 1229 | 97.97% | 5.2 | 7.9 | 2.7 | 2.8 | 1.3 | 0.02 | 0.1 |
| 2.3 KiB | 3933 | w_l | 3895 | 97.92% | 3.6 | 8.8 | 10.3 | 3.6 | 1.1 | 0.03 | 0.3 |
| 3.9 KiB | 6501 | w_l, w_r | 8764 | 98.46% | 6.9 | 17.0 | 11.9 | 2.1 | 1.3 | 0.08 | 0.8 |
| 9.8 KiB | 16129 | w_l, w_o | 21774 | 98.48% | 9.6 | 28.2 | 23.9 | 2.4 | 1.3 | 0.21 | 2.1 |
| 16.0 KiB | 26375 | w_l | 26257 | 97.97% | 7.3 | 31.9 | 31.5 | 4.3 | 1.1 | 0.18 | 1.7 |
| 64.0 KiB | 105256 | w_l, w_o | 142250 | 98.51% | 19.9 | 107.2 | 96.8 | 5.4 | 1.3 | 1.27 | 13.6 |

## Age check (10 committed bytes)

| Backend | Row | Constraints / gates | Prover [ms] | Verifier [ms] | Proof [kB] |
|---|---|---|---|---|---|
| bb | age_blake3 | 10395 | 122.5 | 0.2 | 14.7 |
| bb | age_blake3_bb | 17416 | 150.9 | 0.3 | 14.7 |
| bb | age_keccak256 | 49091 | 352.1 | 0.3 | 14.7 |
| bb | age_keccak256_bb | 40626 | 330.9 | 0.3 | 14.7 |
| bb | age_nocom | 5559 | 68.3 | 0.1 | 14.7 |
| bb | age_sha256 | 36000 | 238.9 | 0.2 | 14.7 |
| provekit | age_artemis | 2695 | 43.0 | 16.3 | 339.4 |
| provekit | age_blake3 | 14267 | 65.7 | 20.3 | 358.1 |
| provekit | age_keccak256 | 158349 | 459.4 | 78.6 | 499.8 |
| provekit | age_nocom | 2693 | 43.2 | 15.5 | 337.9 |
| provekit | age_sha256 | 15934 | 66.7 | 22.6 | 359.1 |
