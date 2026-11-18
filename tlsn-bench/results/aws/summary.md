# TLSNotary / Artemis benchmark: summary (AWS r6i.8xlarge)

Medians over the runs in the CSV. Prover time excludes ACVM witness generation, key generation and setup; the Artemis rows include the IPA opening (ProveKit) or the KZG opening (Barretenberg estimate). The LegoSNARK row is the no-commitment row plus the CP_link prover on the wire columns of a real Barretenberg proof (same circuit; the CSV rows hold the sum with a `bb prove` run of their own session).

## Barretenberg (UltraHonk, KZG): Prover time [s]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis (cost estimate, not sound) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|---|
| 520 B | 0.019 | 0.340 | 0.717 | 0.195 | 0.033 | 0.022 | 0.021 |
| 2.3 KiB | 0.044 | 1.079 | 2.642 | 1.147 | 0.081 | 0.054 | 0.046 |
| 3.9 KiB | 0.066 | 1.517 | 4.514 | 1.970 | 0.124 | 0.078 | 0.069 |
| 9.8 KiB | 0.115 | 4.680 | 10.087 | 4.366 | 0.200 | 0.139 | 0.119 |
| 16.0 KiB | 0.163 | 9.225 | 17.130 | 7.549 | 0.304 | 0.194 | 0.168 |
| 64.0 KiB | 0.469 | 71.018 | 65.765 | 28.572 | 0.949 | 0.566 | 0.477 |
| 83.6 KiB | 0.529 | 114.870 | n/a | n/a | 1.067 | 0.835 | 0.538 |
| 89.4 KiB | 0.542 | 126.990 | n/a | n/a | 1.103 | 0.856 | 0.553 |
| 185.8 KiB | 1.037 | 494.611 | n/a | n/a | 2.158 | 1.705 | 1.053 |
| 256.0 KiB | 1.660 | n/a | n/a | n/a | 3.408 | 1.995 | 1.682 |
| 1 MiB | 6.347 | n/a | n/a | n/a | 12.686 | 7.648 | 6.470 |

## Barretenberg (UltraHonk, KZG): Constraints (ProveKit R1CS) / gates (bb circuit size)

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis (cost estimate, not sound) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|---|
| 520 B | 731 | 39659 | 74067 | 20701 | 1357 | 919 | 731 |
| 2.3 KiB | 3081 | 158149 | 323967 | 193699 | 5834 | 3936 | 3081 |
| 3.9 KiB | 5081 | 260255 | 538127 | 317217 | 9716 | 6504 | 5081 |
| 9.8 KiB | 12581 | 644144 | 1323847 | 791500 | 24017 | 16133 | 12581 |
| 16.0 KiB | 20561 | 1052522 | 2163051 | 1290322 | 39299 | 26378 | 20561 |
| 64.0 KiB | 82001 | 4189226 | 8609263 | 5162242 | 156724 | 105260 | 82001 |
| 83.6 KiB | 107089 | 5467688 | n/a | n/a | 204702 | 137469 | 107089 |
| 89.4 KiB | 114514 | 5847500 | n/a | n/a | 218907 | 147002 | 114514 |
| 185.8 KiB | 237905 | 12145461 | n/a | n/a | 454778 | 305420 | 237905 |
| 256.0 KiB | 327761 | n/a | n/a | n/a | 626570 | 420784 | 327761 |
| 1 MiB | 1310801 | n/a | n/a | n/a | 2506028 | 1682880 | 1310801 |

## Barretenberg (UltraHonk, KZG): Verifier time [ms]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis (cost estimate, not sound) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|---|
| 520 B | 0.0 | 0.2 | 0.2 | 0.3 | 0.0 | 1.8 | 1.1 |
| 2.3 KiB | 0.2 | 0.2 | 0.5 | 0.3 | 0.1 | 1.9 | 1.2 |
| 3.9 KiB | 0.2 | 0.3 | 0.3 | 0.3 | 0.2 | 1.9 | 1.3 |
| 9.8 KiB | 0.2 | 0.3 | 0.3 | 0.4 | 0.2 | 2.0 | 1.5 |
| 16.0 KiB | 0.1 | 0.3 | 0.3 | 0.3 | 0.1 | 1.9 | 1.6 |
| 64.0 KiB | 0.4 | 0.6 | 0.6 | 0.6 | 0.4 | 2.1 | 1.6 |
| 83.6 KiB | 0.2 | 2.6 | n/a | n/a | 0.1 | 1.9 | 1.8 |
| 89.4 KiB | 0.3 | 1.0 | n/a | n/a | 0.3 | 2.0 | 1.7 |
| 185.8 KiB | 0.4 | 2.2 | n/a | n/a | 0.3 | 2.0 | 1.7 |
| 256.0 KiB | 0.2 | n/a | n/a | n/a | 0.1 | 2.0 | 1.7 |
| 1 MiB | 0.3 | n/a | n/a | n/a | 0.3 | 2.1 | 1.6 |

## Barretenberg (UltraHonk, KZG): Proof size [kB]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis (cost estimate, not sound) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|---|
| 520 B | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 2.3 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 3.9 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 9.8 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 16.0 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 64.0 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 83.6 KiB | 14.7 | 14.7 | n/a | n/a | 14.7 | 14.7 | 14.7 |
| 89.4 KiB | 14.7 | 14.7 | n/a | n/a | 14.7 | 14.7 | 14.7 |
| 185.8 KiB | 14.7 | 14.7 | n/a | n/a | 14.7 | 14.7 | 14.7 |
| 256.0 KiB | 14.7 | n/a | n/a | n/a | 14.7 | 14.7 | 14.7 |
| 1 MiB | 14.7 | n/a | n/a | n/a | 14.7 | 14.7 | 14.7 |

## Barretenberg (UltraHonk, KZG): Prover peak memory [GiB]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis (cost estimate, not sound) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|---|
| 520 B | 0.04 | 0.12 | 0.17 | 0.08 | 0.04 | 0.04 | 0.04 |
| 2.3 KiB | 0.05 | 0.29 | 0.53 | 0.34 | 0.05 | 0.05 | 0.05 |
| 3.9 KiB | 0.05 | 0.43 | 0.90 | 0.55 | 0.06 | 0.05 | 0.05 |
| 9.8 KiB | 0.06 | 1.10 | 2.00 | 1.31 | 0.08 | 0.07 | 0.06 |
| 16.0 KiB | 0.08 | 2.37 | 3.28 | 2.12 | 0.11 | 0.09 | 0.08 |
| 64.0 KiB | 0.17 | 36.35 | 12.89 | 8.18 | 0.29 | 0.20 | 0.17 |
| 83.6 KiB | 0.20 | 64.25 | n/a | n/a | 0.34 | 0.25 | 0.20 |
| 89.4 KiB | 0.20 | 71.14 | n/a | n/a | 0.35 | 0.26 | 0.20 |
| 185.8 KiB | 0.38 | 301.10 | n/a | n/a | 0.69 | 0.52 | 0.38 |
| 256.0 KiB | 0.54 | n/a | n/a | n/a | 1.08 | 0.65 | 0.53 |
| 1 MiB | 1.94 | n/a | n/a | n/a | 3.72 | 2.41 | 1.95 |

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
| 83.6 KiB | 107089 | w_l | 106625 | 100.00% | 9.1 | 9.1 | 306.6 | 7.2 | 1.1 | 1.17 | 7.0 |
| 89.4 KiB | 114514 | w_l | 114023 | 100.00% | 10.7 | 10.7 | 313.3 | 6.0 | 1.1 | 1.26 | 7.5 |
| 185.8 KiB | 237905 | w_l | 236940 | 100.00% | 16.2 | 16.2 | 668.4 | 10.2 | 1.1 | 2.61 | 15.6 |
| 256.0 KiB | 327761 | w_l | 326465 | 100.00% | 22.0 | 22.0 | 335.2 | 11.4 | 1.1 | 3.70 | 21.5 |
| 1 MiB | 1310801 | w_l | 1305745 | 100.00% | 122.6 | 122.6 | 1300.8 | 31.5 | 1.1 | 14.54 | 86.1 |

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
| 83.6 KiB | 137465 | w_l, w_o | 185754 | 98.51% | 24.2 | 322.5 | 306.6 | 5.9 | 1.3 | 1.72 | 17.8 |
| 89.4 KiB | 146998 | w_l, w_4 | 201717 | 83.89% | 51.6 | 358.0 | 313.3 | 6.4 | 1.3 | 1.80 | 19.0 |
| 185.8 KiB | 305416 | w_l, w_o | 412768 | 98.51% | 43.5 | 667.9 | 668.4 | 8.5 | 1.3 | 3.88 | 39.5 |
| 256.0 KiB | 420780 | w_l, w_r | 568702 | 98.51% | 64.1 | 351.6 | 335.2 | 12.7 | 1.3 | 5.25 | 54.4 |
| 1 MiB | 1682877 | w_l, w_r | 2274650 | 98.51% | 239.4 | 1424.7 | 1300.8 | 32.0 | 1.3 | 20.51 | 217.6 |

## Age check (10 committed bytes)

| Backend | Row | Constraints / gates | Prover [ms] | Verifier [ms] | Proof [kB] |
|---|---|---|---|---|---|
| bb | age_blake3 | 10395 | 122.5 | 0.2 | 14.7 |
| bb | age_blake3_bb | 17416 | 150.9 | 0.3 | 14.7 |
| bb | age_keccak256 | 49091 | 352.1 | 0.3 | 14.7 |
| bb | age_keccak256_bb | 40626 | 330.9 | 0.3 | 14.7 |
| bb | age_nocom | 5559 | 68.3 | 0.1 | 14.7 |
| bb | age_poseidon2 | 5640 | 71.4 | 0.1 | 14.7 |
| bb | age_sha256 | 36000 | 238.9 | 0.2 | 14.7 |
| provekit | age_artemis | 2695 | 43.0 | 16.3 | 339.4 |
| provekit | age_blake3 | 14267 | 65.7 | 20.3 | 358.1 |
| provekit | age_keccak256 | 158349 | 459.4 | 78.6 | 499.8 |
| provekit | age_nocom | 2693 | 43.2 | 15.5 | 337.9 |
| provekit | age_poseidon2 | 2957 | 43.7 | 15.8 | 338.3 |
| provekit | age_sha256 | 15934 | 66.7 | 22.6 | 359.1 |
