# TLSNotary / Artemis benchmark: summary

Medians over the runs in the CSV. Prover time excludes ACVM witness generation, key generation and setup; the Artemis rows include the IPA opening (ProveKit) or the KZG opening (Barretenberg estimate). The LegoSNARK row is the no-commitment row plus the CP_link prover on the wire columns of a real Barretenberg proof (same circuit; the CSV rows hold the sum with a `bb prove` run of their own session).

## ProveKit (WHIR, pairing-free): Prover time [s]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Artemis |
|---|---|---|---|---|---|
| 520 B | 0.023 | 0.120 | 0.406 | 0.127 | 0.027 |
| 2.3 KiB | 0.025 | 0.365 | 1.462 | 0.423 | 0.032 |
| 3.9 KiB | 0.027 | 0.636 | 2.635 | 0.662 | 0.035 |
| 9.8 KiB | 0.036 | 1.309 | 5.712 | 1.320 | 0.050 |
| 16.0 KiB | 0.048 | 1.620 | 12.429 | 1.632 | 0.066 |
| 64.0 KiB | 0.107 | 7.195 | n/a | 6.262 | 0.151 |

## ProveKit (WHIR, pairing-free): Constraints (ProveKit R1CS) / gates (bb circuit size)

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Artemis |
|---|---|---|---|---|---|
| 520 B | 776 | 96239 | 428192 | 102121 | 794 |
| 2.3 KiB | 2656 | 361598 | 1687558 | 358720 | 2735 |
| 3.9 KiB | 4256 | 553673 | 2764902 | 546838 | 4387 |
| 9.8 KiB | 10256 | 1276503 | 6629006 | 1167546 | 10580 |
| 16.0 KiB | 16640 | 2026804 | 10716074 | 1820790 | 17170 |
| 64.0 KiB | 65792 | 7802164 | n/a | 6893430 | 67908 |

## ProveKit (WHIR, pairing-free): Verifier time [ms]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Artemis |
|---|---|---|---|---|---|
| 520 B | 6.2 | 24.7 | 64.6 | 20.7 | 7.6 |
| 2.3 KiB | 6.6 | 76.1 | 225.9 | 62.2 | 8.1 |
| 3.9 KiB | 6.9 | 105.2 | 358.9 | 88.5 | 8.8 |
| 9.8 KiB | 8.1 | 234.8 | 873.3 | 182.4 | 10.1 |
| 16.0 KiB | 9.3 | 356.9 | 1454.1 | 264.9 | 11.9 |
| 64.0 KiB | 16.2 | 1448.8 | n/a | 990.0 | 21.4 |

## ProveKit (WHIR, pairing-free): Proof size [kB]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Artemis |
|---|---|---|---|---|---|
| 520 B | 336.4 | 447.3 | 522.7 | 445.4 | 338.0 |
| 2.3 KiB | 337.1 | 522.2 | 591.6 | 523.0 | 338.3 |
| 3.9 KiB | 338.5 | 537.9 | 607.1 | 536.4 | 339.9 |
| 9.8 KiB | 357.5 | 590.0 | 623.2 | 590.1 | 356.2 |
| 16.0 KiB | 391.2 | 591.5 | 679.3 | 590.0 | 391.6 |
| 64.0 KiB | 444.0 | 623.5 | n/a | 624.1 | 446.3 |

## ProveKit (WHIR, pairing-free): Prover peak memory [GiB]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Artemis |
|---|---|---|---|---|---|
| 520 B | 0.02 | 0.22 | 0.73 | 0.20 | 0.02 |
| 2.3 KiB | 0.02 | 0.78 | 2.54 | 0.63 | 0.02 |
| 3.9 KiB | 0.02 | 1.22 | 4.90 | 1.18 | 0.02 |
| 9.8 KiB | 0.03 | 3.17 | 10.21 | 2.41 | 0.04 |
| 16.0 KiB | 0.05 | 4.10 | 14.32 | 2.63 | 0.06 |
| 64.0 KiB | 0.18 | 16.09 | n/a | 10.28 | 0.21 |

## ProveKit (WHIR, pairing-free): prover-time ratio to Artemis

| Payload | SHA-256 / Artemis | Keccak-256 / Artemis | BLAKE3 / Artemis | Poseidon / Artemis | Artemis / No commitment |
|---|---|---|---|---|---|
| 520 B | 4.5x | 15.1x | 4.8x | n/a | 1.2x |
| 2.3 KiB | 11.4x | 45.8x | 13.3x | n/a | 1.3x |
| 3.9 KiB | 18.2x | 75.5x | 19.0x | n/a | 1.3x |
| 9.8 KiB | 26.2x | 114.4x | 26.4x | n/a | 1.4x |
| 16.0 KiB | 24.5x | 188.2x | 24.7x | n/a | 1.4x |
| 64.0 KiB | 47.5x | n/a | 41.4x | n/a | 1.4x |

## Barretenberg (UltraHonk, KZG): Prover time [s]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Artemis (cost estimate) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|
| 520 B | 0.010 | 0.207 | 0.439 | 0.122 | 0.012 | 0.011 |
| 2.3 KiB | 0.027 | 0.668 | 1.654 | 0.719 | 0.030 | 0.028 |
| 3.9 KiB | 0.044 | 0.889 | 2.741 | 1.208 | 0.050 | 0.046 |
| 9.8 KiB | 0.065 | 2.693 | 6.210 | 2.671 | 0.078 | 0.067 |
| 16.0 KiB | 0.100 | 5.191 | 10.384 | 4.564 | 0.117 | 0.102 |
| 64.0 KiB | 0.293 | 40.130 | 39.495 | 16.915 | 0.349 | 0.297 |

## Barretenberg (UltraHonk, KZG): Constraints (ProveKit R1CS) / gates (bb circuit size)

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Artemis (cost estimate) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|
| 520 B | 731 | 39659 | 74067 | 20701 | 919 | 731 |
| 2.3 KiB | 3081 | 158149 | 323967 | 193699 | 3936 | 3081 |
| 3.9 KiB | 5081 | 260255 | 538127 | 317217 | 6504 | 5081 |
| 9.8 KiB | 12581 | 644144 | 1323847 | 791500 | 16133 | 12581 |
| 16.0 KiB | 20561 | 1052522 | 2163051 | 1290322 | 26378 | 20561 |
| 64.0 KiB | 82001 | 4189226 | 8609263 | 5162242 | 105260 | 82001 |

## Barretenberg (UltraHonk, KZG): Verifier time [ms]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Artemis (cost estimate) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|
| 520 B | 0.2 | 1.2 | 0.8 | 0.8 | 1.1 | 0.8 |
| 2.3 KiB | 0.6 | 1.3 | 1.2 | 1.1 | 1.3 | 1.1 |
| 3.9 KiB | 0.8 | 1.4 | 1.1 | 1.0 | 1.7 | 1.1 |
| 9.8 KiB | 1.1 | 1.4 | 1.7 | 1.1 | 1.9 | 1.2 |
| 16.0 KiB | 1.2 | 1.5 | 1.4 | 1.1 | 2.0 | 1.4 |
| 64.0 KiB | 1.3 | 4.4 | 5.6 | 3.5 | 2.0 | 1.5 |

## Barretenberg (UltraHonk, KZG): Proof size [kB]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Artemis (cost estimate) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|
| 520 B | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 2.3 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 3.9 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 9.8 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 16.0 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 64.0 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |

## Barretenberg (UltraHonk, KZG): Prover peak memory [GiB]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Artemis (cost estimate) | LegoSNARK CP_link (cost estimate) |
|---|---|---|---|---|---|---|
| 520 B | 0.01 | 0.10 | 0.17 | 0.05 | 0.01 | 0.01 |
| 2.3 KiB | 0.02 | 0.41 | 0.68 | 0.53 | 0.02 | 0.02 |
| 3.9 KiB | 0.02 | 0.61 | 1.21 | 0.93 | 0.02 | 0.02 |
| 9.8 KiB | 0.04 | 2.20 | 2.62 | 2.28 | 0.04 | 0.04 |
| 16.0 KiB | 0.05 | 4.77 | 4.27 | 3.45 | 0.06 | 0.05 |
| 64.0 KiB | 0.20 | 16.77 | 13.93 | 12.06 | 0.24 | 0.20 |

## Barretenberg: LegoSNARK CP_link, no-commitment circuit, byte cells linked (the plotted row)

Link prover, Pedersen commitment and link verifier are in-process medians. Overhead columns: prover time over no commitment. LegoSNARK: the link prover (the circuit is the no-commitment circuit) or, for the packed variant, the difference of the `bb prove` medians of the two LegoSNARK circuits (measured back to back) plus the link prover. Artemis: `artemis_estimate` + KZG opening - `nocom` (measured back to back).

| Payload | Gates | Linked wires | Non-zero cells in the MSM | of at most 16 bits | Link prover [ms] | LegoSNARK overhead [ms] | Artemis estimate overhead [ms] | Pedersen commit [ms] (not counted) | Link verifier [ms] | Setup [s] | Keys [MB] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 520 B | 731 | w_l | 699 | 99.57% | 0.9 | 0.9 | 2.3 | 0.7 | 0.4 | 0.01 | 0.0 |
| 2.3 KiB | 3081 | w_l | 3044 | 99.90% | 1.1 | 1.1 | 3.7 | 0.7 | 0.4 | 0.03 | 0.2 |
| 3.9 KiB | 5081 | w_l | 5024 | 99.94% | 1.5 | 1.5 | 6.1 | 0.9 | 0.4 | 0.04 | 0.3 |
| 9.8 KiB | 12581 | w_l | 12491 | 99.98% | 1.5 | 1.5 | 12.1 | 1.2 | 0.4 | 0.10 | 0.8 |
| 16.0 KiB | 20561 | w_l | 20461 | 99.99% | 2.4 | 2.4 | 17.0 | 1.5 | 0.5 | 0.15 | 1.4 |
| 64.0 KiB | 82001 | w_l | 81651 | 100.00% | 3.1 | 3.1 | 55.3 | 2.5 | 0.4 | 0.62 | 5.4 |

## Barretenberg: LegoSNARK CP_link, variant that packs in the circuit, packed cells linked

Link prover, Pedersen commitment and link verifier are in-process medians. Overhead columns: prover time over no commitment. LegoSNARK: the link prover (the circuit is the no-commitment circuit) or, for the packed variant, the difference of the `bb prove` medians of the two LegoSNARK circuits (measured back to back) plus the link prover. Artemis: `artemis_estimate` + KZG opening - `nocom` (measured back to back).

| Payload | Gates | Linked wires | Non-zero cells in the MSM | of at most 16 bits | Link prover [ms] | LegoSNARK overhead [ms] | Artemis estimate overhead [ms] | Pedersen commit [ms] (not counted) | Link verifier [ms] | Setup [s] | Keys [MB] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 520 B | 915 | w_l, w_o | 1229 | 97.97% | 2.0 | 2.6 | 2.3 | 0.6 | 0.5 | 0.01 | 0.1 |
| 2.3 KiB | 3933 | w_l | 3895 | 97.92% | 1.6 | 2.2 | 3.7 | 0.7 | 0.4 | 0.02 | 0.3 |
| 3.9 KiB | 6501 | w_l, w_r | 8764 | 98.46% | 2.9 | 7.9 | 6.1 | 0.8 | 0.4 | 0.06 | 0.8 |
| 9.8 KiB | 16129 | w_l, w_o | 21774 | 98.48% | 3.9 | 13.8 | 12.1 | 1.2 | 0.4 | 0.13 | 2.1 |
| 16.0 KiB | 26375 | w_l | 26257 | 97.97% | 2.7 | 17.1 | 17.0 | 1.2 | 0.4 | 0.11 | 1.7 |
| 64.0 KiB | 105256 | w_l, w_o | 142250 | 98.51% | 8.5 | 69.7 | 55.3 | 2.5 | 0.5 | 0.83 | 13.6 |

## Age check (10 committed bytes)

| Backend | Row | Constraints / gates | Prover [ms] | Verifier [ms] | Proof [kB] |
|---|---|---|---|---|---|
| bb | age_blake3 | 10395 | 70.7 | 0.6 | 14.7 |
| bb | age_blake3_bb | 17416 | 97.6 | 0.7 | 14.7 |
| bb | age_keccak256 | 49091 | 212.5 | 0.8 | 14.7 |
| bb | age_keccak256_bb | 40626 | 202.1 | 0.7 | 14.7 |
| bb | age_nocom | 5559 | 45.9 | 0.3 | 14.7 |
| bb | age_sha256 | 36000 | 151.1 | 0.8 | 14.7 |
| provekit | age_artemis | 2695 | 24.7 | 7.0 | 338.6 |
| provekit | age_blake3 | 14267 | 38.8 | 8.8 | 358.2 |
| provekit | age_keccak256 | 158349 | 214.3 | 33.6 | 499.4 |
| provekit | age_nocom | 2693 | 24.5 | 6.5 | 339.1 |
| provekit | age_sha256 | 15934 | 38.0 | 9.2 | 357.7 |
