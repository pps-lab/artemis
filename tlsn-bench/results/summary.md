# TLSNotary / Artemis benchmark: summary

Medians over the runs in the CSV. Prover time excludes ACVM witness generation, key generation and setup; the Artemis rows include the IPA opening (ProveKit) or the KZG opening (Barretenberg estimate).

## ProveKit (WHIR, pairing-free): Prover time [s]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis |
|---|---|---|---|---|---|---|
| 520 B | 0.023 | 0.120 | 0.406 | 0.127 | 0.025 | 0.027 |
| 2.3 KiB | 0.025 | 0.365 | 1.462 | 0.423 | 0.029 | 0.032 |
| 3.9 KiB | 0.027 | 0.636 | 2.635 | 0.662 | 0.031 | 0.035 |
| 9.8 KiB | 0.036 | 1.309 | 5.712 | 1.320 | 0.055 | 0.050 |
| 16.0 KiB | 0.048 | 1.620 | 12.429 | 1.632 | 0.065 | 0.066 |
| 64.0 KiB | 0.107 | 7.195 | n/a | 6.262 | 0.184 | 0.151 |
| 83.6 KiB | 0.116 | n/a | n/a | n/a | 0.254 | 0.165 |
| 89.4 KiB | 0.118 | n/a | n/a | n/a | 0.268 | 0.168 |
| 185.8 KiB | 0.208 | n/a | n/a | n/a | 0.502 | 0.298 |
| 256.0 KiB | 0.314 | n/a | n/a | n/a | 0.630 | 0.452 |
| 1 MiB | 1.179 | n/a | n/a | n/a | 2.298 | 1.637 |

## ProveKit (WHIR, pairing-free): Constraints (ProveKit R1CS) / gates (bb circuit size)

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis |
|---|---|---|---|---|---|---|
| 520 B | 776 | 96239 | 428192 | 102121 | 1911 | 794 |
| 2.3 KiB | 2656 | 361598 | 1687558 | 358720 | 6084 | 2735 |
| 3.9 KiB | 4256 | 553673 | 2764902 | 546838 | 9409 | 4387 |
| 9.8 KiB | 10256 | 1276503 | 6629006 | 1167546 | 22857 | 10580 |
| 16.0 KiB | 16640 | 2026804 | 10716074 | 1820790 | 37255 | 17170 |
| 64.0 KiB | 65792 | 7802164 | n/a | 6893430 | 146585 | 67908 |
| 83.6 KiB | 85862 | n/a | n/a | n/a | 191854 | 88625 |
| 89.4 KiB | 91802 | n/a | n/a | n/a | 204681 | 94757 |
| 185.8 KiB | 190515 | n/a | n/a | n/a | 425433 | 196654 |
| 256.0 KiB | 262400 | n/a | n/a | n/a | 585563 | 270858 |
| 1 MiB | 1048832 | n/a | n/a | n/a | 2341501 | 1082659 |

## ProveKit (WHIR, pairing-free): Verifier time [ms]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis |
|---|---|---|---|---|---|---|
| 520 B | 6.2 | 24.7 | 64.6 | 20.7 | 7.4 | 7.6 |
| 2.3 KiB | 6.6 | 76.1 | 225.9 | 62.2 | 9.1 | 8.1 |
| 3.9 KiB | 6.9 | 105.2 | 358.9 | 88.5 | 10.3 | 8.8 |
| 9.8 KiB | 8.1 | 234.8 | 873.3 | 182.4 | 16.2 | 10.1 |
| 16.0 KiB | 9.3 | 356.9 | 1454.1 | 264.9 | 21.1 | 11.9 |
| 64.0 KiB | 16.2 | 1448.8 | n/a | 990.0 | 61.2 | 21.4 |
| 83.6 KiB | 17.7 | n/a | n/a | n/a | 79.4 | 23.3 |
| 89.4 KiB | 18.9 | n/a | n/a | n/a | 84.5 | 24.1 |
| 185.8 KiB | 32.3 | n/a | n/a | n/a | 169.9 | 40.4 |
| 256.0 KiB | 40.4 | n/a | n/a | n/a | 224.9 | 54.2 |
| 1 MiB | 135.0 | n/a | n/a | n/a | 866.0 | 174.2 |

## ProveKit (WHIR, pairing-free): Proof size [kB]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis |
|---|---|---|---|---|---|---|
| 520 B | 336.4 | 447.3 | 522.7 | 445.4 | 337.6 | 338.0 |
| 2.3 KiB | 337.1 | 522.2 | 591.6 | 523.0 | 337.5 | 338.3 |
| 3.9 KiB | 338.5 | 537.9 | 607.1 | 536.4 | 339.5 | 339.9 |
| 9.8 KiB | 357.5 | 590.0 | 623.2 | 590.1 | 391.3 | 356.2 |
| 16.0 KiB | 391.2 | 591.5 | 679.3 | 590.0 | 389.8 | 391.6 |
| 64.0 KiB | 444.0 | 623.5 | n/a | 624.1 | 446.8 | 446.3 |
| 83.6 KiB | 444.8 | n/a | n/a | n/a | 498.4 | 446.3 |
| 89.4 KiB | 444.4 | n/a | n/a | n/a | 498.8 | 447.5 |
| 185.8 KiB | 498.2 | n/a | n/a | n/a | 519.8 | 499.5 |
| 256.0 KiB | 520.3 | n/a | n/a | n/a | 522.1 | 522.8 |
| 1 MiB | 590.6 | n/a | n/a | n/a | 589.1 | 591.3 |

## ProveKit (WHIR, pairing-free): Prover peak memory [GiB]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis |
|---|---|---|---|---|---|---|
| 520 B | 0.02 | 0.22 | 0.73 | 0.20 | 0.02 | 0.02 |
| 2.3 KiB | 0.02 | 0.78 | 2.54 | 0.63 | 0.04 | 0.02 |
| 3.9 KiB | 0.02 | 1.22 | 4.90 | 1.18 | 0.05 | 0.02 |
| 9.8 KiB | 0.03 | 3.17 | 10.21 | 2.41 | 0.10 | 0.04 |
| 16.0 KiB | 0.05 | 4.10 | 14.32 | 2.63 | 0.14 | 0.06 |
| 64.0 KiB | 0.18 | 16.09 | n/a | 10.28 | 0.38 | 0.21 |
| 83.6 KiB | 0.22 | n/a | n/a | n/a | 0.59 | 0.25 |
| 89.4 KiB | 0.22 | n/a | n/a | n/a | 0.64 | 0.26 |
| 185.8 KiB | 0.46 | n/a | n/a | n/a | 1.36 | 0.36 |
| 256.0 KiB | 0.70 | n/a | n/a | n/a | 1.74 | 0.61 |
| 1 MiB | 2.15 | n/a | n/a | n/a | 6.38 | 2.26 |

## ProveKit (WHIR, pairing-free): prover-time ratio to Artemis

| Payload | SHA-256 / Artemis | Keccak-256 / Artemis | BLAKE3 / Artemis | Poseidon2 / Artemis | Artemis / No commitment |
|---|---|---|---|---|---|
| 520 B | 4.5x | 15.1x | 4.8x | 0.9x | 1.2x |
| 2.3 KiB | 11.4x | 45.8x | 13.3x | 0.9x | 1.3x |
| 3.9 KiB | 18.2x | 75.5x | 19.0x | 0.9x | 1.3x |
| 9.8 KiB | 26.2x | 114.4x | 26.4x | 1.1x | 1.4x |
| 16.0 KiB | 24.5x | 188.2x | 24.7x | 1.0x | 1.4x |
| 64.0 KiB | 47.5x | n/a | 41.4x | 1.2x | 1.4x |
| 83.6 KiB | n/a | n/a | n/a | 1.5x | 1.4x |
| 89.4 KiB | n/a | n/a | n/a | 1.6x | 1.4x |
| 185.8 KiB | n/a | n/a | n/a | 1.7x | 1.4x |
| 256.0 KiB | n/a | n/a | n/a | 1.4x | 1.4x |
| 1 MiB | n/a | n/a | n/a | 1.4x | 1.4x |

## Barretenberg (UltraHonk, KZG): Prover time [s]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis (cost estimate, not sound) |
|---|---|---|---|---|---|---|
| 520 B | 0.010 | 0.207 | 0.439 | 0.122 | 0.019 | 0.012 |
| 2.3 KiB | 0.027 | 0.668 | 1.654 | 0.719 | 0.053 | 0.030 |
| 3.9 KiB | 0.044 | 0.889 | 2.741 | 1.208 | 0.074 | 0.050 |
| 9.8 KiB | 0.065 | 2.693 | 6.210 | 2.671 | 0.124 | 0.078 |
| 16.0 KiB | 0.100 | 5.191 | 10.384 | 4.564 | 0.190 | 0.117 |
| 64.0 KiB | 0.293 | 40.130 | 39.495 | 16.915 | 0.605 | 0.349 |
| 83.6 KiB | 0.321 | n/a | n/a | n/a | 0.680 | 0.548 |
| 89.4 KiB | 0.333 | n/a | n/a | n/a | 0.696 | 0.559 |
| 185.8 KiB | 0.626 | n/a | n/a | n/a | 1.320 | 1.050 |
| 256.0 KiB | 0.988 | n/a | n/a | n/a | 2.080 | 1.186 |
| 1 MiB | 3.609 | n/a | n/a | n/a | 7.552 | 4.318 |

## Barretenberg (UltraHonk, KZG): Constraints (ProveKit R1CS) / gates (bb circuit size)

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis (cost estimate, not sound) |
|---|---|---|---|---|---|---|
| 520 B | 731 | 39659 | 74067 | 20701 | 1357 | 919 |
| 2.3 KiB | 3081 | 158149 | 323967 | 193699 | 5834 | 3936 |
| 3.9 KiB | 5081 | 260255 | 538127 | 317217 | 9716 | 6504 |
| 9.8 KiB | 12581 | 644144 | 1323847 | 791500 | 24017 | 16133 |
| 16.0 KiB | 20561 | 1052522 | 2163051 | 1290322 | 39299 | 26378 |
| 64.0 KiB | 82001 | 4189226 | 8609263 | 5162242 | 156724 | 105260 |
| 83.6 KiB | 107089 | n/a | n/a | n/a | 204702 | 137469 |
| 89.4 KiB | 114514 | n/a | n/a | n/a | 218907 | 147002 |
| 185.8 KiB | 237905 | n/a | n/a | n/a | 454778 | 305420 |
| 256.0 KiB | 327761 | n/a | n/a | n/a | 626570 | 420784 |
| 1 MiB | 1310801 | n/a | n/a | n/a | 2506028 | 1682880 |

## Barretenberg (UltraHonk, KZG): Verifier time [ms]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis (cost estimate, not sound) |
|---|---|---|---|---|---|---|
| 520 B | 0.2 | 1.2 | 0.8 | 0.8 | 0.2 | 1.1 |
| 2.3 KiB | 0.6 | 1.3 | 1.2 | 1.1 | 0.9 | 1.3 |
| 3.9 KiB | 0.8 | 1.4 | 1.1 | 1.0 | 1.0 | 1.7 |
| 9.8 KiB | 1.1 | 1.4 | 1.7 | 1.1 | 1.3 | 1.9 |
| 16.0 KiB | 1.2 | 1.5 | 1.4 | 1.1 | 1.3 | 2.0 |
| 64.0 KiB | 1.3 | 4.4 | 5.6 | 3.5 | 1.3 | 2.0 |
| 83.6 KiB | 0.9 | n/a | n/a | n/a | 1.0 | 1.8 |
| 89.4 KiB | 1.0 | n/a | n/a | n/a | 1.1 | 1.8 |
| 185.8 KiB | 1.1 | n/a | n/a | n/a | 1.2 | 2.1 |
| 256.0 KiB | 1.2 | n/a | n/a | n/a | 1.2 | 1.9 |
| 1 MiB | 1.2 | n/a | n/a | n/a | 1.2 | 2.0 |

## Barretenberg (UltraHonk, KZG): Proof size [kB]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis (cost estimate, not sound) |
|---|---|---|---|---|---|---|
| 520 B | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 2.3 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 3.9 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 9.8 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 16.0 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 64.0 KiB | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 | 14.7 |
| 83.6 KiB | 14.7 | n/a | n/a | n/a | 14.7 | 14.7 |
| 89.4 KiB | 14.7 | n/a | n/a | n/a | 14.7 | 14.7 |
| 185.8 KiB | 14.7 | n/a | n/a | n/a | 14.7 | 14.7 |
| 256.0 KiB | 14.7 | n/a | n/a | n/a | 14.7 | 14.7 |
| 1 MiB | 14.7 | n/a | n/a | n/a | 14.7 | 14.7 |

## Barretenberg (UltraHonk, KZG): Prover peak memory [GiB]

| Payload | No commitment | SHA-256 (as shipped) | Keccak-256 | BLAKE3 | Poseidon2 (faster of t16 / native) | Artemis (cost estimate, not sound) |
|---|---|---|---|---|---|---|
| 520 B | 0.01 | 0.10 | 0.17 | 0.05 | 0.01 | 0.01 |
| 2.3 KiB | 0.02 | 0.41 | 0.68 | 0.53 | 0.02 | 0.02 |
| 3.9 KiB | 0.02 | 0.61 | 1.21 | 0.93 | 0.03 | 0.02 |
| 9.8 KiB | 0.04 | 2.20 | 2.62 | 2.28 | 0.06 | 0.04 |
| 16.0 KiB | 0.05 | 4.77 | 4.27 | 3.45 | 0.09 | 0.06 |
| 64.0 KiB | 0.20 | 16.77 | 13.93 | 12.06 | 0.39 | 0.24 |
| 83.6 KiB | 0.25 | n/a | n/a | n/a | 0.47 | 0.37 |
| 89.4 KiB | 0.26 | n/a | n/a | n/a | 0.49 | 0.39 |
| 185.8 KiB | 0.52 | n/a | n/a | n/a | 1.01 | 0.79 |
| 256.0 KiB | 0.82 | n/a | n/a | n/a | 1.58 | 1.00 |
| 1 MiB | 3.01 | n/a | n/a | n/a | 5.42 | 3.76 |

## Age check (10 committed bytes)

| Backend | Row | Constraints / gates | Prover [ms] | Verifier [ms] | Proof [kB] |
|---|---|---|---|---|---|
| bb | age_blake3 | 10395 | 70.7 | 0.6 | 14.7 |
| bb | age_blake3_bb | 17416 | 97.6 | 0.7 | 14.7 |
| bb | age_keccak256 | 49091 | 212.5 | 0.8 | 14.7 |
| bb | age_keccak256_bb | 40626 | 202.1 | 0.7 | 14.7 |
| bb | age_nocom | 5559 | 45.9 | 0.3 | 14.7 |
| bb | age_poseidon2 | 5640 | 49.1 | 0.5 | 14.7 |
| bb | age_sha256 | 36000 | 151.1 | 0.8 | 14.7 |
| provekit | age_artemis | 2695 | 24.7 | 7.0 | 338.6 |
| provekit | age_blake3 | 14267 | 38.8 | 8.8 | 358.2 |
| provekit | age_keccak256 | 158349 | 214.3 | 33.6 | 499.4 |
| provekit | age_nocom | 2693 | 24.5 | 6.5 | 339.1 |
| provekit | age_poseidon2 | 2957 | 25.2 | 6.7 | 338.4 |
| provekit | age_sha256 | 15934 | 38.0 | 9.2 | 357.7 |
