use halo2_proofs::halo2curves::pasta::Fp;
use ff::Field;

/// Lazy representation of the zkFFT b matrix that exploits geometric sequence structure.
///
/// The b matrix has the property that b[i][j] = omega^(i*j) * row_start[i], which means
/// each row is a geometric sequence with ratio omega^i. This allows us to:
/// 1. Store only omega_base[i] = omega^i (n elements) instead of full n×n matrix
/// 2. Compute elements on-demand via iterative multiplication
/// 3. Update the representation during folding without materialization
///
/// Memory usage: 2*n*32 bytes (~2MB for n=32,768) vs n²*32 bytes (~32GB for n=32,768)
#[derive(Clone, Debug)]
pub struct LazyBMatrix {
    /// omega^i for each row i (NEVER changes during folding)
    pub omega_base: Vec<Fp>,
    /// Starting value for row i (updated each fold)
    pub row_start: Vec<Fp>,
    /// Current row size (halves each fold)
    pub row_size: usize,
}

impl LazyBMatrix {
    /// Create a new lazy b matrix from omega and size n.
    ///
    /// Initially:
    /// - omega_base[i] = omega^i
    /// - row_start[i] = 1 for all i
    /// - row_size = n
    ///
    /// This represents the matrix where b[i][j] = omega^(i*j)
    pub fn new(omega: Fp, n: usize) -> Self {
        // Precompute omega^i for all i (O(n) multiplications instead of n² exponentiations)
        let mut omega_base = Vec::with_capacity(n);
        omega_base.push(Fp::ONE);
        for i in 1..n {
            omega_base.push(omega_base[i - 1] * omega);
        }

        LazyBMatrix {
            omega_base,
            row_start: vec![Fp::ONE; n],
            row_size: n,
        }
    }

    /// Get the number of rows in the matrix (equals number of omega_base elements)
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.omega_base.len()
    }

    /// Get element b[i][j] on-demand.
    ///
    /// Returns: omega_base[i]^j * row_start[i]
    pub fn get(&self, i: usize, j: usize) -> Fp {
        let mut result = self.row_start[i];
        for _ in 0..j {
            result *= self.omega_base[i];
        }
        result
    }

    /// Compute inner product sum(a[j] * b[i][j]) without materializing row i.
    ///
    /// Uses iterative multiplication to compute b[i][j] on-the-fly:
    /// - b[i][0] = row_start[i]
    /// - b[i][j+1] = b[i][j] * omega_base[i]
    ///
    /// Complexity: O(n) field multiplications, no memory allocation
    pub fn inner_product_with(&self, a: &[Fp], row: usize) -> Fp {
        let mut sum = Fp::ZERO;
        let mut current = self.row_start[row];
        for &a_j in a.iter() {
            sum += a_j * current;
            current *= self.omega_base[row];
        }
        sum
    }

    /// Compute inner product sum(a[j] * b[i][offset+j]) without materialization.
    ///
    /// This is used for cross inner products with split vectors during folding.
    /// For example, if we split row i in half, b2[i] is the second half starting at offset=n/2.
    ///
    /// Returns: sum(a[j] * b[i][offset+j]) where b[i][offset+j] = omega^(i*(offset+j)) * row_start[i]
    ///
    /// Complexity: O(n) + O(log offset) for the power computation
    pub fn inner_product_with_offset(&self, a: &[Fp], row: usize, offset: usize) -> Fp {
        // b[i][offset+j] = omega^(i*(offset+j)) * row_start[i]
        //                = omega^(i*offset) * omega^(i*j) * row_start[i]
        let mut sum = Fp::ZERO;
        let omega_i_offset = self.omega_base[row].pow([offset as u64]);
        let mut current = omega_i_offset * self.row_start[row];
        for &a_j in a.iter() {
            sum += a_j * current;
            current *= self.omega_base[row];
        }
        sum
    }

    /// Compute ALL inner products using FFT: O(k log k) instead of O(k * n)
    ///
    /// Returns: result[i] = sum_j a[j] * b[i][j] = sum_j a[j] * row_start[i] * omega^(i*j)
    ///                    = row_start[i] * DFT(a_padded)[i]
    ///
    /// Key insight: omega is a k-th root of unity (where k = original matrix size),
    /// NOT an n-th root of unity (where n = current vector size after folding).
    /// So we must zero-pad `a` to size k before computing FFT, ensuring omega^k = 1.
    pub fn compute_all_inner_products_fft(&self, a: &[Fp]) -> Vec<Fp> {
        use halo2_proofs::arithmetic::best_fft;
        use halo2_proofs::fft::recursive::FFTData;

        let k = self.omega_base.len(); // Original matrix size where omega^k = 1
        assert!(k.is_power_of_two(), "FFT requires power of 2 size");
        let log_k = k.trailing_zeros();

        let omega = self.omega_base[1];
        let omega_inv = omega.invert().unwrap();

        let fft_data = FFTData::new(k, omega, omega_inv);

        // Zero-pad input to original size k so FFT is correct (omega^k = 1)
        let mut dft_a = vec![Fp::ZERO; k];
        dft_a[..a.len()].copy_from_slice(a);
        best_fft(&mut dft_a, omega, log_k, &fft_data, false);

        // dft_a[i] = sum_j a[j] * omega^(i*j) for all i in 0..k
        let mut result = Vec::with_capacity(k);
        for i in 0..k {
            result.push(self.row_start[i] * dft_a[i]);
        }

        result
    }

    /// Compute ALL inner products with offset using FFT: O(k log k) instead of O(k * n)
    ///
    /// Returns: result[i] = sum_j a[j] * b[i][offset+j]
    ///                    = sum_j a[j] * row_start[i] * omega^(i*(offset+j))
    ///                    = row_start[i] * omega^(i*offset) * DFT(a_padded)[i]
    ///
    /// Zero-pads `a` to size k (original matrix size) before FFT, same as
    /// compute_all_inner_products_fft, since omega is a k-th root of unity.
    pub fn compute_all_inner_products_with_offset_fft(&self, a: &[Fp], offset: usize) -> Vec<Fp> {
        use halo2_proofs::arithmetic::best_fft;
        use halo2_proofs::fft::recursive::FFTData;

        let k = self.omega_base.len(); // Original matrix size where omega^k = 1
        assert!(k.is_power_of_two(), "FFT requires power of 2 size");
        let log_k = k.trailing_zeros();

        let omega = self.omega_base[1];
        let omega_inv = omega.invert().unwrap();

        let fft_data = FFTData::new(k, omega, omega_inv);

        // Zero-pad input to original size k so FFT is correct (omega^k = 1)
        let mut dft_a = vec![Fp::ZERO; k];
        dft_a[..a.len()].copy_from_slice(a);
        best_fft(&mut dft_a, omega, log_k, &fft_data, false);

        // dft_a[i] = sum_j a[j] * omega^(i*j), multiply by row_start[i] * omega^(i*offset)
        let mut result = Vec::with_capacity(k);
        for i in 0..k {
            let twiddle = self.omega_base[i].pow([offset as u64]);
            result.push(self.row_start[i] * twiddle * dft_a[i]);
        }

        result
    }

    /// Fold the matrix using challenge e, updating the lazy representation.
    ///
    /// Mathematical operation:
    /// - Split each row i into halves: b1[i] (first half), b2[i] (second half)
    /// - New row: b_new[i][j] = b1[i][j] * inv_e + b2[i][j] * e
    ///
    /// Key insight: The folded row is still geometric!
    /// - b_new[i][j] = omega^(i*j) * row_start[i] * (inv_e + omega^(i*half_size) * e)
    ///
    /// So we just update:
    /// - row_start[i] *= (inv_e + omega^(i*half_size) * e)
    /// - row_size /= 2
    ///
    /// Complexity: O(n) field operations (vs O(n²) for materialized folding)
    pub fn fold(&mut self, inv_e: Fp, e: Fp) {
        let half_size = self.row_size / 2;

        // Update each row's starting value
        for i in 0..self.omega_base.len() {
            // Compute omega^(i*half_size) = (omega^i)^half_size
            let omega_i_half = self.omega_base[i].pow([half_size as u64]);

            // Update: row_start[i] *= (inv_e + omega^(i*half_size) * e)
            self.row_start[i] *= inv_e + omega_i_half * e;
        }

        // Halve the row size
        self.row_size = half_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::halo2curves::pasta::Fp;
    use ff::Field;

    /// Helper function to build materialized b matrix for testing
    fn build_materialized_b_matrix(omega: Fp, n: usize) -> Vec<Vec<Fp>> {
        let mut omega_powers = Vec::with_capacity(n);
        omega_powers.push(Fp::ONE);
        for i in 1..n {
            omega_powers.push(omega_powers[i - 1] * omega);
        }

        let mut b = Vec::with_capacity(n);
        for i in 0..n {
            let mut b_i = Vec::with_capacity(n);
            let omega_i = omega_powers[i];
            let mut current = Fp::ONE;
            for _ in 0..n {
                b_i.push(current);
                current *= omega_i;
            }
            b.push(b_i);
        }
        b
    }

    #[test]
    fn test_lazy_b_matrix_construction() {
        let omega = Fp::from(5); // Arbitrary omega for testing
        let n = 16;

        let lazy = LazyBMatrix::new(omega, n);

        // Check omega_base is computed correctly
        assert_eq!(lazy.omega_base.len(), n);
        assert_eq!(lazy.omega_base[0], Fp::ONE);
        assert_eq!(lazy.omega_base[1], omega);
        assert_eq!(lazy.omega_base[2], omega * omega);

        // Check row_start is all ones initially
        assert!(lazy.row_start.iter().all(|&x| x == Fp::ONE));

        // Check row_size
        assert_eq!(lazy.row_size, n);
    }

    #[test]
    fn test_get_element_matches_materialized() {
        let omega = Fp::from(5);
        let n = 16;

        let lazy = LazyBMatrix::new(omega, n);
        let materialized = build_materialized_b_matrix(omega, n);

        // Check that lazy.get(i, j) matches materialized[i][j]
        for i in 0..n {
            for j in 0..n {
                assert_eq!(lazy.get(i, j), materialized[i][j],
                    "Mismatch at b[{}][{}]", i, j);
            }
        }
    }

    #[test]
    fn test_inner_product_matches_materialized() {
        let omega = Fp::from(5);
        let n = 16;

        let lazy = LazyBMatrix::new(omega, n);
        let materialized = build_materialized_b_matrix(omega, n);

        // Create a random test vector
        let a: Vec<Fp> = (0..n).map(|i| Fp::from(i as u64 + 1)).collect();

        // Compute inner products both ways and compare
        for i in 0..n {
            let lazy_ip = lazy.inner_product_with(&a, i);

            let mut expected_ip = Fp::ZERO;
            for j in 0..n {
                expected_ip += a[j] * materialized[i][j];
            }

            assert_eq!(lazy_ip, expected_ip,
                "Inner product mismatch for row {}", i);
        }
    }

    #[test]
    fn test_inner_product_with_offset() {
        let omega = Fp::from(5);
        let n = 16;
        let offset = 8;

        let lazy = LazyBMatrix::new(omega, n);
        let materialized = build_materialized_b_matrix(omega, n);

        // Create a test vector (size n/2 for the split half)
        let a: Vec<Fp> = (0..(n - offset)).map(|i| Fp::from(i as u64 + 1)).collect();

        // Compute inner products with offset
        for i in 0..n {
            let lazy_ip = lazy.inner_product_with_offset(&a, i, offset);

            let mut expected_ip = Fp::ZERO;
            for j in 0..a.len() {
                expected_ip += a[j] * materialized[i][offset + j];
            }

            assert_eq!(lazy_ip, expected_ip,
                "Inner product with offset mismatch for row {}", i);
        }
    }

    #[test]
    fn test_fold_preserves_geometric_structure() {
        let omega = Fp::from(5);
        let n = 16;
        let inv_e = Fp::from(3);
        let e = Fp::from(7);

        let mut lazy = LazyBMatrix::new(omega, n);
        let materialized = build_materialized_b_matrix(omega, n);

        // Fold the lazy matrix
        lazy.fold(inv_e, e);

        // Manually compute what the folded materialized matrix should be
        let half = n / 2;
        let mut folded_materialized = Vec::with_capacity(n);
        for i in 0..n {
            let mut folded_row = Vec::with_capacity(half);
            for j in 0..half {
                let b1_ij = materialized[i][j];
                let b2_ij = materialized[i][j + half];
                folded_row.push(b1_ij * inv_e + b2_ij * e);
            }
            folded_materialized.push(folded_row);
        }

        // Check that row_size was updated
        assert_eq!(lazy.row_size, half);

        // Check that lazy.get matches folded materialized matrix
        for i in 0..n {
            for j in 0..half {
                let lazy_val = lazy.get(i, j);
                let expected_val = folded_materialized[i][j];
                assert_eq!(lazy_val, expected_val,
                    "Fold mismatch at b[{}][{}]", i, j);
            }
        }
    }

    #[test]
    fn test_multiple_folds() {
        let omega = Fp::from(5);
        let n = 16;

        let mut lazy = LazyBMatrix::new(omega, n);
        let mut materialized = build_materialized_b_matrix(omega, n);

        // Perform multiple folding rounds
        let challenges = vec![
            (Fp::from(3), Fp::from(7)),
            (Fp::from(11), Fp::from(13)),
            (Fp::from(17), Fp::from(19)),
        ];

        for (inv_e, e) in challenges {
            // Fold lazy
            lazy.fold(inv_e, e);

            // Fold materialized
            let half = materialized[0].len() / 2;
            let mut new_materialized = Vec::with_capacity(n);
            for i in 0..n {
                let mut new_row = Vec::with_capacity(half);
                for j in 0..half {
                    let b1_ij = materialized[i][j];
                    let b2_ij = materialized[i][j + half];
                    new_row.push(b1_ij * inv_e + b2_ij * e);
                }
                new_materialized.push(new_row);
            }
            materialized = new_materialized;

            // Verify they match
            let current_size = lazy.row_size;
            for i in 0..n {
                for j in 0..current_size {
                    assert_eq!(lazy.get(i, j), materialized[i][j],
                        "Multiple fold mismatch at b[{}][{}]", i, j);
                }
            }
        }
    }

    #[test]
    fn test_row_start_values_after_fold() {
        let omega = Fp::from(5);
        let n = 8;
        let inv_e = Fp::from(3);
        let e = Fp::from(7);

        let mut lazy = LazyBMatrix::new(omega, n);

        // Before fold: row_start[i] = 1 for all i
        assert!(lazy.row_start.iter().all(|&x| x == Fp::ONE));

        // After fold: row_start[i] should be updated
        lazy.fold(inv_e, e);

        // Manually compute expected row_start values
        let half = n / 2;
        for i in 0..n {
            let omega_i_half = lazy.omega_base[i].pow([half as u64]);
            let expected_start = inv_e + omega_i_half * e;
            assert_eq!(lazy.row_start[i], expected_start,
                "row_start[{}] incorrect after fold", i);
        }

        // b[i][0] should now equal row_start[i]
        for i in 0..n {
            assert_eq!(lazy.get(i, 0), lazy.row_start[i],
                "b[{}][0] should equal row_start[{}]", i, i);
        }
    }

    #[test]
    fn test_fft_matches_naive_after_folding() {
        use halo2_proofs::poly::EvaluationDomain;

        // Use a proper root of unity so omega^n = 1
        let n = 16;
        let k = (n as f64).log2() as u32;
        let domain: EvaluationDomain<Fp> = EvaluationDomain::new(1, k);
        let omega = domain.get_omega();

        let mut lazy = LazyBMatrix::new(omega, n);

        // Test vector for initial (unfolded) case
        let a: Vec<Fp> = (0..n).map(|i| Fp::from(i as u64 + 1)).collect();

        // Verify FFT matches naive before folding
        let naive_results: Vec<Fp> = (0..n).map(|i| lazy.inner_product_with(&a, i)).collect();
        let fft_results = lazy.compute_all_inner_products_fft(&a);
        for i in 0..n {
            assert_eq!(fft_results[i], naive_results[i],
                "FFT mismatch at row {} (before fold)", i);
        }

        // Now fold and test with smaller vector
        let inv_e = Fp::from(3);
        let e = Fp::from(7);
        lazy.fold(inv_e, e);

        let half = n / 2;
        let a_small: Vec<Fp> = (0..half).map(|i| Fp::from(i as u64 + 10)).collect();

        // Verify FFT matches naive AFTER folding (this was the bug)
        let naive_results: Vec<Fp> = (0..n).map(|i| lazy.inner_product_with(&a_small, i)).collect();
        let fft_results = lazy.compute_all_inner_products_fft(&a_small);
        for i in 0..n {
            assert_eq!(fft_results[i], naive_results[i],
                "FFT mismatch at row {} (after fold, vector size {})", i, half);
        }

        // Also test offset version after folding
        let offset = half;
        let naive_offset: Vec<Fp> = (0..n).map(|i| lazy.inner_product_with_offset(&a_small, i, offset)).collect();
        let fft_offset = lazy.compute_all_inner_products_with_offset_fft(&a_small, offset);
        for i in 0..n {
            assert_eq!(fft_offset[i], naive_offset[i],
                "FFT offset mismatch at row {} (after fold)", i);
        }

        // Second fold
        lazy.fold(Fp::from(11), Fp::from(13));
        let quarter = half / 2;
        let a_tiny: Vec<Fp> = (0..quarter).map(|i| Fp::from(i as u64 + 20)).collect();

        let naive_results: Vec<Fp> = (0..n).map(|i| lazy.inner_product_with(&a_tiny, i)).collect();
        let fft_results = lazy.compute_all_inner_products_fft(&a_tiny);
        for i in 0..n {
            assert_eq!(fft_results[i], naive_results[i],
                "FFT mismatch at row {} (after 2 folds, vector size {})", i, quarter);
        }
    }
}
