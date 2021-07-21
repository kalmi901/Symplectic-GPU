#ifndef STATIONARYSOLVERS_CUH
#define STATIONARYSOLVERS_CUH

#include "Tools.cuh"
#include <iostream>

template<typename T, int n>
__host__ __device__ void GAUSS_SEIDEL_Solve(T* A, T* b, T* x, T atol, int max_iter, bool pivot = true) {
	// Test initial convergence
	T r[n];
	CalculateResidual<T, n>(A, b, x, r);
	T r_norm = GetNorm<T, n>(r);
	T error = r_norm;

	if (error <= atol) {
		printf("\nGAUSS_SEIDEL_Solve: Initial residual less then the user specified tolerance. Iteration does not required");
		return;
	}

	// x is not accurate; iteration is required
	// pivot rows
	if (pivot) {
		for (int j = 0; j < n; j++) {
			PivotRow<T, n>(A, b, j);
		}
	}

	T sum;
	for (int iter = 0; iter < max_iter; iter++) {
		for (int i = 0; i < n; i++) {
			sum = 0;
			for (int j = 0; j < n; j++) {
				if (j != i) {
					sum += A[i * n + j] * x[j];
				}
			}
			x[i] = (b[i] - sum) / A[i * n + i];
		}

		CalculateResidual<T, n>(A, b, x, r);
		r_norm = GetNorm<T, n>(r);
		error = r_norm;

		if (error < atol) {
			printf("\nGAUSS_SEIDEL: Iteration Succeded: Number of Iterations: %d\n", iter + 1);
			return;
		}

		if (error >= 1e6) {
			printf("\nErr: GAUSS_SEIDEL_Solve: solution does not converge!\n");
			return;
		}
	}
	printf("\nErr: GAUSS_SEIDEL_Solve: Max iteration is reached!\n");
}

template<typename T, int n>
__host__ __device__ void JACOBI_Solve(T* A, T* b, T* x, T atol, int max_iter, bool pivot = true) {
	// Test initial convergence
	T r[n];
	CalculateResidual<T, n>(A, b, x, r);
	T r_norm = GetNorm<T, n>(r);
	T error = r_norm;

	if (error <= atol) {
		printf("\nJacobi_Solve: Initial residual less then the user specified tolerance. Iteration does not required\n");
		return;
	}

	// x is not accurate; iteration is required
	// pivot rows -> prepare the matrix for iteration
	if (pivot) {
		for (int j = 0; j < n; j++) {
			PivotRow<T, n>(A, b, j);
		}
	}

	T xold[n], sum;
	for (int iter = 0; iter < max_iter; iter++) {
		for (int i = 0; i < n; i++) {
			xold[i] = x[i];
		}
		for (int i = 0; i < n; i++) {
			sum = 0;
			for (int j = 0; j < n; j++) {
				if (j != i) {
					sum += A[i * n + j] * xold[j];
				}
			}
			x[i] = (b[i] - sum) / A[i * n + i];
		}
		CalculateResidual<T, n>(A, b, x, r);
		r_norm = GetNorm<T, n>(r);
		error = r_norm;		// * rb_norm (rel. error control?)

		if (error <= atol) {
			printf("\nJacobi_Solve: Iteration Succeded: Number of Iterations: %d\n", iter + 1);
			return;
		}

		if (error >= 1e6) {
			printf("\nErr: JACOBI_Solve: solution does not converge!\n");
			return;
		}
	}

	printf("\nErr: JACOBI_Solve: Max iteration is reached!\n");
}

#endif
