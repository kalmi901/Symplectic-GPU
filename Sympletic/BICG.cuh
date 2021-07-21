#ifndef BICG_CUH
#define BICG_CUH
#include "Tools.cuh"
#include <math.h>


template<typename T, int n>
__host__ __device__ void InitializeBICG_Variables(T* r, T* rtilde, T* p, T* ptilde) {
	// r is a know residual vector of size n
	// initialization -> copy the elements of r into rtilde, p, ptilde
	// rtilde = r
	// p = r
	// ptilde = rtilde

	T r_val;
	for (int i = 0; i < n; i++) {
		r_val = r[i];
		rtilde[i] = r_val;
		p[i] = r_val;
		ptilde[i] = r_val;
	}
}

template<typename T, int n>
__host__ __device__ void BICG_Solve(T* A, T* b, T* x, const T atol, const int max_iter) {
	
	// Test initial convergence -------------------
	// residual vector
	T r[n];
	CalculateResidual<T, n>(A, b, x, r);
	T r_norm = GetNorm<T, n>(r);	

	// Convergence check
	// ISSUE
	if (r_norm <= atol)
		return;

	// x is not accurete; iteration is required

	// Initialize workspace
	T p[n], rtilde[n], ptilde[n], q[n], qtilde[n];
	InitializeBICG_Variables<T, n>(r, rtilde, p, ptilde);

	// Additional scalar variables are reqiured
	T rho1, rho2, alpha, beta;

	// Iteration
	for (int iter = 0; iter < max_iter; iter++) {
		rho1 = VecDot<T, n>(r, rtilde);			// ISSUE
		if (rho1 == 0) {
			// Method Failure
			printf("\nErr: BICG_Solve: Method failure\n");
			return;
		}

		if (iter != 0) {
			beta = rho1 / rho2;
			for (int i = 0; i < n; i++) {
				p[i] = r[i] + beta * p[i];
				ptilde[i] = rtilde[i] + beta * ptilde[i];
			}
		}
		// ISSUE - Merge
		MatMul<T, n>(A, p, q);
		TransMatMul<T>(A, ptilde, qtilde, n);

		alpha = rho1 / VecDot<T, n>(ptilde, q);

		for (int i = 0; i < n; i++) {
			x[i] += alpha * p[i];
			r[i] -= alpha * q[i];
			rtilde[i] -= alpha * qtilde[i];
		}

		rho2 = rho1;

		if (GetNorm<T, n>(r) <= atol) {
			// Converged
			printf("\nIteration succeeded: Number of Iterations: %d\n", iter + 1);
			return;
		}
	}
	printf("\nErr: BICG_Solve: Max iteration is reached!\n");
}

#endif