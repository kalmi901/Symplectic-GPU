#ifndef DIRECTSOLVERS_CUH
#define DIRECTSOLVERS_CUH
#include "Tools.cuh"

// Gaussian Elimination ------------------------------------------------
template<typename T, int n>
__host__ __device__ void GE_Solve(T* A, T* b, T* x, bool pivot = true) {
	// Create Upper Triangle Matrix
	// Make all rows below the current diagonal element zero
	T c;
	for (int j = 0; j < n; j++) {
		if (pivot)
			PivotRow<T, n>(A, b, j);
		for (int k = j + 1; k < n; k++) {
			c = -A[k * n + j] / A[j * n + j];
			for (int i = j; i < n; i++) {
				if (j == i)
					A[k * n + j] = 0;
				else
					A[k * n + i] += c * A[j * n + i];
			}
			b[k] += c * b[j];
		}
	}
	// "A" has the form of an upper triangle matrix
	BackwardSubstitution<T>(A, b, x, n);
}

// Divison-Free Gaussian elimination (without optimization)
template<typename T, int n>
__host__ __device__ void DGE_Solve(T* A, T* b, T* x, bool pivot = true) {
	// Create Upper Triangle Matrix
	// Make all rows below the current diagonal element zero
	// Avoid the "division" ->multiply the equation by the denominator
	double M, L;
	for (int j = 0; j < n; j++) {
		if (pivot)
			PivotRow<T, n>(A, b, j);
		M = A[j * n + j];
		for (int k = j + 1; k < n; k++) {
			L = A[k * n + j];
			for (int i = j; i < n; i++) {
				if (j == i)
					A[k * n + j] = 0;
				else
					A[k * n + i] = A[k * n + i] * M - L * A[j * n + i];
			}
			b[k] = M * b[k] - L * b[j];
		}
	}
	// "A" has the form of an upper triangle matrix
	// range increases ~ of order M
	BackwardSubstitution<T>(A, b, x, n);
}


// Divison-Free Gaussian elimination (with optimization)
template<typename T, int n>
__host__ __device__ void GE_Solve_Estimate(T* A, T* b, T* x, bool pivot = true) {
	// Create Upper Triangle Matrix
	// Make all rows below the current diagonal element zero
	// Avoid the "division" ->multiply the equation by the denominator
	double M, L, r;
	for (int j = 0; j < n; j++) {
		if (pivot)
			PivotRow<T, n>(A, b, j);
		M = A[j * n + j];
		r = f_div_estimate(abs(M));

		for (int k = j + 1; k < n; k++) {
			L = A[k * n + j];
			for (int i = j; i < n; i++) {
				if (j == i)
					A[k * n + j] = 0;
				else {
					A[k * n + i] = (A[k * n + i] * M - L * A[j * n + i]) * r;
				}
			}
			b[k] = (M * b[k] - L * b[j]) * r;
		}
	}
	// "A" has the form of an upper triangle matrix
	BackwardSubstitution<T>(A, b, x, n);
}

// Gaussian elimination using fast_division algorithm
template<typename T, int n>
__host__ __device__ void GE_Solve_Fdiv(T* A, T* b, T* x, bool pivot = true) {
	// Create Upper Triangle Matrix
	// Make all rows below the current diagonal element zero
	// Avoid the "division" ->multiply the equation by the denominator
	double M, L, r;
	for (int j = 0; j < n; j++) {
		if (pivot)
			PivotRow<T, n>(A, b, j);
		M = A[j * n + j];
		r = f_div2(abs(M));

		for (int k = j + 1; k < n; k++) {
			L = A[k * n + j];
			for (int i = j; i < n; i++) {
				if (j == i)
					A[k * n + j] = 0;
				else {
					A[k * n + i] = (A[k * n + i] * M - L * A[j * n + i]) * r;
				}
			}
			b[k] = (M * b[k] - L * b[j]) * r;
		}
	}
	// "A" has the form of an upper triangle matrix
	BackwardSubstitution_FDiv<T>(A, b, x, n);
}


// Analitic -------------------------------------------

template <typename T>
__inline__ __host__ __device__ void solve2x2(T* A, T* b, T* x) {
	T rD = 1 / (A[0] * A[3] - A[1] * A[2]);
	x[0] = (A[3] * b[0] - A[1] * b[1]) * rD;
	x[1] = (-A[2] * b[0] + A[0] * b[1]) * rD;
}

template<typename T>
__inline__ __host__ __device__ void solve3x3(T* A, T* b, T* x) {
	T rD = 1 / (A[0] * A[4] * A[8] - A[0] * A[5] * A[7]
				- A[1] * A[3] * A[8] + A[1] * A[5] * A[6]
				+ A[2] * A[3] * A[7] - A[2] * A[4] * A[6]);

	x[0] = ((A[4] * A[8] - A[5] * A[7]) * b[0]
		+ (A[2] * A[7] - A[1] * A[8]) * b[1]
		+ (A[1] * A[5] - A[2] * A[4]) * b[2]) * rD;
	x[1] = ((A[5] * A[6] - A[3] * A[8]) * b[0]
		+ (A[0] * A[8] - A[2] * A[6]) * b[1]
		+ (A[2] * A[3] - A[0] * A[5]) * b[2]) * rD;
	x[2] = ((A[3] * A[7] - A[4] * A[6]) * b[0]
		+ (A[1] * A[6] - A[0] * A[7]) * b[1]
		+ (A[0] * A[4] - A[1] * A[3]) * b[2]) * rD;
}

template<typename T>
__inline__ __host__ __device__ void solve4x4(T* A, T* b, T* x) {

	T rD = 1 / (A[0] * A[5] * A[10] * A[15] - A[0] * A[5] * A[11] * A[14]
		- A[0] * A[6] * A[9] * A[15] + A[0] * A[6] * A[11] * A[13]
		+ A[0] * A[7] * A[9] * A[14] - A[0] * A[7] * A[10] * A[13]
		- A[1] * A[4] * A[10] * A[15] + A[1] * A[4] * A[11] * A[14]
		+ A[1] * A[6] * A[8] * A[15] - A[1] * A[6] * A[11] * A[12]
		- A[1] * A[7] * A[8] * A[14] + A[1] * A[7] * A[10] * A[12]
		+ A[2] * A[4] * A[9] * A[15] - A[2] * A[4] * A[11] * A[13]
		- A[2] * A[5] * A[8] * A[15] + A[2] * A[5] * A[11] * A[12]
		+ A[2] * A[7] * A[8] * A[13] - A[2] * A[7] * A[9] * A[12]
		- A[3] * A[4] * A[9] * A[14] + A[3] * A[4] * A[10] * A[13]
		+ A[3] * A[5] * A[8] * A[14] - A[3] * A[5] * A[10] * A[12]
		- A[3] * A[6] * A[8] * A[13] + A[3] * A[6] * A[9] * A[12]);

	x[0] = ((A[5] * A[10] * A[15] - A[5] * A[11] * A[14]
		- A[6] * A[9] * A[15] + A[6] * A[11] * A[13]
		+ A[7] * A[9] * A[14] - A[7] * A[10] * A[13]) * b[0] +
		(-A[1] * A[10] * A[15] + A[1] * A[11] * A[14] 
		+ A[2] * A[9] * A[15] - A[2] * A[11] * A[13]
		- A[3] * A[9] * A[14] + A[3] * A[10] * A[13]) * b[1] +
		( A[1] * A[6] * A[15] - A[1] * A[7] * A[14]
		- A[2] * A[5] * A[15] + A[2] * A[7] * A[13]
		+ A[3] * A[5] * A[14] - A[3] * A[6] * A[13]) * b[2] +
		(-A[1] * A[6] * A[11] + A[1] * A[7] * A[10]
		+ A[2] * A[5] * A[11] - A[2] * A[7] * A[9]
		- A[3] * A[5] * A[10] + A[3] * A[6] * A[9]) * b[3]) * rD;	
	x[1] = ((-A[4] * A[10] * A[15] + A[4] * A[11] * A[14]
		+ A[6] * A[8] * A[15] - A[6] * A[11] * A[12]
		- A[7] * A[8] * A[14] + A[7] * A[10] * A[12]) * b[0] +
		( A[0] * A[10] * A[15] - A[0] * A[11] * A[14]
		- A[2] * A[8] * A[15] + A[2] * A[11] * A[12]
		+ A[3] * A[8] * A[14] - A[3] * A[10] * A[12]) * b[1] +
		(-A[0] * A[6] * A[15] + A[0] * A[7] * A[14]
		+ A[2] * A[4] * A[15] - A[2] * A[7] * A[12]
		- A[3] * A[4] * A[14] + A[3] * A[6] * A[12]) * b[2] +
		( A[0] * A[6] * A[11] - A[0] * A[7] * A[10]
		- A[2] * A[4] * A[11] + A[2] * A[7] * A[8]
		+ A[3] * A[4] * A[10] - A[3] * A[6] * A[8]) * b[3]) * rD;
	x[2] = ((A[4] * A[9] * A[15] - A[4] * A[11] * A[13]
		- A[5] * A[8] * A[15] + A[5] * A[11] * A[12]
		+ A[7] * A[8] * A[13] - A[7] * A[9] * A[12]) * b[0] +
		(-A[0] * A[9] * A[15] + A[0] * A[11] * A[13]
		+ A[1] * A[8] * A[15] - A[1] * A[11] * A[12]
		- A[3] * A[8] * A[13] + A[3] * A[9] * A[12]) * b[1] +
		( A[0] * A[5] * A[15] - A[0] * A[7] * A[13]
		- A[1] * A[4] * A[15] + A[1] * A[7] * A[12]
		+ A[3] * A[4] * A[13] - A[3] * A[5] * A[12]) * b[2] +
		(-A[0] * A[5] * A[11] + A[0] * A[7] * A[9]
		+ A[1] * A[4] * A[11] - A[1] * A[7] * A[8]
		- A[3] * A[4] * A[9] + A[3] * A[5] * A[8]) * b[3]) * rD;
	x[3] = ((-A[4] * A[9] * A[14] + A[4] * A[10] * A[13]
		+ A[5] * A[8] * A[14] - A[5] * A[10] * A[12]
		- A[6] * A[8] * A[13] + A[6] * A[9] * A[12]) * b[0] +
		( A[0] * A[9] * A[14] - A[0] * A[10] * A[13]
		- A[1] * A[8] * A[14] + A[1] * A[10] * A[12]
		+ A[2] * A[8] * A[13] - A[2] * A[9] * A[12]) * b[1] +
		(-A[0] * A[5] * A[14] + A[0] * A[6] * A[13]
		+ A[1] * A[4] * A[14] - A[1] * A[6] * A[12]
		- A[2] * A[4] * A[13] + A[2] * A[5] * A[12]) * b[2] +
		( A[0] * A[5] * A[10] - A[0] * A[6] * A[9]
		- A[1] * A[4] * A[10] + A[1] * A[6] * A[8]
		+ A[2] * A[4] * A[9] - A[2] * A[5] * A[8]) * b[3]) * rD;
}

template<typename T, int n>
__host__ __device__ void ANAL_Solve(T* A, T* b, T* x){
	if (n == 1) {
		x[0] = b[0] / A[0];
	}
	else if (n == 2) {
		solve2x2(A, b, x);
	}
	else if (n == 3) {
		solve3x3(A, b, x);
	}
	else if (n == 4) {
		solve4x4(A, b, x);
	}
	else {
		printf("\n Err: Analytic solver for (%d x %d) matrix is not availabel", n, n);
		return;
	}
}


#endif 