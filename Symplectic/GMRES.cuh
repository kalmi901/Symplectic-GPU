#ifndef GMRES_CUH
#define GMRES_CUH
#include "Tools.cuh"
#include <math.h>

template<typename T>
__host__ __device__ void ApplyPlaneRotation(T& dx, T& dy, T& cs, T& sn) {
	T temp = cs * dx + sn * dy;
	dy = -sn * dx + cs * dy;
	dx = temp;
}

template<typename T>
__host__ __device__ void GeneratePlaneRotation(T& dx, T& dy, T& cs, T& sn) {
	if (abs(dy) <= 1e-16) {
		cs = 1;
		sn = 0;
	}
	else if (abs(dx) <= 1e-16) {
		cs = 0.0;
		sn = copysign(1.0, dy);
	}
	else if (abs(dy) > abs(dx)) {
		T temp = dx / dy;
		sn = copysign(1 / sqrt(1 + temp * temp), dy);
		cs = temp * sn;
	}
	else {
		T temp = dy / dx;
		cs = copysign(1 / sqrt(1 + temp * temp), dx);
		sn = temp * cs;
	}
}


template<typename T, int n, int m>
__inline__ __host__ __device__ void ArnoldiIteration(T* A, T* VT, T* H, T* w, const int i) {
	MatMul<T, n>(A, VT, w, i);
	for (int j = 0; j < n; j++) {
		//printf("\nw[%d]: %.4f", j, w[j]);
	}
	int kn, km, i1n;
	T h;
	for (int k = 0; k < i + 1; k++) {
		km = k * m;
		kn = k * n;
		H[km + i] = VecDot<T, n>(w, VT, k);
		for (int z = 0; z < n; z++) {
			w[z] -= H[km + i] * VT[kn + z];
		}
	}
	h = GetNorm<T, n>(w);
	H[(i + 1)*m + i] = h;
	if (h < 1e-16) {
		printf("\nErr: GMRES_Solve: Method failure\n");
		return;
	}
	h = 1 / h;
	i1n = (i + 1) * n;
	for (int z = 0; z < n; z++) {
		VT[i1n + z] = w[z] * h;
	}
}

template<typename T, int n, int m>
__host__ __device__ bool GMRES_ITERATION(T* A, T* b, T* x, T* r, T r_norm, T atol) {
	T inv_r_norm = (T)1 / r_norm;

	// Initialize workspace
	T VT[(m + 1) * n]{};		// Transpose of V
	T H[(m + 1) * m]{};			// Hessenber matrix
	T cs[m]{};					// cosines of Givens rotations
	T sn[m]{};					// sines of Givens rotations
	T w[n]{};					// Krylov vector
	T y[m]{};					// solution of "sub-system"
	T s[(m + 1)]{};				// residual vector

	for (int j = 0; j < n; j++) {
		VT[j] = r[j] * inv_r_norm;
	}

	s[0] = r_norm;
	for (int j = 1; j < m + 1; j++) {
		s[j] = 0;
	}

	// GMRES Iteration
	int im, i1m;
	for (int i = 0; i < m; i++) {
		ArnoldiIteration<T, n, m>(A, VT, H, w, i);

		for (int k = 0; k < i; k++) {
			ApplyPlaneRotation<T>(H[k * m + i], H[(k + 1) * m + i], cs[k], sn[k]);
		}

		im = i * m;		i1m = (i + 1) * m;
		GeneratePlaneRotation<T>(H[im + i], H[i1m + i], cs[i], sn[i]);
		ApplyPlaneRotation<T>(H[im + i], H[i1m + i], cs[i], sn[i]);
		ApplyPlaneRotation<T>(s[i], s[i + 1], cs[i], sn[i]);
		H[i1m + i] = 0;

		// abs(s[i + 1]* rb_norm in case of relative error control
		if (abs(s[i + 1]) < atol) {
			// Converged
			BackwardSubstitution<T>(H, s, y, m, i + 1);
			// Reuse w as a temporary array
			TransMatMul<T, n>(VT, y, w, i + 1);
			for (int z = 0; z < n; z++) {
				x[z] += w[z];
			}
			return true;
		}
	}
	BackwardSubstitution<T>(H, s, y, m);
	TransMatMul<T, n>(VT, y, w, m);
	for (int z = 0; z < n; z++) {
		x[z] += w[z];
	}
	//printVector(x, n, "x:");
	return false;
}


template<typename T, int n, int m>
__host__ __device__ void GMRES_Solve(T* A, T* b, T* x, const T atol, const int max_iter) {
	T r[n];
	CalculateResidual<T, n>(A, b, x, r);
	for (int j = 0; j < n; j++) {
		//printf("\n r[%d]: %.4f", j, r[j]);
	}
	T r_norm = GetNorm<T, n>(r);
	
	if (r_norm <= atol) {
		// The initial condition is accurate
		return;
	}

	int iter = 0;
	while (iter < max_iter) {
		if (GMRES_ITERATION<T, n, m>(A, b, x, r, r_norm, atol) == true) {
			//printf("\nIteration succeeded: Number of Iterations: %d\n", iter + 1);
			break;
		}

		iter++;
		for (int j = 0; j < n; j++) {
			//printf("\n x[%d]: %.4f", j, x[j]);
		}
		CalculateResidual<T, n>(A, b, x, r);
		//for (int j = 0; j < n; j++) {
			//printf("\n r[%d]: %.4f", j, r[j]);
		//}
		r_norm = GetNorm<T, n>(r);
	}
	//printf("\nErr: GMRES_Solve: Max iteration is reached!\n");
}

#endif