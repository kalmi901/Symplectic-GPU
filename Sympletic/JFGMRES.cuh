#ifndef JFGMRES_CUH
#define JFGMRES_CUH
#include "Tools.cuh"
#include "GMRES.cuh"
#include <math.h>



template<typename T, int n, int m>
__inline__ __host__ void ArnoldiIteration(T* VT, T* H, T* w, const int i) {
	//MatMul<T, n>(A, VT, w, i);
	JV_approx<T, n>(w, VT, i);	// Külsõ loop-ban állhat elõ
	for (int j = 0; j < n; j++) {
		//printf("\nVT[%d]: %.4f", j, VT[j]);
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
__host__ bool GMRES_ITERATION(T* b, T* x, T* r, T r_norm, T atol) {
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
		//JV_approx<T, n>(w, VT, i);
		ArnoldiIteration<T, n, m>(VT, H, w, i);

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
__host__ void JFGMRES_Solve(T* b, T* x, const T atol, const int max_iter) {
	T r[n];
	T JV[n];
	JV_approx<T, n>(JV, x, 0);
	for (int j = 0; j < n; j++) {
		r[j] = b[j] - JV[j];
		//printf("\n r[%d]: %.4f", j, r[j]);
	}
	T r_norm = GetNorm<T, n>(r);

	if (r_norm <= atol) {
		return;
	}

	int iter = 0;
	while (iter < max_iter) {
		if (GMRES_ITERATION<T, n, m>(b, x, r, r_norm, atol) == true) {
			//printf("\nIteration succeeded: Number of Iterations: %d\n", iter + 1);
			break;
		}
		iter++;
		for (int j = 0; j < n; j++) {
			//printf("\n x[%d]: %.4f", j, x[j]);
		}
		// CALCULATE RESIDUAL
		JV_approx<T, n>(JV, x, 0);

		for (int j = 0; j < n; j++) {
			r[j] = b[j] - JV[j];
			//printf("\n r[%d]: %.4f", j, r[j]);
		}
		r_norm = GetNorm<T, n>(r);
	}
	//printf("\nErr: JFGMRES_Solve: Max iteration is reached!\n");
}


#endif // JFGMRES_CUH
