#ifndef JACOBIFREENEWTONITERATION_CUH
#define JACOBIFREENEWTONITERATION_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Dependecies\AD.cuh"
#include <cmath>
#include "GMRES.cuh"

#define B 1e-6

template<typename T, int NT, int SD>
__forceinline__ __device__ void JV_ApproxFirstOrder(T* JV, T* V, const int offset, const int tid, T* k, T* xk, T* CP, T tk, T h) {

	// <---- Estimate perturbation --------------------- 
	T sum = 0;
	for (int i = 0; i < SD; i++) {
		sum += B * (1 + abs(xk[i]));
	}

	T temp = 0;
	int OFFSET = offset * SD;
	for (int i = 0; i < SD; i++) {
		temp += V[OFFSET + i] * V[OFFSET + i];
	}
	T v_norm = sqrt(temp);

	T perturbation = (v_norm > 1e-12) ? sum / (SD * v_norm) : sum / SD;
	// ----------------------------------------------->


	T kPer[SD];			// Ode Function value
	T xkPer[SD];		// Perturbed State
	// F(xPer) = x - xPer + h*kPer
	// kPer = f(tk, xPer)

	for (int i = 0; i < SD; i++) {
		xkPer[i] = xk[i] + V[OFFSET + i] * perturbation;
	}

	OdeFunction<T, T>(tid, NT, kPer, xkPer, tk, CP);
	T CONST = h / perturbation;
	for (int i = 0; i < SD; i++) {
		JV[i] = CONST * (kPer[i] - k[i]) - V[OFFSET + i];
	}
}



template<typename T, int NT, int SD>
__forceinline__ __device__ void JV_ApproxSecondOrder(T* JV, T* V, const int offset, const int tid, T* xk, T* CP, T tk, T h) {

	// <---- Estimate perturbation --------------------- 
	T sum = 0;
	for (int i = 0; i < SD; i++) {
		sum += B * (1 + abs(xk[i]));
	}

	T temp = 0;
	int OFFSET = offset * SD;
	for (int i = 0; i < SD; i++) {
		temp += V[OFFSET + i] * V[OFFSET + i];
	}
	T v_norm = sqrt(temp);

	T perturbation = (v_norm > 1e-12) ? sum / (SD * v_norm) : sum / SD;
	// ----------------------------------------------->

	T kPerF[SD], kPerB[SD];
	T xkPer[SD];

	for (int i = 0; i < SD; i++) {
		xkPer[i] = xk[i] + V[OFFSET + i] * perturbation;
	}

	OdeFunction<T, T>(tid, NT, kPerF, xkPer, tk, CP);
	
	for (int i = 0; i < SD; i++) {
		xkPer[i] = xk[i] - V[OFFSET + i] * perturbation;
	}
	OdeFunction<T, T>(tid, NT, kPerB, xkPer, tk, CP);
	
	T CONST = h / 2 / perturbation;
	for (int i = 0; i < SD; i++) {
		JV[i] = CONST * (kPerF[i] - kPerB[i]) - V[OFFSET + i];
	}
}

template<typename T, int NT, int SD, int R>
__forceinline__ __device__ void ArnoldiIteration(T* VT, T* H, T* w, const int i,
											const int tid, T* k, T* x, T* xk, T* CP, T tk, T h) {

	/*T J[SD * SD];
	J[0] = -1;
	J[1] = h / CP[0];
	J[2] = -h * CP[1];
	J[3] = -1;
	MatMul<T, SD>(J, VT, w, i);
	if (tid == 0 && tk == h) {
		for (int i = 0; i < SD; i++) {
			printf("\n w[%d]: %.4f", i, w[i]);
		}
	}
	*/
#if(ORDER == 1)
	JV_ApproxFirstOrder<T, NT, SD>(w, VT, i, tid, k, xk, CP, tk, h);
#elif (ORDER == 2)
	JV_ApproxSecondOrder<T, NT, SD>(w, VT, i, tid, xk, CP, tk, h);
#endif

	if (tid == 0 && tk == h) {
		for (int i = 0; i < SD; i++) {
			printf("\n w[%d]: %.4f", i, w[i]);
		}
	}
	int kn, km, i1n;
	T h_element;
	for (int k = 0; k < i + 1; k++) {
		km = k * R;
		kn = k * R;
		H[km + i] = VecDot<T, SD>(w, VT, k);
		for (int z = 0; z < SD; z++) {
			w[z] -= H[km + i] * VT[kn + z];
		}
	}
	h_element = GetNorm<T, SD>(w);
	H[(i + 1)*R + i] = h_element;
	if (h_element < 1e-16) {
		printf("\nErr: GMRES_Solve: Method failure\n");
		return;
	}
	T r_h = 1 / h_element;
	i1n = (i + 1) * SD;
	for (int z = 0; z < SD; z++) {
		VT[i1n + z] = w[z] * r_h;
	}
}


template<typename T, int NT, int SD, int R>
__device__ bool JFGMRES_ITERATION(T* F, T* dx, T FNORM, T AbsTol,
								  const int tid, T* k, T* x, T* xk, T* CP, T tk, T h) {
	
	// tid	- ID of Actual Thread
	// x	- Actual State
	// xk	- MidPoint State
	// CP	- Control Parameter
	// tk	- MidPoint Time
	// h	- TimeStep (half)

	// F(xk) = x - xk + h*k
	// k = f(tk, xk)	// OdeFunction

	// Ha dx nulla, akkor F = r(esidual) -> FNORM = r_norm
	T inv_r_norm = (T)1 / FNORM;
	// Initialize workspace
	T VT[(R + 1) * SD]{};		// Transpose of V
	T H[(R + 1) * R]{};			// Hessenber matrix
	T cs[R]{};					// cosines of Givens rotations
	T sn[R]{};					// sines of Givens rotations
	T w[SD]{};					// Krylov vector
	T y[R]{};					// solution of "sub-system"
	T s[(R + 1)]{};				// residual vector

	for (int i = 0; i < SD; i++) {
		//VT[i] = r[i] * inv_r_norm;
		VT[i] = F[i] * inv_r_norm;		// F = r
	}

	s[0] = FNORM;
	for (int i = 1; i < R + 1; i++) {
		s[i] = 0;
	}

	// GMRES Iteration
	int iR, i1R;
	for (int i = 0; i < R; i++) {
		ArnoldiIteration<T, NT, SD, R>(VT, H, w, i, tid, k, x, xk, CP, tk, h);

		for (int k = 0; k < i; k++) {
			ApplyPlaneRotation<T>(H[k * R + i], H[(k + 1) * R + i], cs[k], sn[k]);
		}

		iR = i * R;		i1R = (i + 1) * R;
		GeneratePlaneRotation<T>(H[iR + i], H[i1R + i], cs[i], sn[i]);
		ApplyPlaneRotation<T>(H[iR + i], H[i1R + i], cs[i], sn[i]);
		ApplyPlaneRotation<T>(s[i], s[i + 1], cs[i], sn[i]);
		H[i1R + i] = 0;

		// abs(s[i + 1]* rb_norm in case of relative error control
		if (abs(s[i + 1]) < AbsTol) {
			// Converged
			BackwardSubstitution<T>(H, s, y, R, i + 1);
			// Reuse w as a temporary array
			TransMatMul<T, SD>(VT, y, w, i + 1);
			for (int z = 0; z < SD; z++) {
				dx[z] += w[z];
			}
			return true;
		}
	}
	BackwardSubstitution<T>(H, s, y, R);
	TransMatMul<T, SD>(VT, y, w, R);
	for (int i = 0; i < SD; i++) {
		dx[i] += w[i];
	}
	//printVector(x, n, "x:");
	return false;
}


template<typename T, int SD, int R>
__inline__ __device__ void JFGMRES_Solve(T* F, T* dx, const T atol, const int max_iter,
							const int tid, T* k, T* x, T* xk, T* CP, T tk, T h) {
	T r[SD];
	T JV[SD];
#if (ORDER == 1)
	JV_ApproxFirstOrder<T, NT, SD>(JV, dx, 0, tid, k, xk, CP, tk, h);
#elif (ORDER == 2)
	JV_ApproxSecondOrder<T, NT, SD>(JV, dx, 0, tid, xk, CP, tk, h);
#endif
	for (int j = 0; j < SD; j++) {
		r[j] = F[j] - JV[j];
	}
	T r_norm = GetNorm<T, SD>(r);

	if (r_norm <= atol) {
		return;
	}

	int iter = 0;

	while (iter < max_iter) {
		if (JFGMRES_ITERATION<T, NT, SD, R>(F, dx, r_norm, atol, tid, k, x, xk, CP, tk, h) == true) {
			//printf("\nIteration succeeded: Number of Iterations: %d\n", iter + 1);
			break;
		}
		iter++;
		// CALCULATE RESIDUAL
#if (ORDER == 1)
		JV_ApproxFirstOrder<T, NT, SD>(JV, dx, 0, tid, k, xk, CP, tk, h);
#elif (ORDER == 2)
		JV_ApproxSecondOrder<T, NT, SD>(JV, dx, 0, tid, xk, CP, tk, h);
#endif

		for (int j = 0; j < SD; j++) {
			r[j] = F[j] - JV[j];
			//printf("\n r[%d]: %.4f", j, r[j]);
		}
		r_norm = GetNorm<T, SD>(r);
	}
	//printf("\nErr: JFGMRES_Solve: Max iteration is reached!\n");
}


#include "DirectSolvers.cuh"
template<typename T, int NT, int SD, int NCP, int R>
__inline__ __device__ void FSOLVE(const int tid, T* x, T* xk, T* CP, T tk, T h, T AbsTol, const int MaxNonlinearIter, const int MaxLinearIter) {
	// tid	- ID of Actual Thread
	// x	- Actual State 
	// xk	- MidPoint State
	// CP	- Control Parameter
	// tk	- MidPoint Time
	// h	- TimeStep (half)

	// F(xk) = x - xk + h*k
	// k = f(tk, xk)	// OdeFunction

	T F[SD], FNORM{ 0 };

	T dx[SD];
	T k[SD];

	/*
	T J[SD*SD];
	J[0] = -1;
	J[1] = h / CP[0];
	J[2] = -h * CP[1];
	J[3] = -1;
	*/
	for (int iter = 0; iter < MaxNonlinearIter; iter++) {
		OdeFunction<T, T>(tid, NT, k, xk, tk, CP);
		FNORM = 0;
		for (int i = 0; i < SD; i++) {
			dx[i] = 0;
			F[i] = x[i] - xk[i] + h * k[i];
			FNORM = F[i] * F[i];
		}
		FNORM = sqrt(FNORM);
		if (FNORM < AbsTol) { return; }

		//ANAL_Solve<T, SD>(J, F, dx);
		//GMRES_Solve<T, SD, R>(J, F, dx, AbsTol, MaxLinearIter);
		JFGMRES_Solve<T, SD, R>(F, dx, AbsTol, MaxLinearIter, tid, k, x, xk, CP, tk, h);

		// UPDATE VARIABLES
		for (int i = 0; i < SD; i++) {
			xk[i] -= dx[i];
		}
	}
}

#endif // !JACOBIFREENEWTONITERATION_CUH
