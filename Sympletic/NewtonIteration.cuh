#ifndef FIXEDPOINTITERATION_CUH
#define FIXEDPOINTITERATION_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Dependecies\AD.cuh"
#include <cmath>



#if (LINSOLVER < 3)
#include "DirectSolvers.cuh"
#elif (LINSOLVER > 2 && LINSOLVER < 5)
#include "StationarySolvers.cuh"
#elif (LINSOLVER == 5)
#include "BICG.cuh"
#elif (LINSOLVER == 6)
#include "GMRES.cuh"
#endif

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
	/*
	T k[SD];
	OdeFunction<T, T>(tid, NT, k, xk, tk, CP);
	for (int j = 0; j < SD; j++) {
		F[j] = x[j] - xk[j] + h * k[j];
		FNORM += F[j] * F[j];
	}
	if (sqrt(FNORM) < AbsTol) { return; }
	*/

	T J[SD*SD];
	T dx[SD];

	Dual<SD, T> k_Dual[SD];
	Dual<SD, T> xk_Dual[SD];

	for (int i = 0; i < SD; i++) {
		xk_Dual[i].real = xk[i];
		xk_Dual[i].dual[i] = 1;
	}

	int iSD;
	for (int iter = 0; iter < MaxNonlinearIter; iter++) {
		OdeFunction<Dual<SD, T>, T>(tid, NT, k_Dual, xk_Dual, tk, CP);
		FNORM = 0;
		for (int i = 0; i < SD; i++) {
			dx[i] = 0;
			F[i] = x[i] - xk[i] + h * k_Dual[i].real;
			//printf("\n %.4f", k_Dual[i].real);
			FNORM = F[i] * F[i];
			iSD = i * SD;
			for (int j = 0; j < SD; j++) {
				J[iSD + j] = h * k_Dual[i].dual[j];
				if (j == i) {
					J[iSD + j] -= 1;
				}
				if (tid == 0 && tk == h) {
					//printf("\nJ[%d]: %.5f", iSD + j, J[iSD + j]);
				}
			}
		}
		if (sqrt(FNORM) < AbsTol) { return; }

#if (LINSOLVER == 0)
		// DIRECT - ANALITIC
		ANAL_Solve<T, SD>(J, F, dx);
#elif (LINSOLVER == 1)
		// DIRECT - GAUSS
		GE_Solve<T, SD>(J, F, dx);
#elif (LINSOLVER == 2)
		// DIRECT - DIVISION FREE GAUSS
		GE_Solve_Estimate<T, SD>(J, F, dx);
#elif (LINSOLVER == 3)
		// STATIONARY - JACOBI ITERATION
		JACOBI_Solve<T, SD>(J, F, dx, AbsTol, MaxLinearIter);
#elif (LINSOLVER == 4)
		// STATIONARY - GAUSS-SEIDEL ITERATION
		GAUSS_SEIDEL_Solve<T, SD>(J, F, dx, AbsTol, MaxLinearIter);
#elif (LINSOLVER == 5)
		// KRYLOV SUBSPACE - BICG
		BICG_Solve<T, SD>(J, F, dx, AbsTol, MaxLinearIter);
#elif (LINSOLVER == 6)
		// KRYLOV SUBSPACE - GMRES (RESTARTED)
		GMRES_Solve<T, SD, R>(J, F, dx, AbsTol, MaxLinearIter);
#endif

		// UPDATE VARIABLES
		for (int i = 0; i < SD; i++) {
			xk[i]				-= dx[i];
			xk_Dual[i].real		= xk[i];
			xk_Dual[i].dual[i]  = 1;
		}

	}
}
#endif // FIXEDPOINTITERATION_CUH