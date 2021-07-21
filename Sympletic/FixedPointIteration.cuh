#ifndef FIXEDPOINTITERATION_CUH
#define FIXEDPOINTITERATION_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>


template<typename T, int NT, int SD, int NCP>
__inline__ __device__ void FSOLVE(const int tid, T* x, T* xk, T* CP, T tk, T h, T AbsTol, const int MaxNonlinerIter) {
	// tid	- ID of Actual Thread
	// x	- Actual State 
	// xk	- MidPoint State
	// CP	- Control Parameter
	// tk	- MidPoint Time
	// h	- TimeStep (half)

	// F(xk) = x - xk + h*k
	// k = f(tk, xk)	// OdeFunction

	T F[SD], k[SD], FNORM{ 0 };
	OdeFunction<T, T>(tid, NT, k, xk, tk, CP);
	for (int i = 0; i < SD; i++) {
		F[i]	= x[i] - xk[i] + h * k[i];
		FNORM	+= F[i] * F[i];
	}
	if (sqrt(FNORM) < AbsTol) { return; }

	for (int iter = 0; iter < MaxNonlinerIter; iter++) {
		for (int i = 0; i < SD; i++) {
			xk[i] += F[i];
		}
		OdeFunction<T, T>(tid, NT, k, xk, tk, CP);
		FNORM = 0;
		for (int i = 0; i < SD; i++) {
			F[i]	= x[i] - xk[i] + h * k[i];
			FNORM	+= F[i] * F[i];
		}
		if (sqrt(FNORM) < AbsTol) { return; }
	}

}
#endif // FIXEDPOINTITERATION_CUH