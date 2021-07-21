#ifndef SYSTEMDEFINITION_CUH
#define SYSTEMDEFINITION_CUH
#include "cuda_runtime.h"

template<typename T1, typename T2>
__host__ __device__ void OdeFunction(const int tid, const int NT,
	T1* F, T1* X, T2 t,
	T2* cPar) {

	//F[0] = X[1] / cPar[0];
	//F[1] = -cPar[1] * X[0];

#pragma unroll
	for (int i = 0; i < N; i++) {
		F[i] = X[N + i] / cPar[i];
	}
	F[N] = -cPar[N] * X[0] + cPar[N + 1] * (X[1] - X[0]);

#pragma unroll
	for (int i = 1; i < N - 1; i++) {
		F[N + i] = -cPar[N + i] * (X[i] - X[i - 1]) + cPar[N + 1 + i] * (X[i + 1] - X[i]);
	}
	F[2 * N - 1] = -cPar[2 * N] * X[N - 1] - cPar[2 * N - 1] * (X[N - 1] - X[N - 2]);


	/*if (tid == 0)
	{
		for (int i = 0; i < N; i++) {
			printf("\n%.5f", cPar[i]);
		}
	}*/
}

#endif