#ifndef SYSTEMDEFINITION_CUH
#define SYSTEMDEFINITION_CUH
#include "cuda_runtime.h"

template<typename T1, typename T2>
__host__ __device__ void OdeFunction(const int tid, const int NT,
									T1* F, T1* X, T2 t, 
									T2* cPar)
{

}

#endif