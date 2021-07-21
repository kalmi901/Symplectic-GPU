#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>

#if (SOLVER == 0)
#include "FixedPointIteration.cuh"
#elif (SOLVER == 1)
#include "NewtonIteration.cuh"
#elif (SOLVER == 2)
#include "JacobiFreeNewtonIteration.cuh"
#endif


// ------------------------ General Functions ---------------------

template<typename T, int NT, int SD>
__inline__ __device__ void EulerStep(const int tid, T* x, T* xk, T* CP, T t, T h) {
	// tid - ID of Actual Thread
	// x   - Actual State
	// xk  - MidPoint State
	// CP  - Control Parameters
	// t   - Actual Time
	// h   - Timestep (half of Symplettic-Step)

	T k[SD];
	OdeFunction<T, T>(tid, NT, k, x, t, CP);

#pragma unroll
	for (int i = 0; i < SD; i++) {
		xk[i] = x[i] + k[i] * h;
	}
}


template<typename T, int SD>
__inline__ __device__ void UpdateActualState(T* x, T* xk) {

	// x  - ActualState (NexState)
	// xk - MidState (Converged)

#pragma unroll
	for (int i = 0; i < SD; i++) {
		x[i] = 2 * xk[i] - x[i];
	}
}


// ---------------------------- Kernel ----------------------------
// Called by Solver.Solve()

template<typename T, int NT, int SD, int NCP, int R>
__global__ void SolveSympleticGPU(
	const int NumberOfActiveThreads,
	T* d_TimeDomain,
	T* d_ActualState,
	T* d_ActualTime,
	T* d_ControlParameter,
	const T AbsoluteTolerance,
	const T TimeStep,
	const int MaxNonlinerIter,
	const int MaxLinearIter)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;


	if (tid < NumberOfActiveThreads) {

		// INITIALIZATION
		// COPY DATA FROM TGE GLOBAL MEMORY TO REGISTERS
		T r_TimeDomain[2];
		T r_ActualState[SD];
		T r_MidPointState[SD];
		T r_ControlParameter[(NCP == 0 ? 1 : NCP)];

#pragma unroll
		for (int i = 0; i < 2; i++) {
			r_TimeDomain[i] = d_TimeDomain[tid + i * NT];
		}

#pragma unroll
		for (int i = 0; i < SD; i++) {
			r_ActualState[i] = d_ActualState[tid + i * NT];
		}

#pragma unroll
		for (int i = 0; i < NCP; i++) {
			r_ControlParameter[i] = d_ControlParameter[tid + i * NT];
		}

		T r_ActualTime		= r_TimeDomain[0];
		T r_HalfStep		= TimeStep * 0.5;
		T r_MidPointTime	= r_ActualTime + r_HalfStep;

		// STEPPER
		while (r_ActualTime <= r_TimeDomain[1])
		{
#if (EULERPREDICT == 0)
			for (int i = 0; i < SD; i++) {
				r_MidPointState[i] = r_ActualState[i];
			}
#elif (EULERPREDICT == 1)
			// PREDICT MIDPOINT VIA AN EULER STEP
			EulerStep<T, NT, SD>(tid, r_ActualState, r_MidPointState, r_ControlParameter, r_ActualTime, r_HalfStep);
#endif

			// SWITCH THE NONLINEAR SOLVER
#if (SOLVER == 0)
			// FIXED-POINT ITERATION
			FSOLVE<T, NT, SD, NCP>(tid, r_ActualState, r_MidPointState, r_ControlParameter, r_MidPointTime, r_HalfStep, AbsoluteTolerance, MaxNonlinerIter);
#elif (SOLVER == 1)
			// NEWTON ITERATION (FULL JACOBI)
			FSOLVE<T, NT, SD, NCP, R>(tid, r_ActualState, r_MidPointState, r_ControlParameter, r_MidPointTime, r_HalfStep, AbsoluteTolerance, MaxNonlinerIter, MaxLinearIter);
#elif (SOLVER == 2)
			// JACOBIFREE-NEWTON KRYLOV ITERATION
			FSOLVE<T, NT, SD, NCP, R>(tid, r_ActualState, r_MidPointState, r_ControlParameter, r_MidPointTime, r_HalfStep, AbsoluteTolerance, MaxNonlinerIter, MaxLinearIter);
#endif

			// UPDATE ACTUAL STATE AND TIME
			UpdateActualState<T, SD>(r_ActualState, r_MidPointState);
			r_ActualTime	+= TimeStep;
			r_MidPointTime	+= TimeStep;

			if (tid == 0 && r_ActualTime != -10.0) {
				printf("\n %.6f, %.6f, %.6f", r_ActualTime, r_ActualState[0], r_ActualState[1]);
			}


		}	// STEPPER

		// FINALIZATION
		// COPY DATA TO THE GLOBAL MEMORY FROM REGISTER
#pragma unroll
		for (int i = 0; i < 2; i++) {
			d_TimeDomain[tid + i * NT] = r_TimeDomain[i];
		}

#pragma unroll
		for (int i = 0; i < SD; i++) {
			d_ActualState[tid + i * NT] = r_ActualState[i];
		}

#pragma unroll
		for (int i = 0; i < NCP; i++) {
			d_ControlParameter[tid + i * NT] = d_ControlParameter[i];
		}
	} // if (tid < NumberOfActiveThreads)
}


