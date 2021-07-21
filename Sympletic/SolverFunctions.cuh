#ifndef SOLVERFUNCTIONS_CUH
#define SOLVERFUNCTIONS_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Dependecies\AD.cuh"
#include "DirectSolvers.cuh"
#include "StationarySolvers.cuh"
#include "BICG.cuh"
#include "GMRES.cuh"

#include <cstdio>


// ---------------------------------------- General Steps ---------------------------------------------

template<typename T, int NT, int SD, int NCP>
__inline__ __host__ __device__ void ForwardEulerStep(int tid, T* AS, T* MP, T* CP, T t, T dt) {
	// tid - Thread ID
	// AS - Actual State
	// MP- MidPoint (Function value in OdeFunction) 
	// CP - Control Parameter
	// t - Actual time
	// dt - Timestep (half of the Sympletic-Step)

	OdeFunction<T, T>(tid, NT, MP, AS, t, CP);

	for (int i = 0; i < SD; i++) {
		MP[i] = AS[i] + MP[i] * dt;
	}
}

template<typename T, int SD>
__inline__ __device__ void MidPointRule(T* AS, T* MP) {

#pragma unroll
	for (int i = 0; i < SD; i++) {
		AS[i] = 2 * MP[i] - AS[i];
	}
}

// ----------------------------------------- Fix Point iteration -------------------------------------------

template<typename T, int NT, int SD, int NCP>
__inline__ __device__ void FixPointStep(int tid, T* AS, T* MP, T* CP, T t, T dt, T atol, int maxIter) {
	T Fval[SD], FNORM2;
	T atol2 = atol * atol;
	for (int k = 0; k < maxIter; k++) {
		FNORM2 = 0;
		OdeFunction<T, T>(tid, NT, Fval, MP, t, CP);
		for (int i = 0; i < SD; i++) {
			Fval[i] = AS[i] - MP[i] + dt * Fval[i];
			MP[i] = MP[i] + Fval[i];				// Fix-Point Step
			FNORM2 += Fval[i] * Fval[i];
		}
		if (FNORM2 < atol2) {
			// CONVERGED
			return;
		}
	}
}
// ----------------------------------------- Newton Iteration -----------------------------------------------
template<typename T, int NT, int SD, int NCP, int R>
__inline__ __device__ void NewtonStep(int tid, T* AS, T* MP, T* CP, T t, T dt, T atol, int maxIter, int maxLinsolveIter) {

	// MP - Midpoint (x*_k)
	// AS - Actual State (x_n)
	// t - t_(n+1/2)

	Dual<SD, T> Fval_DUAL[SD];
	Dual<SD, T> MP_DUAL[SD];
	int iSD;
	T FNORM2;
	T atol2 = atol * atol;
	T mFval[SD];		// RHS of lin Eq. -F(x*_k)
	T J[SD * SD];		// Jacobi Matrix	J_F(x*_k)
	T dx[SD];

	for (int k = 0; k < maxIter; k++) {
		FNORM2 = 0;
		// Create Dual Numbers for derivatives
		for (int i = 0; i < SD; i++) {
			MP_DUAL[i].real = MP[i];
			MP_DUAL[i].dual[i] = 1;
			dx[i] = 0;
		}
		// Derive the ODE
		OdeFunction<Dual<SD, T>, T>(tid, NT, Fval_DUAL, MP_DUAL, t, CP);
		// Get Values from the evaluated ODE
		// Fval_DUAl -> real part contains the function values
		// Fval_DUAL -> dual part contains the function derivatives

		for (int i = 0; i < SD; i++) {
			mFval[i] = MP_DUAL[i].real - AS[i] - dt * Fval_DUAL[i].real;
			printf("\n%.4f", Fval_DUAL[i].real);
			FNORM2 += mFval[i] * mFval[i];
			iSD = i * SD;
			for (int j = 0; j < SD; j++) {
				J[iSD + j] = dt * Fval_DUAL[i].dual[j];
				if (j == i) {
					J[iSD + j] -= 1;
				}
			}
		}
		if (FNORM2 < atol2) {
			// CONVERGED
			return;
		}

#if (LINSOLVER == 0)
		// DIRECT - ANALYTIC
		ANAL_Solve<T, SD>(J, mFval, dx);
#elif (LINSOLVER == 1)
		// GAUSS ELIM.
		GE_Solve<T, SD>(J, mFval, dx);
#elif (LINSOLVER == 2)
		//DFREEGAUSS ELIM.
		GE_Solve_Estimate<T, SD>(J, mFval, dx);
#elif (LINSOLVER == 3)
		// JACOBI ITER.
		JACOBI_Solve<T, SD>(J, mFval, dx, atol, maxLinsolveIter);
#elif (LINSOLVER == 4)
		// GAUSS_SEID.ITER,
		GAUSS_SEIDEL_Solve<T, SD>(J, mFval, dx, atol, maxLinsolveIter);
#elif (LINSOLVER == 5)
		// BICG ITER.
		BICG_Solve<T, SD>(J, mFval, dx, atol, maxLinsolveIter);
#elif (LINSOLVER == 6)
		// GMRES
		GMRES_Solve<T, SD, R>(J, mFval, dx, atol, maxLinsolveIter);
#endif

		// UPDATE MIDPOINT STATE
		for (int i = 0; i < SD; i++) {
			MP[i] += dx[i];
			dx[i] = 0;
		}
	}
}

 // ----------------------------------- Jacobi Free Newton Krylov (GMRES) -----------------------------------

template<typename T, int n>
__host__ __device__ void JV_approx(T* JV, T* v, const int offset, int tid,
	T* AS, T* MP, T* CP, T t, T dt) {
	T Fval0[n];
	T Fval1[n];
	T xk[n];
	int ID = offset * n;

#if (ORDER == 1)
	T C = dt / EPS;
	OdeFunction<T, T>(tid, NT, Fval0, MP, t, CP);
	for (int j = 0; j < n; j++) {
		xk[j] = MP[j] + EPS * v[ID + j];
	}
	OdeFunction<T, T>(tid, NT, Fval1, xk, t, CP);
	for (int j = 0; j < n; j++) {
		JV[j] = (Fval1[j] - Fval0[j])*C - v[ID + j];
	}
#elif (ORDER == 2)
	T C = dt / 2 / EPS;
	for (int j = 0; j < n; j++) {
		xk[j] = MP[j] - EPS * v[ID + j];
	}
	OdeFunction<T, T>(tid, NT, Fval0, xk, t, CP);
	for (int j = 0; j < n; j++) {
		xk[j] = MP[j] + EPS * v[ID + j];
	}
	OdeFunction<T, T>(tid, NT, Fval1, xk, t, CP);
	for (int j = 0; j < n; j++) {
		JV[j] = (Fval1[j] - Fval0[j])*C;
	}
#endif
}

template<typename T, int n, int m>
__inline__ __host__ void ArnoldiIteration(T* VT, T* H, T* w, const int i) {
	//MatMul<T, n>(A, VT, w, i);
	//JV_approx<T, n>(w, VT, i);	// Külsõ loop-ban áll elõ
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
__host__ __device__ bool GMRES_ITERATION(T* b, T* x, T* r, T r_norm, T atol, int tid, int NT,
										 T* AS, T* MP, T* CP, T t, T dt) { // ODE ARGS
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
		JV_approx<T, n>(w, VT, i, tid, AS, MP, CP, t, dt);
		if (tid == 0) {
			for (int r = 0; r < n; r++) {
				printf("\nw[%d]: %.6f", r, w[r]);
			}
		}
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
	return false;
}


template<typename T, int NT, int SD, int NCP, int R>
__inline__ __device__ void JacobiFreeNewtonStep(int tid, T* AS, T* MP, T* CP, T t, T dt, T atol, int maxIter) {
	// MP - Midpoint (x*_k)
	// AS - Actual State (x_n)
	// t - t_(n+1/2)

	T FNORM2;
	T atol2 = atol * atol;
	T mFval[SD];		// RHS of lin Eq. -F(x*_k)
	T r[SD];
	T dx[SD];

	for (int k = 0; maxIter < 1; k++) {
		FNORM2 = 0;
		OdeFunction<T, T>(tid, NT, mFval, MP, t, CP);
		for (int i = 0; i < SD; i++) {
			mFval[i] = MP[i] - AS[i] -  dt * mFval[i];
			r[i] = -mFval[i];		// Ha dx = 0
			FNORM2 += mFval[i] * mFval[i];
			dx[i] = 0;
		}
		if (FNORM2 < atol2) {
			// CONVERGED
			return;
		}
		FNORM2 = sqrt(FNORM2);
		GMRES_ITERATION<T, SD, R>(mFval, dx, r, FNORM2, atol, tid, NT, AS, MP, CP, t, dt);

		// UPDATE MIDPOINT STATE
		for (int i = 0; i < SD; i++) {
			MP[i] += dx[i];
			dx[i] = 0;
		}
	}
}

// ----------------------------------- Kernel ----------------------------------------------------------

template<typename T, int NT, int SD, int NCP, int R>
__global__ void SolveSympleticGPU(
	int ActiveThreads,
	T* d_TimeDomain,
	T* d_ActualState,
	T* d_ActualTime,
	T* d_ControlParameter,
	T AbsoluteTolerance,
	T TimeStep,
	int MaxNewtonIter,
	int MaxLinSolveIter) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	T TimeStep2 = TimeStep * 0.5;

	if (tid < ActiveThreads) {

		// Copy from global memory to registry
		T r_TimeDomain[2];
		T r_ActualState[SD];
		T r_MidPoint[SD];
		T r_ControlParameter[(NCP == 0 ? 1 : NCP)];

#pragma unroll
		for (int i = 0; i < 2; i++) {
			r_TimeDomain[i] = d_TimeDomain[tid + i * NT];
			//printf("\n Hello, I'm tid: %d, my r_TimeDomain[%d] is: %.2f", tid, i, r_TimeDomain[i]);
		}

#pragma unroll
		for (int i = 0; i < SD; i++) {
			r_ActualState[i] = d_ActualState[tid + i * NT];
			//printf("\n Hello, I'm tid: %d, my r_ActualState[%d] is: %.2f", tid, i, r_ActualState[i]);
			r_MidPoint[i] = 0;
		}

#pragma unroll
		for (int i = 0; i < NCP; i++) {
			r_ControlParameter[i] = d_ControlParameter[tid + i * NT];
			//printf("\n Hello, I'm tid: %d, my r_ControlParameter[%d] is: %.2f", tid, i, r_ControlParameter[i]);
		}

		// Initialization
		T r_ActualTime = r_TimeDomain[0];		

		// Stepper
		while (r_ActualTime <= r_TimeDomain[1]) {

			// Forward step - estimate
			ForwardEulerStep<T, NT, SD, NCP>(tid, r_ActualState, r_MidPoint, r_ControlParameter, r_ActualTime, TimeStep2);

			if (tid == 0 && r_ActualTime == 0.0) {
				printf("\n xM: [%.5f, %.5f]", r_MidPoint[0], r_MidPoint[1]);
			}

			// Switch method
#if (SOLVER == 0)
			FixPointStep<T, NT, SD, NCP>(tid, r_ActualState, r_MidPoint, r_ControlParameter, r_ActualTime+TimeStep2, TimeStep2, AbsoluteTolerance, MaxNewtonIter);
#elif (SOLVER == 1)
			NewtonStep<T, NT, SD, NCP, R>(tid, r_ActualState, r_MidPoint, r_ControlParameter, r_ActualTime + TimeStep2, TimeStep2, AbsoluteTolerance, MaxNewtonIter, MaxLinSolveIter);
#elif (SOLVER == 2)
			JacobiFreeNewtonStep<T, NT, SD, NCP, R>(tid, r_ActualState, r_MidPoint, r_ControlParameter, r_ActualTime + TimeStep2, TimeStep2, AbsoluteTolerance, MaxNewtonIter);
#endif 

			/*if (tid == 0 && r_ActualTime == 0.0) {
				printf("\n xM: [%.5f, %.5f]", r_MidPoint[0], r_MidPoint[1]);
			}*/


			MidPointRule<T, SD>(r_ActualState, r_MidPoint);
			r_ActualTime += TimeStep;

			if (tid == 0 && r_ActualTime != -10.0) {
				printf("\n %.6f, %.6f, %.6f", r_ActualTime, r_ActualState[0], r_ActualState[1]);
			}
		}

		// Finalization

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
	}
}


#endif