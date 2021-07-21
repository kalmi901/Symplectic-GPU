#include <iostream>
#include <time.h> 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "..\Sympletic\DirectSolvers.cuh"
#include "..\Sympletic\BICG.cuh"
#include "..\Sympletic\GMRES.cuh"
#include "..\Sympletic\StationarySolvers.cuh"


#define SIZE 3
#define RESTART 3
#define PRECISION double
#define ATOL 1e-12
#define MAX_ITER 1000

template<typename T, int n>
void InitializeProblem(T* A0, T* b0, T* x0, T* A, T* b, T* x) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			A[i * n + j] = A0[i * n + j];
			if (i == 0) {
				x[j] = 0.0;
				b[j] = b0[j];
			}
		}
	}
}

enum Direction {
	HostToDevice,
	DeviceToHost
};

template<typename T, int n>
void SyncronizeData(T* A, T* b, T* x, T* d_A, T* d_b, T* d_x, Direction dir) {
	switch (dir)
	{
	case HostToDevice:
		cudaMemcpy(d_A, A, n*n * sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, n * sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_x, x, n * sizeof(T), cudaMemcpyHostToDevice);
		break;
	case DeviceToHost:
		cudaMemcpy(x, d_x, n * sizeof(T), cudaMemcpyDeviceToHost);
		break;
	default:
		break;
	}
}

enum Solver {
	GE_SOLVE,
	GE_SOLVE_ESTIMATE,
	ANAL_SOLVE,
	JACOBI_SOLVE,
	GAUSS_SEIDEL_SOLVE,
	BICG_SOLVE,
	GMRES_SOLVE
};

template<typename T, int n>
__global__ void TestOnGPU(T* d_A, T* d_b, T* d_x, Solver LinearSolve) {
	//printf("BlockIdx: %d, TreadIdx: %d\n", blockIdx.x, threadIdx.x);
	//GE_Solve<T, n>(A, b, x);
	//GE_Solve_Estimate<T, n>(A, b, x);
	//GE_Solve_Fdiv<T, n>(A, b, x);


	switch (LinearSolve)
	{
	case GE_SOLVE:
		GE_Solve<T, n>(d_A, d_b, d_x);
		break;
	case GE_SOLVE_ESTIMATE:
		GE_Solve_Estimate<T, n>(d_A, d_b, d_x);
		break;
	case ANAL_SOLVE:
		ANAL_Solve<T, n>(d_A, d_b, d_x);
		break;
	case JACOBI_SOLVE:
		JACOBI_Solve<T, n>(d_A, d_b, d_x, ATOL, MAX_ITER);
		break;
	case GAUSS_SEIDEL_SOLVE:
		GAUSS_SEIDEL_Solve<T, n>(d_A, d_b, d_x, ATOL, MAX_ITER);
		break;
	case BICG_SOLVE:
		BICG_Solve<T, n>(d_A, d_b, d_x, ATOL, MAX_ITER);
		break;
	case GMRES_SOLVE:
		GMRES_Solve<T, n, RESTART>(d_A, d_b, d_x, ATOL, MAX_ITER);
		break;
	default:
		break;
	}
}


int main() {

	// INITIALIZE MATRIX-VECTOR ARRAYS:
	PRECISION A0[SIZE * SIZE], A[SIZE * SIZE];
	PRECISION b0[SIZE], b[SIZE];
	PRECISION x0[SIZE], x[SIZE];

	// GENERATE RANDOM PROBLEM
	
	srand(time(NULL));
	for (int i = 0; i < SIZE; i++) {
		b0[i] = 0;
		for (int j = 0; j < SIZE; j++) {
			A0[i * SIZE + j] = ((PRECISION)rand()) / ((PRECISION)RAND_MAX);
			if (i == 0) {
				x0[j] = ((PRECISION)rand()) / ((PRECISION)RAND_MAX);
			}
			b0[i] += A0[i * SIZE + j] * x0[j];
		}
	}
	
	/*
	A0[0] = 2.0; A0[1] = 3.0; A0[2] = 2.0;
	A0[3] = 4.0; A0[4] = -1.0; A0[5] = 5.0;
	A0[6] = 2.0; A0[7] = 3.0; A0[8] = -7.0;

	b0[0] = 12.0; b0[1] = 3.5; b0[2] = 7.5;
	x0[0] = 0.0; x0[1] = 0.0; x0[2] = 0.0;
	//x0[0] = 1.76; x0[1] = 1.96; x0[2] = 0.0523;
	*/


	// INITIALIZE GPU ARRAYS
	PRECISION *d_A, *d_b, *d_x;
	cudaMalloc((void**)&d_A, SIZE*SIZE * sizeof(PRECISION));
	cudaMalloc((void**)&d_b, SIZE * sizeof(PRECISION));
	cudaMalloc((void**)&d_x, SIZE * sizeof(PRECISION));


	std::cout << "Actual Random generated test problem:" << std::endl;
	printMatrix(A0, SIZE, "A0");
	printVector(b0, SIZE, "b0");
	printVector(x0, SIZE, "x0");
#if (GMRES == 0)
	// Test Different Solvers on CPU
	std::cout << "..... Test Different Solvers ......" << std::endl << std::endl;
	std::cout << "--------- Direct Solvers ----------" << std::endl;

	// GAUSSIAN ELIMINATION -------------------------
	std::cout << "Gaussian Elimination:" << std::endl;
	InitializeProblem<PRECISION, SIZE>(A0, b0, x0, A, b, x);
	SyncronizeData<PRECISION, SIZE>(A, b, x, d_A, d_b, d_x, HostToDevice);
	//std::cout << "Inputs Arrays:" << std::endl;
	//printMatrix(A, SIZE, "A");
	//printVector(b, SIZE, "b");
	//printVector(x, SIZE, "x");

	GE_Solve<PRECISION, SIZE>(A, b, x);
	TestOnGPU<PRECISION, SIZE> <<<1, 1>>> (d_A, d_b, d_x, GE_SOLVE);

	std::cout << "Output Arrays (results):" << std::endl;
	//printMatrix(A, SIZE, "A");
	//printVector(b, SIZE, "b");
	printVector(x, SIZE, "x_CPU:");
	SyncronizeData<PRECISION, SIZE>(A, b, x, d_A, d_b, d_x, DeviceToHost);
	printVector(x, SIZE, "x_GPU:");


	// "DIVISION FREE" GAUSSIAN ELIMINATION ---------
	std::cout << "Div-Free-Gaussian Elimination (Estimate Division):" << std::endl;
	InitializeProblem<PRECISION, SIZE>(A0, b0, x0, A, b, x);
	SyncronizeData<PRECISION, SIZE>(A, b, x, d_A, d_b, d_x, HostToDevice);
	//std::cout << "Inputs Arrays:" << std::endl;
	//printMatrix(A, SIZE, "A");
	//printVector(b, SIZE, "b");
	//printVector(x, SIZE, "x");

	GE_Solve_Estimate<PRECISION, SIZE>(A, b, x);
	TestOnGPU<PRECISION, SIZE> <<<1, 1>>> (d_A, d_b, d_x, GE_SOLVE_ESTIMATE);

	std::cout << "Output Arrays (results):" << std::endl;
	//printMatrix(A, SIZE, "A");
	//printVector(b, SIZE, "b");
	printVector(x, SIZE, "x_CPU:");
	SyncronizeData<PRECISION, SIZE>(A, b, x, d_A, d_b, d_x, DeviceToHost);
	printVector(x, SIZE, "x_GPU:");


	// DIRECT ANALYTIC ----------------------------
	std::cout << "DIRECT ANALYTIC:" << std::endl;
	InitializeProblem<PRECISION, SIZE>(A0, b0, x0, A, b, x);
	SyncronizeData<PRECISION, SIZE>(A, b, x, d_A, d_b, d_x, HostToDevice);
	//std::cout << "Inputs Arrays:" << std::endl;
	//printMatrix(A, SIZE, "A");
	//printVector(b, SIZE, "b");
	//printVector(x, SIZE, "x");

	ANAL_Solve<PRECISION, SIZE>(A, b, x);
	TestOnGPU<PRECISION, SIZE> <<<1, 1>>> (d_A, d_b, d_x, ANAL_SOLVE);

	std::cout << "Output Arrays (results):" << std::endl;
	//printMatrix(A, SIZE, "A");
	//printVector(b, SIZE, "b");
	printVector(x, SIZE, "x_CPU:");
	SyncronizeData<PRECISION, SIZE>(A, b, x, d_A, d_b, d_x, DeviceToHost);
	printVector(x, SIZE, "x_GPU:");

	std::cout << "--- Stationary Iterative Solvers --" << std::endl;

	// JACOBI ITERATION
	std::cout << "Jacobi Iteration:" << std::endl;
	InitializeProblem<PRECISION, SIZE>(A0, b0, x0, A, b, x);
	SyncronizeData<PRECISION, SIZE>(A, b, x, d_A, d_b, d_x, HostToDevice);
	//std::cout << "Inputs Arrays:" << std::endl;
	//printMatrix(A, SIZE, "A");
	//printVector(b, SIZE, "b");
	//printVector(x, SIZE, "x");

	JACOBI_Solve<PRECISION, SIZE>(A, b, x, ATOL, MAX_ITER);
	TestOnGPU<PRECISION, SIZE> <<<1, 1 >>> (d_A, d_b, d_x, JACOBI_SOLVE);

	std::cout << "Output Arrays (results):" << std::endl;
	//printMatrix(A, SIZE, "A");
	//printVector(b, SIZE, "b");
	printVector(x, SIZE, "x_CPU:");
	SyncronizeData<PRECISION, SIZE>(A, b, x, d_A, d_b, d_x, DeviceToHost);
	printVector(x, SIZE, "x_GPU:");

	// GAUSS-SEIDEL ITERATION
	std::cout << "Gauss-Seidel Iteration:" << std::endl;
	InitializeProblem<PRECISION, SIZE>(A0, b0, x0, A, b, x);
	SyncronizeData<PRECISION, SIZE>(A, b, x, d_A, d_b, d_x, HostToDevice);
	//std::cout << "Inputs Arrays:" << std::endl;
	//printMatrix(A, SIZE, "A");
	//printVector(b, SIZE, "b");
	//printVector(x, SIZE, "x");

	GAUSS_SEIDEL_Solve<PRECISION, SIZE>(A, b, x, ATOL, MAX_ITER);
	TestOnGPU<PRECISION, SIZE> <<<1, 1>>> (d_A, d_b, d_x, GAUSS_SEIDEL_SOLVE);

	std::cout << "Output Arrays (results):" << std::endl;
	//printMatrix(A, SIZE, "A");
	//printVector(b, SIZE, "b");
	printVector(x, SIZE, "x_CPU:");
	SyncronizeData<PRECISION, SIZE>(A, b, x, d_A, d_b, d_x,DeviceToHost);
	printVector(x, SIZE, "x_GPU:");
	
	std::cout << "---- Krylov Iterative Solvers -----" << std::endl;

	// BICG ITERATION
	std::cout << "BICG Iteration:" << std::endl;
	InitializeProblem<PRECISION, SIZE>(A0, b0, x0, A, b, x);
	SyncronizeData<PRECISION, SIZE>(A, b, x, d_A, d_b, d_x, HostToDevice);
	//std::cout << "Inputs Arrays:" << std::endl;
	//printMatrix(A, SIZE, "A");
	//printVector(b, SIZE, "b");
	//printVector(x, SIZE, "x");


	BICG_Solve<PRECISION, SIZE>(A, b, x, ATOL, MAX_ITER);
	TestOnGPU<PRECISION, SIZE> <<<1, 1>>> (d_A, d_b, d_x, BICG_SOLVE);

	std::cout << "Output Arrays (results):" << std::endl;
	//printMatrix(A, SIZE, "A");
	//printVector(b, SIZE, "b");
	printVector(x, SIZE, "x_CPU:");
	SyncronizeData<PRECISION, SIZE>(A, b, x, d_A, d_b, d_x, DeviceToHost);
	printVector(x, SIZE, "x_GPU:");

//#elif (GMRES == 1)
	// GMRES(m) Iteration
	std::cout << "GMRES(m) Iteration:" << std::endl;
	InitializeProblem<PRECISION, SIZE>(A0, b0, x0, A, b, x);
	SyncronizeData<PRECISION, SIZE>(A, b, x, d_A, d_b, d_x, HostToDevice);
	//std::cout << "Inputs Arrays:" << std::endl;
	//printMatrix(A, SIZE, "A");
	//printVector(b, SIZE, "b");
	//printVector(x, SIZE, "x");

	GMRES_Solve<PRECISION, SIZE, RESTART>(A, b, x, ATOL, MAX_ITER);
	TestOnGPU<PRECISION, SIZE> <<<1, 1 >>> (d_A, d_b, d_x, GMRES_SOLVE);

	std::cout << "Output Arrays (results):" << std::endl;
	//printMatrix(A, SIZE, "A");
	//printVector(b, SIZE, "b");
	printVector(x, SIZE, "x_CPU");
	SyncronizeData<PRECISION, SIZE>(A, b, x, d_A, d_b, d_x, DeviceToHost);
	printVector(x, SIZE, "x_GPU");
#endif

	cudaFree(d_A);
	cudaFree(d_b);
	cudaFree(d_x);

	return 0;
}
