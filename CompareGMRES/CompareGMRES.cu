#include <iostream>
#include <time.h> 


#define SIZE 8
#define RESTART 4
#define PRECISION double
#define ATOL 1e-12
#define MAX_ITER 3
#define EPS 1e-3

PRECISION J[SIZE * SIZE];
PRECISION b[SIZE];
PRECISION x[SIZE];
PRECISION u[SIZE];



template<typename T, int n>
void JV_approx(T* JV, T* v, const int offset) {
	T Fval1[n];
	T Fval0[n];
	
	for (int i = 0; i < n; i++) {
		Fval0[i] = 0;
		Fval1[i] = 0;
		for (int j = 0; j < n; j++)
		{
			Fval0[i] += J[i * n + j] * u[j];
			Fval1[i] += J[i * n + j] * (u[j] + EPS * v[offset * n + j]);
			JV[i] = (Fval1[i] - Fval0[i]) / EPS;
		}
	}
}


#include "..\Sympletic\GMRES.cuh"
#include "..\Sympletic\JFGMRES.cuh"

int main()
{

	srand(time(NULL));
	for (int i = 0; i < SIZE; i++) {
		b[i] = 0;
		for (int j = 0; j < SIZE; j++) {
			J[i * SIZE + j] = ((PRECISION)rand()) / ((PRECISION)RAND_MAX);
			if (i == 0) {
				x[j] = ((PRECISION)rand()) / ((PRECISION)RAND_MAX);
				u[j] = -x[j];
			}
			b[i] += J[i * SIZE + j] * x[j];
		}
	}


	std::cout << "Actual Random generated test problem:" << std::endl;
	printMatrix(J, SIZE, "J(u):");
	printVector(b, SIZE, "b(-F):");
	printVector(x, SIZE, "x:");
	printVector(u, SIZE, "u:");

	// Initialize
	for (int j = 0; j < SIZE; j++) {
		x[j] = 0.0;
	}
	GMRES_Solve<PRECISION, SIZE, RESTART>(J, b, x, ATOL, MAX_ITER);
	printVector(x, SIZE, "\nx_GMRES");


	// Initialize
	for (int j = 0; j < SIZE; j++) {
		x[j] = 0.0;
	}
	JFGMRES_Solve<PRECISION, SIZE, RESTART>(b, x, ATOL, MAX_ITER);
	printVector(x, SIZE, "\nx_JFGMRES");

	return 0;
}