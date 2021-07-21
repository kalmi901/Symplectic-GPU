#ifndef TOOLS_CUH
#define TOOLS_CUH

#include <iostream>
#include <string>
#include <cuda.h>
#include <device_launch_parameters.h>
#include "fast_math.cuh"

void printMatrix(double*A, int n, std::string name)
{
	std::cout << name << std::endl;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			std::cout.width(10);
			std::cout.precision(3);
			std::cout << std::left << A[i * n + j] << "   ";
		}
		std::cout << std::endl;
	}
	std::cin.get();
}

void printVector(double*A, int n, std::string name)
{
	std::cout << name << std::endl;
	std::cout << "[ ";
	for (int i = 0; i < n; i++)
	{
		std::cout.width(10);
		std::cout.precision(4);
		std::cout << std::left << A[i];
	}
	std::cout << "]" << std::endl;
	std::cin.get();
}

void printMatrix(double*A, int n, int m, std::string name)
{
	std::cout << name << " " << std::endl;
	std::cout.width(10);
	std::cout.precision(4);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			std::cout.width(16);
			std::cout << std::right << A[i * m + j] << "   ";
		}
		std::cout << std::endl;
	}
	std::cin.get();
}

// Tools for direct solvers

template<typename T>
inline __host__ __device__ void BackwardSubstitution(T* U, T* b, T* x, const int n) {
	// U - Uppter Triangle Matrix of size n x n
	// b - rhs vector
	// x - solution vector
	T s;
	for (int i = n - 1; i >= 0; i--) {
		s = 0;
		for (int j = i; j < n; j++) {
			s += U[i * n + j] * x[j];
		}
		x[i] = (b[i] - s) / U[i * n + i];
	}
}
 
// GMRES-hez
template<typename T>
inline __host__ __device__ void BackwardSubstitution(T* U, T* b, T* x, const int n, const int offset) {
	T s;
	for (int i = offset - 1; i >= 0; i--) {
		s = 0;
		for (int j = i + 1; j < offset; j++) {
			s += U[i * n + j] * x[j];
		}
		x[i] = (b[i] - s) / U[i * n + i];
	}
}


template<typename T>
inline __host__ __device__ void BackwardSubstitution_FDiv(T* U, T* b, T* x, const int n) {
	// U - Uppter Triangle Matrix of size n x n
	// b - rhs vector
	// x - solution vector
	T s, r_U;
	for (int i = n - 1; i >= 0; i--) {
		s = 0;
		for (int j = i; j < n; j++) {
			s += U[i * n + j] * x[j];
		}
		r_U = copysign(f_div2(abs(U[i * n + i])), U[i * n + i]);
		x[i] = (b[i] - s) * r_U;
	}
}

template<typename T, int n>
inline __host__ __device__ void PivotRow(T* A, T* b, int j) {
	// A - Matrix of size n x n
	// b - rhs vector of size n
	// j - index of actual diagonal element of A[j, j]
	T maxA = 0;
	int jmax = j;
	T temp;		
	for (int k = j; k < n; k++){
		temp = abs(A[k * n + j]);
		if (temp > maxA) {
			maxA = temp;
			jmax = k;
		}
	}
	if (j != jmax) {
		// Swap the rows
		for (int k = 0; k < n; k++) {
			temp = A[jmax *  n + k];
			A[jmax * n + k] = A[j * n + k];
			A[j * n + k] = temp;
		}
		temp = b[jmax];
		b[jmax] = b[j];
		b[j] = temp;
	}
}

// Tools for iterative solvers


template<typename T, int n>
__host__ __device__ void CalculateResidual(T* A, T*b, T* x, T* r) {
	// Calculatt the residual vector as r = b - A*x equation
	// A is a known matrix of size n*n
	// x is a known vector (guess) of size n
	// b is the rhs vector of size n
	// r is the residual vector of size n

	T SUM;
	for (int i = 0; i < n; i++) {
		SUM = 0;
		for (int j = 0; j < n; j++) {
			SUM += A[i * n + j] * x[j];
		}
		r[i] = b[i] - SUM;
	}
}

template<typename T, int n>
__host__ __device__ T GetNorm(T* v) {
	// Calculate the norm of a vector v of size n
	T result{ 0 };
	for (int i = 0; i < n; i++) {
		result += v[i] * v[i];
	}
	return sqrt(result);
}


// MATRIX VECTOR MANIPULATION

template<typename T, int n>
__host__ __device__ T VecDot(T* x, T* y) {
	// Calculate the dot product of two vector
	T result{ 0 };
	for (int i = 0; i < n; i++) {
		result += x[i] * y[i];
	}
	return result;
}

template<typename T, int n>
__host__ __device__ T VecDot(T* x, T* y, const int m) {
	// Calculate the dot product of two vector
	// y is a matrox composed by vectors of size n
	T result{ 0 };
	int nm{ n * m };
	for (int i = 0; i < n; i++) {
		result += x[i] * y[nm + i];
	}
	return result;
}

template<typename T, int n>
__host__ __device__ void MatMul(T* A, T* x, T* y) {
	// Calculate the matrix-vector multiplacation y = A*x
	// A is a square matrix of size n*n
	// x is a vector of size n or a matrix of vectors -> m denotes the row of x
	// y is a vector (result) of size n
	for (int i = 0; i < n; i++) {
		y[i] = 0;
		for (int j = 0; j < n; j++) {
			y[i] += A[i * n + j] * x[j];
		}
	}
}

template<typename T, int n>
__host__ __device__ void MatMul(T* A, T* x, T* y, const int m) {
	// Calculate the matrix-vector multiplacation y = A*x
	// A is a square matrix of size n*n
	// x is a metrix compised by vectors of size n
	// m denotes the specific row of x
	// y is a vector (result) of size n
	int in, nm{ n * m };
	for (int i = 0; i < n; i++) {
		y[i] = 0;
		in = i * n;
		for (int j = 0; j < n; j++) {
			y[i] += A[in + j] * x[nm + j];
		}
	}
}

template<typename T>
__host__ __device__ void TransMatMul(T* A, T* x, T* y, const int n) {
	// Calculate the transpose(matrix)-vector multiplacation y = (A^T)*x
	// A is a square matrix of size n*n
	// x is a vector of size n
	// y is a vector (result) of size n
	for (int i = 0; i < n; i++) {
		y[i] = 0;
		for (int j = 0; j < n; j++) {
			y[i] += A[j * n + i] * x[j];
		}
	}
}

template<typename T, int n>
__host__ __device__ void TransMatMul(T* A, T* x, T* y, const int m)
{
	for (int i = 0; i < n; i++) {
		y[i] = 0;
		for (int j = 0; j < m; j++) {
			y[i] += A[j * n + i] * x[j];
		}
	}
}

#endif