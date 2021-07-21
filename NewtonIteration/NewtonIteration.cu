#include <iostream>
#include <math.h>
#include "..\Symplectic\DirectSolvers.cuh"
#include "..\Symplectic\Dependecies\AD.cuh"

double x0[3] = { 1, 2, 3 };
double dx[3] = { 0, 0, 0 };
double x[3];
int maxiter = 10;

template<typename T>
void NonlinearFunction(T* F, T* x) {
	/*
	F[0] = 3 * x[0] - cos(x[1] * x[2]) - 1.5;
	F[1] = 4 * x[0] * x[0] - 625 * x[1] * x[1] + 2 * x[2] - 1;
	F[2] = 20 * x[2] + exp(-1 * x[0] * x[1]) + 9;
	*/
	F[0] = x[0] * x[0] - 2 * x[0] + x[1] * x[1] - x[2] + 1;
	F[1] = x[0] * x[1] * x[1] - x[0] - 3 * x[1] + x[1] * x[2] + 2;
	F[2] = x[0] * x[2] * x[2] - 3 * x[2] + x[1] * x[2] * x[2] + x[0] * x[1];
}

void Jacobian(double* J, double* x) {
	/*
	double x1 = x[0], x2 = x[1], x3 = x[2];
	

	J[0] = 3;
	J[1] = x3 * sin(x2*x3);
	J[2] = x2 * sin(x2*x3);

	J[3] = 8 * x1;
	J[4] = -1250 * x2;
	J[5] = 2;

	J[6] = -x2 * exp(-x1 * x2);
	J[7] = -x1 * exp(-x1 * x2);
	J[8] = 20;
	*/

	J[0] = 2 * x[0] - 2;
	J[1] = 2 * x[1];
	J[2] = -1;

	J[3] = x[1] * x[1] - 1;
	J[4] = 2 * x[0] * x[1] - 3 + x[2];
	J[5] = x[1];

	J[6] = x[2] * x[2] + x[1];
	J[7] = x[2] * x[2] + x[0];
	J[8] = 2 * x[0] * x[2] - 3 + 2 * x[1] * x[2];
}

void NewtonIterationAnalytic()
{
	double F[3];
	double J[9];
	//double x[3];
	for (int j = 0; j < 3; j++) {
		x[j] = x0[j];
	}
	NonlinearFunction<double>(F, x);
	printf("\nx:  %.6f,  %.6f,  %.6f  | err:   %.3e", x[0], x[1], x[2], GetNorm<double, 3>(F));

	for (int k = 0; k < maxiter; k++) {
		Jacobian(J, x);
		//printMatrix(J, 3, 3, "J");
		/*for (int j = 0; j < 3; j++) {
			F[j] = -F[j];
		}*/	
		ANAL_Solve<double, 3>(J, F, dx);
		for (int j = 0; j < 3; j++) {
			//x[j] += dx[j];
			x[j] -= dx[j];
		}
		NonlinearFunction<double>(F, x);
		printf("\nx:  %.6f,  %.6f,  %.6f  | err:   %.3e", x[0], x[1], x[2], GetNorm<double, 3>(F));
	}
}


#include "..\Symplectic\GMRES.cuh"
void NewtonIterationDual() {
	Dual<3, double>FDual[3];
	Dual<3, double>xDual[3];
	double F[3];
	double J[9];
	//double x[3];
	for (int j = 0; j < 3; j++) {
		x[j] = x0[j];
		dx[j] = 0;
	}
	for (int j = 0; j < 3; j++) {
		xDual[j].real = x[j];
		xDual[j].dual[j] = 1;
	}

	for (int k = 0; k < maxiter; k++) {
		NonlinearFunction<Dual<3, double>>(FDual, xDual);

		for (int i = 0; i < 3; i++) {
			//F[i] = -FDual[i].real;
			F[i] = FDual[i].real;
			for (int j = 0; j < 3; j++) {
				J[i * 3 + j] = FDual[i].dual[j];
				xDual[i].dual[j] = 0;
			}
		}

		if (k == 0) { printf("\nx:  %.6f,  %.6f,  %.6f  | err:   %.3e", x[0], x[1], x[2], GetNorm<double, 3>(F)); }
		//ANAL_Solve<double, 3>(J, F, dx);
		GMRES_Solve<double, 3, 3>(J, F, dx, 1e-10, 1);
		for (int j = 0; j < 3; j++) {
			//x[j] += dx[j];
			x[j] -= dx[j];
			xDual[j].real = x[j];
			xDual[j].dual[j] = 1;
		}
		NonlinearFunction<double>(F, x);
		printf("\nx:  %.6f,  %.6f,  %.6f  | err:   %.3e", x[0], x[1], x[2], GetNorm<double, 3>(F));
	}
}

template<typename T, int n>
void JV_approx(T* JV, T* V, const int offset) {

	double F0[3];
	double FPer[3];
	double xPer[3];
	double b = 1e-6;
	double v[3];
	double per;
	for (int j = 0; j < 3; j++) {
		v[j] = V[offset * 3 + j];
		//printf("\nJV_ v[%d]: %.4f", j, v[j]);
	}

	double sum = 0;

	for (int j = 0; j < 3; j++) {
		sum += b * (1 + abs(x[j]));
		//printf(" xi: %.4e", x[j]);
		//printf("\n sum: %.4e", sum);
	}

	double v_norm = GetNorm<double, 3>(v);
	per = (v_norm > 1e-12) ? sum / (n * v_norm) : sum / n;

	for (int j = 0; j < 3; j++) {
		xPer[j] = x[j] + v[j] * per;
	}

	NonlinearFunction<double>(F0, x);
	NonlinearFunction<double>(FPer, xPer);

	/*
	for (int j = 0; j < 3; j++) {
		printf("\n F(x   )[%d]: %.4f", j, F0[j]);
		printf("\n F(x+ep)[%d]: %.4f", j, FPer[j]);
		printf("\n Diff   [%d]: %.4e", j, FPer[j]-F0[j]);
	}
	*/

	for (int j = 0; j < 3; j++) {
		JV[j] = (FPer[j] - F0[j]) / per;
	}
}

#include "..\Symplectic\JFGMRES.cuh"
void NewtonIterationJacFreeGMRES() {

	double F[3];
	//double x[3];
	for (int j = 0; j < 3; j++) {
		x[j] = x0[j];
		dx[j] = 0;
	}

	NonlinearFunction<double>(F, x);
	printf("\nx:  %.6f,  %.6f,  %.6f  | err:   %.3e", x[0], x[1], x[2], GetNorm<double, 3>(F));
	for (int k = 0; k < maxiter; k++) {
		JFGMRES_Solve<double, 3, 3>(F, dx, 1e-10, 1);
		for (int j = 0; j < 3; j++) {
			x[j] -= dx[j];
			dx[j] = 0;
		}
		NonlinearFunction<double>(F, x);
		printf("\nx:  %.6f,  %.6f,  %.6f  | err:   %.3e", x[0], x[1], x[2], GetNorm<double, 3>(F));
	}
}

int main() {
	std::cout << "Newton Iteration (Analitic)" << std::endl;
	NewtonIterationAnalytic();
	std::cout << std::endl << std::endl;
	std::cout << "Newton Iteration (Dual Jacobi - GMRES)" << std::endl;
	NewtonIterationDual();
	std::cout << std::endl << std::endl;
	std::cout << "Newton Iteration (JFGMRES)" << std::endl;
	NewtonIterationJacFreeGMRES();

	return 0;
}