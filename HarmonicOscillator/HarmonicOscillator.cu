/*	SOLVER:
	- FIXEDPOINT		: 0,
	- NEWTON			: 1, 
	- JACFREENEWTON		: 2.

	LINSOLVER (for SOLVER 1 (NEWTON)): 
	- DIRECT			: 0,
	- GAUSS ELIM.		: 1, 
	- DFREEGAUSS ELIM.	: 2,
	- JACOBI ITER.		: 3, 
	- GAUSS_SEID. ITER.	: 4,
	- BICG ITER.		: 5,
	- GMRES (Restarted) : 6.

	PRECISION			: double : float
	ORDER				: 1, 2	(order of Jv approximation for JFGMRES)
	EULRPREDICT			: 0 - Nonliner iteration is initialized from them previous timestep)
						: 1 - Nonliner iterarion is initialized from a predicted Euler Step

*/
#define SOLVER 1
#define LINSOLVER 0
#define PRECISION double
#define ORDER 2
#define EULERPREDICT 1
const int NT = 10;		// NumberOfThreads
const int SD = 2;		// SystemDimension
const int NCP = 2;		// NumberOfControlParameters
const int R = 2;		// GMRES


#include<iostream>
#include<vector>
#include "SystemDefinition.cuh"
#include "Sympletic/Accessories.cuh"
#include "Sympletic/Solvers.cuh"


void Linspace(std::vector<PRECISION>&, PRECISION, PRECISION, int);

void FillSolverObject(Solver<PRECISION, NT, SD, NCP, R>&,
	const std::vector<PRECISION>&);



int main() {
	std::cout << "Harmonic Oscillator" << std::endl;

	ListCUDADevices();
	int SelectedDevice = SelectDeviceByClosestRevision(6, 1);
	PrintPropertiesOfSpecificDevice(SelectedDevice);


	// Configure Problem and Solver
	std::vector<PRECISION> Mass(NT, 0);
	Linspace(Mass, 1.0, 10.0, NT);
	Solver<PRECISION, NT, SD, NCP, R> SolverObject{ SelectedDevice };
	FillSolverObject(SolverObject, Mass);

	SolverObject.AbsoluteTolerace = 1e-10;
	SolverObject.MaxNonlinerIter = 100;
	SolverObject.MaxLinearIter = 10;
	SolverObject.TimeStep = 1e-2;
	SolverObject.numBlocks = 1;
	SolverObject.threadsPerBlock = NT;

	SolverObject.CopyFromHostToDevice(Variable::All);
	SolverObject.Solve();
	SolverObject.CopyFromDeviceToHost(Variable::All);
	return 0;
}


void Linspace(std::vector<PRECISION>& x, PRECISION Min, PRECISION Max, int N) {
	x[0] = Min;
	if (N > 1) {
		x[N - 1] = Max;
		PRECISION Increment = (Max - Min) / (N - 1);

		for (int i = 1; i < N - 1; i++) {
			x[i] = Min + i * Increment;
		}
	}
}


void FillSolverObject(Solver<PRECISION, NT, SD, NCP, R>& SolverObject,
	const std::vector<PRECISION>& Mass) {

	PRECISION Amplitude = 1.0;
	PRECISION SpringStiffnes = 2;


	for (int ProblemID = 0; ProblemID < NT; ProblemID++) {
		SolverObject.SetHost(Variable::TimeDomain, ProblemID, 0, 0.0);
		SolverObject.SetHost(Variable::TimeDomain, ProblemID, 1, 1000);
		SolverObject.SetHost(Variable::ActualState, ProblemID, 0, Amplitude);
		SolverObject.SetHost(Variable::ActualState, ProblemID, 1, 0.5);
		SolverObject.SetHost(Variable::ControlParameter, ProblemID, 0, Mass[ProblemID]);
		SolverObject.SetHost(Variable::ControlParameter, ProblemID, 1, SpringStiffnes);
	}

}