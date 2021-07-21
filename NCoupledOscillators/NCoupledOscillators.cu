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
#define SOLVER 0
#define LINSOLVER 5
#define PRECISION double
#define ORDER 2
#define EULERPREDICT 1

const int N = 3;			// Number Of Masses
const int NT = 1;			// NumberOfThreads
const int SD = 2 * N;		// SystemDimension
const int NCP = 2 * N +1;	// NumberOfControlParameters
const int R = 2;			// GMRES


#include<iostream>
#include<vector>
#include "SystemDefinition.cuh"
#include "..\Symplectic\Accessories.cuh"
#include "..\Symplectic\Solvers.cuh"

void FillSolverObject(Solver<PRECISION, NT, SD, NCP, R>&);

int main() {
	std::cout << "N-Coupled Oscillators" << std::endl;

	ListCUDADevices();
	int SelectedDevice = SelectDeviceByClosestRevision(6, 1);
	PrintPropertiesOfSpecificDevice(SelectedDevice);

	Solver<PRECISION, NT, SD, NCP, R> SolverObject{ SelectedDevice };
	FillSolverObject(SolverObject);

	SolverObject.AbsoluteTolerace = 1e-10;
	SolverObject.MaxNonlinerIter = 100;
	SolverObject.MaxLinearIter = 10;
	SolverObject.TimeStep = 1e-2;
	SolverObject.numBlocks = 1;
	SolverObject.threadsPerBlock = NT;

	SolverObject.CopyFromHostToDevice(Variable::All);
	SolverObject.Solve();
	//SolverObject.CopyFromDeviceToHost(Variable::All);
	return 0;
}



void FillSolverObject(Solver<PRECISION, NT, SD, NCP, R>& SolverObject) {

	PRECISION Amplitude			= 1.0;
	PRECISION Mass				= 1;
	PRECISION SpringStiffnes	= 200;

	for (int ProblemID = 0; ProblemID < NT; ProblemID++) {
		SolverObject.SetHost(Variable::TimeDomain, ProblemID, 0, 0.0);
		SolverObject.SetHost(Variable::TimeDomain, ProblemID, 1, 10.0);

		for (int j = 0; j < N; j++) {
			SolverObject.SetHost(Variable::ActualState, ProblemID, j, 0.0);
			SolverObject.SetHost(Variable::ActualState, ProblemID, N + j, 0.0);
			SolverObject.SetHost(Variable::ControlParameter, ProblemID, j, Mass);
			SolverObject.SetHost(Variable::ControlParameter, ProblemID, N + j, SpringStiffnes);
		}
		SolverObject.SetHost(Variable::ActualState, ProblemID, 0, Amplitude);
		SolverObject.SetHost(Variable::ControlParameter, ProblemID, 2 * N, SpringStiffnes);
	}
}