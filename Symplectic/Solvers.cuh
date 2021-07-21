#ifndef SOLVERS_CUH
#define SOLVERS_SUH
#include "Accessories.cuh"

template<typename T>
T* AllocateHostMemory(int Size) {
	T* HostMemory = new (std::nothrow) T[Size];
	if (HostMemory == NULL) {
		std::cerr << "\nFailed to allocate Memory on the HOST" << std::endl;
		exit(1);
	}
	return HostMemory;
}

template<typename T>
T* AllocateDeviceMemory(int Size) {
	T* DeviceMemory;
	gpuErrCHK(cudaMalloc((void**)&DeviceMemory, Size * sizeof(T)));
	return DeviceMemory;
}


enum class Variable {
	All,
	TimeDomain,
	ActualState,
	ActualTime,
	ControlParameter
};


#include "MidPointRuleStepper.cuh"

template<typename T, int NT, int SD, int NCP, int R>
class Solver {
private:
	int DeviceID;
	int NumberOfActiveThreads;

	int SizeOfTimeDomain;			// NT * 2
	int SizeOfActualState;			// NT * SD
	int SizeOfActualTime;			// NT
	int SizeOfControlParameters;	// NT * NCP


	// Host Paramters
	T* h_TimeDomain;
	T* h_ActualState;
	T* h_ActualTime;
	T* h_ControlParameter;

	// Device Parameters
	T* d_TimeDomain;
	T* d_ActualState;
	T* d_ActualTime;
	T* d_ControlParameter;

public:
	Solver(int);
	~Solver();

	int Solve();
	// Default Operation Parameters

	int threadsPerBlock;
	int numBlocks;
	T AbsoluteTolerace = 1e-6;
	T TimeStep = 1e-1;
	int MaxNonlinerIter = 100;
	int MaxLinearIter = 100;


	// SETTERS
	void SetHost(Variable, int, int, T);


	void CopyFromHostToDevice(Variable);
	void CopyFromDeviceToHost(Variable);

};


// CONSTRUCTOR
template<typename T, int NT, int SD, int NCP, int R>
Solver<T, NT, SD, NCP, R>::Solver(int Device) {
	std::cout << "\nInitializing SolverObject ...";
	gpuErrCHK(cudaSetDevice(Device));

	SizeOfActualState		= NT * SD;
	SizeOfTimeDomain		= NT * 2;
	SizeOfActualTime		= NT;
	SizeOfControlParameters = NT * NCP;

	h_TimeDomain	= AllocateHostMemory<T>(SizeOfTimeDomain);
	h_ActualState	= AllocateHostMemory<T>(SizeOfActualState);
	h_ActualTime	= AllocateHostMemory<T>(SizeOfActualTime);
	h_ControlParameter = AllocateHostMemory<T>(SizeOfControlParameters);

	d_TimeDomain	= AllocateDeviceMemory<T>(SizeOfTimeDomain);
	d_ActualState	= AllocateDeviceMemory<T>(SizeOfActualState);
	d_ActualTime	= AllocateDeviceMemory<T>(SizeOfActualTime);
	d_ControlParameter = AllocateDeviceMemory<T>(SizeOfControlParameters);

	std::cout << " Done" << std::endl;
}

// DESTRUCTOR
template<typename T, int NT, int SD, int NCP, int R>
Solver<T, NT, SD, NCP, R>::~Solver() {
	std::cout << "\nDeleting SolverObject ...";
	delete h_TimeDomain;
	delete h_ActualState;
	delete h_ActualTime;
	delete h_ControlParameter;

	gpuErrCHK(cudaFree(d_TimeDomain));
	gpuErrCHK(cudaFree(d_ActualState));
	gpuErrCHK(cudaFree(d_ActualTime));
	gpuErrCHK(cudaFree(d_ControlParameter));

	std::cout << " Done" << std::endl;
}


// MEMBER FUNCTIONS
template<typename T, int NT, int SD, int NCP, int R>
int Solver<T, NT, SD, NCP, R>::Solve() {
	printf("\n%s.Solve() is running on GPU...", typeid(this).name());
	NumberOfActiveThreads = numBlocks * threadsPerBlock;
	SolveSympleticGPU<T, NT, SD, NCP, R><<<numBlocks, threadsPerBlock >>> (
		NumberOfActiveThreads,
		d_TimeDomain,
		d_ActualState,
		d_ActualTime,
		d_ControlParameter,
		AbsoluteTolerace,
		TimeStep,
		MaxNonlinerIter,
		MaxLinearIter);

	return 0;
}


// ACCESSORIES

template<typename T, int NT, int SD, int NCP, int R>
void Solver<T, NT, SD, NCP, R>::SetHost(Variable Variable, int ProblemNumber, int SerialNumber, T value) {

	int idx = ProblemNumber + SerialNumber * NT;
	switch (Variable)
	{
	case Variable::All:
		std::cout << "Err: SetHost(): All is not a legal option to choose" << std::endl;
		break;
	case Variable::TimeDomain:
		h_TimeDomain[idx] = value;
		break;
	case Variable::ActualState:
		h_ActualState[idx] = value;
		break;
	case Variable::ActualTime:
		h_ActualTime[idx] = value;
		break;
	case Variable::ControlParameter:
		h_ControlParameter[idx] = value;
		break;
	default:
		break;
	}
}


template<typename T, int NT, int SD, int NCP, int Restart>
void Solver<T, NT, SD, NCP,  Restart>::CopyFromHostToDevice(Variable Variable) {
	//printf("%s", typeid(T).name());

	switch (Variable)
	{
	case Variable::All:
		gpuErrCHK(cudaMemcpy(d_TimeDomain, h_TimeDomain, SizeOfTimeDomain * sizeof(T), cudaMemcpyHostToDevice));
		gpuErrCHK(cudaMemcpy(d_ActualState, h_ActualState, SizeOfActualState * sizeof(T), cudaMemcpyHostToDevice));
		gpuErrCHK(cudaMemcpy(d_ActualTime, h_ActualTime, SizeOfActualTime * sizeof(T), cudaMemcpyHostToDevice));
		gpuErrCHK(cudaMemcpy(d_ControlParameter, h_ControlParameter, SizeOfControlParameters * sizeof(T), cudaMemcpyHostToDevice));
		break;
	case Variable::TimeDomain:
		gpuErrCHK(cudaMemcpy(d_TimeDomain, h_TimeDomain, SizeOfTimeDomain * sizeof(T), cudaMemcpyHostToDevice));
		break;
	case Variable::ActualState:
		gpuErrCHK(cudaMemcpy(d_ActualState, h_ActualState, SizeOfActualState * sizeof(T), cudaMemcpyHostToDevice));
		break;
	case Variable::ActualTime:
		gpuErrCHK(cudaMemcpy(d_ActualTime, h_ActualTime, SizeOfActualTime * sizeof(T), cudaMemcpyHostToDevice));
		break;
	case Variable::ControlParameter:
		gpuErrCHK(cudaMemcpy(d_ControlParameter, h_ControlParameter, SizeOfControlParameters * sizeof(T), cudaMemcpyHostToDevice));
		break;
	default:
		break;
	}
}

template<typename T, int NT, int SD, int NCP, int Restart>
void Solver<T, NT, SD, NCP, Restart>::CopyFromDeviceToHost(Variable variable) {
	switch (variable)
	{
	case Variable::All:
		gpuErrCHK(cudaMemcpy(h_TimeDomain, d_TimeDomain, SizeOfTimeDomain * sizeof(T), cudaMemcpyDeviceToHost));
		gpuErrCHK(cudaMemcpy(h_ActualState, d_ActualState, SizeOfActualState * sizeof(T), cudaMemcpyDeviceToHost));
		gpuErrCHK(cudaMemcpy(h_ActualTime, d_ActualTime, SizeOfActualTime * sizeof(T), cudaMemcpyDeviceToHost));
		gpuErrCHK(cudaMemcpy(h_ControlParameter, d_ControlParameter, SizeOfControlParameters * sizeof(T), cudaMemcpyDeviceToHost));
		break;
	case Variable::TimeDomain:
		gpuErrCHK(cudaMemcpy(h_TimeDomain, d_TimeDomain, SizeOfTimeDomain * sizeof(T), cudaMemcpyDeviceToHost));
		break;
	case Variable::ActualState:
		gpuErrCHK(cudaMemcpy(h_ActualState, d_ActualState, SizeOfActualState * sizeof(T), cudaMemcpyDeviceToHost));
		break;
	case Variable::ActualTime:
		gpuErrCHK(cudaMemcpy(h_ActualTime, d_ActualTime, SizeOfActualTime * sizeof(T), cudaMemcpyDeviceToHost));
		break;
	case Variable::ControlParameter:
		gpuErrCHK(cudaMemcpy(h_ControlParameter, d_ControlParameter, SizeOfControlParameters * sizeof(T), cudaMemcpyDeviceToHost));
		break;
	default:
		break;
	}
}

#endif
