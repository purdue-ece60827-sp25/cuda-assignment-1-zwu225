
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i<size) y[i] = y[i] + scale * x[i];
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	// 	Init floating point vectors
	size_t vectorBytes = vectorSize * sizeof(float);
	float *a, *b, *c;
	float *a_d, *c_d;
	
	a = (float *) malloc(vectorBytes);
	b = (float *) malloc(vectorBytes);
	c = (float *) malloc(vectorBytes);

	if (a == NULL || b == NULL || c == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);
	std::memcpy(c, b, vectorBytes);	//	C = B
	
	cudaMalloc(&a_d, vectorBytes);
	cudaMemcpy(a_d, a, vectorBytes, cudaMemcpyHostToDevice);
	cudaMalloc(&c_d, vectorBytes);
	cudaMemcpy(c_d, c, vectorBytes, cudaMemcpyHostToDevice);

	float scale = (float)(rand() % 10);

	//	GPU SAXPY
	dim3 DimGrid(vectorSize/256, 1, 1);
	if (vectorSize%256) DimGrid.x++;
	dim3 DimBlock(256, 1, 1);
	saxpy_gpu<<<DimGrid, DimBlock>>>(a_d, c_d, scale, vectorSize);
	cudaMemcpy(c, c_d, vectorBytes, cudaMemcpyDeviceToHost);
	
	//	Check answer with CPU
	int errorCount = verifyVector(a, b, c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	std::cout << "Freeing ...\n";
	free(a);
	free(b);
	free(c);
	cudaFree(a_d);
	cudaFree(c_d);
	std::cout << "... done!\n";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	pSums[i] = 0;
	float x, y;
	
	// Setup RNG
	curandState_t rng;
	curand_init(clock64(), i, 0, &rng);

	if (i < pSumSize){
		for (uint64_t idx = 0; idx < sampleSize; ++idx) {
			x = curand_uniform(&rng);
			y = curand_uniform(&rng);
			pSums[i] += (uint64_t) 1 - (uint64_t)(x * x + y * y);
		}
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//	Insert code here
	//	Init
	uint64_t *hitVector, *hitVector_d;
	hitVector = (uint64_t *) malloc(generateThreadCount * sizeof(uint64_t));
	cudaMalloc(&hitVector_d, generateThreadCount * sizeof(uint64_t));

	//	GPU launch
	dim3 DimGrid(generateThreadCount/256, 1, 1);
	if (generateThreadCount%256) DimGrid.x++;
	dim3 DimBlock(256, 1, 1);
	generatePoints<<<DimGrid, DimBlock>>>(hitVector_d, generateThreadCount, sampleSize);
	cudaMemcpy(hitVector, hitVector_d, generateThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	//	CPU sum
	for (uint64_t i = 0; i < generateThreadCount; ++i){
		approxPi += (double)hitVector[i];
	}

	approxPi = approxPi / (double)sampleSize / (double)generateThreadCount;
	approxPi *= 4.0f; 

	std::cout << "Freeing ...\n";
	free(hitVector);
	cudaFree(hitVector_d);
	std::cout << "... done!\n";

	return approxPi;
}
