// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include    <wb.h>

#define BLOCK_SIZE 512

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void total(float * input, float * output, int len) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int inx = bx * blockDim.x + tx;

    __shared__ float DS[BLOCK_SIZE * 2];

    // Loading phase
    if(inx < len) DS[tx] = input[inx];
    else DS[tx] = 0.0;

    // Reduction phase
    for(int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int idx = (tx + 1) * stride * 2 - 1;
        __syncthreads();
        if(idx < 2 * BLOCK_SIZE) DS[idx] += DS[idx - stride];
    }

    // Reverse phase
    for(int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
	int idx = (tx + 1) * stride * 2 - 1;
        __syncthreads();
	if(idx + stride < 2 * BLOCK_SIZE) {
            DS[idx + stride] += DS[idx];
        }
    }

    // Using the first thread of the block to save the result
    __syncthreads();
    if(inx < len && tx == 0) output[bx] = DS[BLOCK_SIZE * 2 - 1];
}

int main(int argc, char ** argv) {
    int ii;
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
    hostOutput = (float*) malloc(numOutputElements * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void**) &deviceInput, numInputElements * sizeof(float));
	cudaMalloc((void**) &deviceOutput, numOutputElements * sizeof(float));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
	dim3 DimGrid(numOutputElements, 1, 1);
	dim3 DimBlock(BLOCK_SIZE * 2, 1, 1);
		
    wbTime_start(Compute, "Performing CUDA computation");
    
	total<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements);
    cudaDeviceSynchronize();
	wbCheck(cudaGetLastError());
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }

    wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}


