// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void scan(int bx, float * input, float * output, int len) {
    int tx = threadIdx.x;
    int inx = bx * blockDim.x + tx;

    __shared__ float DS[BLOCK_SIZE * 2];

    // Loading phase
    // Adding the previous block's last result to the first item, if there was any
    if(tx == 0 && inx > 0) DS[tx] = input[inx] + output[inx-1];
    else if(inx < len) DS[tx] = input[inx];
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
    if(inx < len) output[inx] = DS[tx];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

	dim3 DimGrid(1, 1, 1);
	dim3 DimBlock(BLOCK_SIZE * 2, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");

    // Note: wasn't quite sure how to implement this loop inside the kernel - since there's a data dependency between the blocks, I think this is the most efficient way
    for(int bx = 0; bx < (numElements - 1) / BLOCK_SIZE * 2 + 1; bx++) {
        scan<<<DimGrid, DimBlock>>>(bx, deviceInput, deviceOutput, numElements);
        cudaDeviceSynchronize();
        wbCheck(cudaGetLastError());
    }

    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

