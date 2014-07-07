#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define TILE_SIZE 12

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
	
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int row = blockIdx.y * blockDim.y + ty;
	int col = blockIdx.x * blockDim.x + tx;
	
	__shared__ float ds_A[TILE_SIZE][TILE_SIZE];
	__shared__ float ds_B[TILE_SIZE][TILE_SIZE];
	
	float cValue = 0.0;
	for(int t=0; t < (numAColumns - 1)/TILE_SIZE+1; t++) {
		
		// Copying tile from A
		if(row < numARows && t*TILE_SIZE+tx < numAColumns) {
			ds_A[ty][tx] = A[row * numAColumns + t*TILE_SIZE+tx];
		} else {
			ds_A[ty][tx] = 0.0;
		}
		
		// Copying tile from B
		if(t*TILE_SIZE+ty < numBRows && col < numBColumns) {
			ds_B[ty][tx] = B[(t*TILE_SIZE+ty) * numBColumns + col];
		} else {
			ds_B[ty][tx] = 0.0;
		}
		
		// Waiting for other threads to finish loading
		__syncthreads();
		
		// Calculating a single cell's result
		for(int i=0; i<TILE_SIZE; i++) {
			cValue += ds_A[ty][i] * ds_B[i][tx];
		}
		
		// Waiting for other threads to finish calculation
		__syncthreads();
	}
	
	if(row < numCRows && col < numCColumns) {
		C[row * numCColumns + col] = cValue;
	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    numCRows = numARows;
    numCColumns = numBColumns;
	hostC = (float *) malloc(numCRows * numCColumns * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**) &deviceA, numARows * numAColumns * sizeof(float)));
	wbCheck(cudaMalloc((void**) &deviceB, numBRows * numBColumns * sizeof(float)));
	wbCheck(cudaMalloc((void**) &deviceC, numCRows * numCColumns * sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    dim3 DimGrid((numCColumns-1)/TILE_SIZE+1, (numCRows-1)/TILE_SIZE+1, 1);
	dim3 DimBlock(TILE_SIZE, TILE_SIZE, 1);
    
    wbTime_start(Compute, "Performing CUDA computation");
    matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    cudaThreadSynchronize();
	wbCheck(cudaGetLastError());
	
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    wbCheck(cudaFree(deviceA));
	wbCheck(cudaFree(deviceB));
	wbCheck(cudaFree(deviceC));
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}


