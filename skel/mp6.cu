#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius 2

#define TILE_WIDTH 12
#define BLOCK_WIDTH 16


__device__ float clamp(float value) {
	return value > 1.0 ? 1.0 : value;
}

__global__ void imageConvolution(float * input, float * output, int imageWidth, int imageHeight, int channels, 
								 const float * __restrict__ mask) {
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col_o = blockIdx.x * TILE_WIDTH + tx;
	int row_o = blockIdx.y * TILE_WIDTH + ty;
	int col_i = col_o - Mask_radius;
	int row_i = row_o - Mask_radius;
	
	__shared__ float ds_I[BLOCK_WIDTH][BLOCK_WIDTH];
	
	// Processing every channel
	for(int c = 0; c < channels; c++) {

		// Loading the input image to shared memory - all threads participating (except at edges)
		if(col_i >= 0 && col_i < imageWidth && row_i >= 0 && row_i < imageHeight) {
			ds_I[ty][tx] = input[(row_i * imageWidth + col_i) * channels + c];
		} else {
			ds_I[ty][tx] = 0.0;
		}

		// Waiting for other threads to finish loading
		__syncthreads();

		// Calculating the convolution for a single element - only TILE_WIDTH * TILE_WIDTH threads are participating
		float value = 0.0;
		if(tx < TILE_WIDTH && ty < TILE_WIDTH) {
			for(int i = 0; i < Mask_width; i++) {
				for(int j = 0; j < Mask_width; j++) {
					value += mask[i * Mask_width + j] * ds_I[ty + i][tx + j];
				}
			}
		}

		// Waiting for other threads to finish calculation
		__syncthreads();

		// Saving result - only the calculator threads are participating
		if(tx < TILE_WIDTH && ty < TILE_WIDTH && col_o < imageWidth && row_o < imageHeight) {
			output[(row_o * imageWidth + col_o) * channels + c] = clamp(value);
		}
	}
}


int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");
	
    dim3 DimGrid((imageWidth-1)/TILE_WIDTH+1, (imageHeight-1)/TILE_WIDTH+1, 1);
    dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    wbTime_start(Compute, "Doing the computation on the GPU");
    imageConvolution<<<DimGrid, DimBlock>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight, imageChannels, deviceMaskData);
    cudaThreadSynchronize();
	wbCheck(cudaGetLastError());
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

