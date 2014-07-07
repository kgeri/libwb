// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256
#define CAST_BLOCK 16
#define GRAY_BLOCK 32

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void toUnsigned(float * input, unsigned char * output, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x < width && y < height) {
        int idx = 3 * (width * y + x) + threadIdx.z;
        output[idx] = (unsigned char) (255.0f * input[idx]);
    }
}

__global__ void toGray(unsigned char * input, unsigned char * output, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x < width && y < height) {
        int idx = width * y + x;
        unsigned char r = input[3*idx];
        unsigned char g = input[3*idx + 1];
        unsigned char b = input[3*idx + 2];
        output[idx] = (unsigned char) (0.21f*r + 0.71f*g + 0.07f*b);
    }
}

__global__ void computeHistogram(unsigned char * input, unsigned int * output, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    __shared__ unsigned int histogram[HISTOGRAM_LENGTH];

    int idx = width * y + x;
    int bidx = blockDim.x * threadIdx.y + threadIdx.x;
    if(idx < HISTOGRAM_LENGTH) output[idx] = 0;
    if(bidx < HISTOGRAM_LENGTH) histogram[bidx] = 0;
    __syncthreads();

    if(x < width && y < height) {
	atomicAdd(&(histogram[input[idx]]), 1);
    }
    __syncthreads();

    if(bidx < HISTOGRAM_LENGTH) atomicAdd(&(output[bidx]), histogram[bidx]);
}

__device__ unsigned char clamp(float value) { return (unsigned char)((value > 255.0f) ? 255.0f : (value < 0.0f ? 0.0f : value)); }
__global__ void equalizeHistogram(unsigned char * image, const float * __restrict__ cdf, float cdfMin, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x < width && y < height) {
        int idx = 3 * (width * y + x) + threadIdx.z;
	unsigned char value = image[idx];
        float correctedValue = 255.0f*(cdf[value] - cdfMin)/(1.0f - cdfMin);
        image[idx] = clamp(correctedValue);
    }
}

__global__ void toFloat(unsigned char * input, float * output, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x < width && y < height) {
        int idx = 3 * (width * y + x) + threadIdx.z;
        output[idx] = (float) input[idx] / 255.0f;
    }
}

inline float histogramProb(unsigned int count, int width, int height) {
    return (float) count / (float)(width * height);
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    unsigned char * deviceUCharImageData;
    unsigned char * deviceGrayImageData;
    unsigned int * deviceHistogram;
    unsigned int * hostHistogram;
    float * deviceCDF;
    float * hostCDF;
    float cdfMin;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
    hostHistogram = (unsigned int*) malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
    hostCDF = (float*) malloc(HISTOGRAM_LENGTH * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input image width is ", imageWidth);
    wbLog(TRACE, "The input image height is ", imageHeight);

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceUCharImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
    cudaMalloc((void **) &deviceGrayImageData, imageWidth * imageHeight * sizeof(unsigned char));
    cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
    cudaMalloc((void **) &deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    {   // Casting to unsigned
        wbTime_start(Compute, "Doing the computation on the GPU - cast to unsigned");
        dim3 DimGrid((imageWidth-1)/CAST_BLOCK+1, (imageHeight-1)/CAST_BLOCK+1, 1);
        dim3 DimBlock(CAST_BLOCK, CAST_BLOCK, imageChannels);
        toUnsigned<<<DimGrid, DimBlock>>>(deviceInputImageData, deviceUCharImageData, imageWidth, imageHeight);
        cudaDeviceSynchronize();
        wbCheck(cudaGetLastError());
        wbTime_stop(Compute, "Doing the computation on the GPU - cast to unsigned");
    }

    {   // Converting to grayscale
        wbTime_start(Compute, "Doing the computation on the GPU - convert to grayscale");
        dim3 DimGrid((imageWidth-1)/GRAY_BLOCK+1, (imageHeight-1)/GRAY_BLOCK+1, 1);
        dim3 DimBlock(GRAY_BLOCK, GRAY_BLOCK, 1);
        toGray<<<DimGrid, DimBlock>>>(deviceUCharImageData, deviceGrayImageData, imageWidth, imageHeight);
        cudaDeviceSynchronize();
        wbCheck(cudaGetLastError());
        wbTime_stop(Compute, "Doing the computation on the GPU - convert to grayscale");
    }

    {   // Computing histogram
        wbTime_start(Compute, "Doing the computation on the GPU - compute histogram");
        dim3 DimGrid((imageWidth-1)/GRAY_BLOCK+1, (imageHeight-1)/GRAY_BLOCK+1, 1);
        dim3 DimBlock(GRAY_BLOCK, GRAY_BLOCK, 1);
        computeHistogram<<<DimGrid, DimBlock>>>(deviceGrayImageData, deviceHistogram, imageWidth, imageHeight);
        cudaDeviceSynchronize();
        wbCheck(cudaGetLastError());
        wbTime_stop(Compute, "Doing the computation on the GPU - compute histogram");
    }

    {
        // We could write the following with fancy kernels, but there's no way it would be faster for just 256 elements
	cudaMemcpy(hostHistogram, deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        // Computing Cumulative Distribution Function
        hostCDF[0] = histogramProb(hostHistogram[0], imageWidth, imageHeight);
        for(int i = 1; i < HISTOGRAM_LENGTH; i++) hostCDF[i] = hostCDF[i-1] + histogramProb(hostHistogram[i], imageWidth, imageHeight);
        // Computing minimum of CDF
        cdfMin = hostCDF[0];
        for(int i = 1; i < HISTOGRAM_LENGTH; i++) cdfMin = cdfMin < hostCDF[i] ? cdfMin : hostCDF[i];
        wbLog(TRACE, "min(CDF) is ", cdfMin);
    }

    {   // Applying equalization
        cudaMemcpy(deviceCDF, hostCDF, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
        wbTime_start(Compute, "Doing the computation on the GPU - histogram equalization");
        dim3 DimGrid((imageWidth-1)/CAST_BLOCK+1, (imageHeight-1)/CAST_BLOCK+1, 1);
        dim3 DimBlock(CAST_BLOCK, CAST_BLOCK, imageChannels);
        equalizeHistogram<<<DimGrid, DimBlock>>>(deviceUCharImageData, deviceCDF, cdfMin, imageWidth, imageHeight);
        cudaDeviceSynchronize();
        wbCheck(cudaGetLastError());
        wbTime_stop(Compute, "Doing the computation on the GPU - histogram equalization");
    }

    {   // Casting back to float
        wbTime_start(Compute, "Doing the computation on the GPU - cast to float");
        dim3 DimGrid((imageWidth-1)/CAST_BLOCK+1, (imageHeight-1)/CAST_BLOCK+1, 1);
        dim3 DimBlock(CAST_BLOCK, CAST_BLOCK, imageChannels);
        toFloat<<<DimGrid, DimBlock>>>(deviceUCharImageData, deviceOutputImageData, imageWidth, imageHeight);
        cudaDeviceSynchronize();
        wbCheck(cudaGetLastError());
        wbTime_stop(Compute, "Doing the computation on the GPU - cast to float");
    }

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceUCharImageData);
    cudaFree(deviceGrayImageData);
    cudaFree(deviceHistogram);
    cudaFree(deviceCDF);
    cudaFree(deviceOutputImageData);

    free(hostHistogram);
    free(hostCDF);

    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
