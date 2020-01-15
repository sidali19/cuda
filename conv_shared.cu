#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"

#define FILTRE_SIZE 3 
#define BLOCK_HEIGHT 32
#define BLOCK_WIDTH 32

#define SHARE_SIZE_HEIGHT (BLOCK_HEIGHT + FILTRE_SIZE -1)
#define SHARE_SIZE_WIDTH (BLOCK_WIDTH + FILTRE_SIZE -1)
#include <iostream>

#define TILE_WIDTH 32
#define maskCols 3
#define maskRows 3
#define w (TILE_WIDTH + maskCols -1)

__global__ void tilingKernelProcessing(unsigned char* InputImageData, const float *__restrict__ kernel,
		unsigned char* outputImageData, int channels, int width, int height){

	__shared__ float N_ds[w][w];  //block of image in shared memory


	// allocation in shared memory of image blocks
	int maskRadius = maskRows/2;
 	for (int k = 0; k <channels; k++) {
 		int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
 		int destY = dest/w;     //row of shared memory
 		int destX = dest%w;		//col of shared memory
 		int srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius; // index to fetch data from input image
 		int srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius; // index to fetch data from input image
 		int src = (srcY *width +srcX) * channels + k;   // index of input image
 		if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
 			N_ds[destY][destX] = InputImageData[src];  // copy element of image in shared memory
 		else
 			N_ds[destY][destX] = 0;



 		dest = threadIdx.y * TILE_WIDTH+ threadIdx.x + TILE_WIDTH * TILE_WIDTH;
 		destY = dest/w;
		destX = dest%w;
		srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius;
		srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius;
		src = (srcY *width +srcX) * channels + k;
		if(destY < w){
			if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
				N_ds[destY][destX] = InputImageData[src];
			else
				N_ds[destY][destX] = 0;
		}

 		__syncthreads();


 		//compute kernel convolution
 		float accum = 0;
 		int y, x;
 		for (y= 0; y < maskCols; y++)
 			for(x = 0; x<maskRows; x++)
 				accum += N_ds[threadIdx.y + y][threadIdx.x + x] *kernel[y * maskCols + x];

 		y = blockIdx.y * TILE_WIDTH + threadIdx.y;
 		x = blockIdx.x * TILE_WIDTH + threadIdx.x;
 		if(y < height && x < width)
 			outputImageData[(y * width + x) * channels + k] = accum;
 		__syncthreads();

 	}

}

int main(void)
{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		int width = 0, height = 0, nchannels = 0;
		int const desired_channels = 1; // request to convert image to gray
		char const * const filename = "im.jpg"; 
	// Load the image 
	unsigned char* data_in = stbi_load(filename, &width, &height, &nchannels, desired_channels);

	// check for errors 
	if (!data_in || !width || !height || !nchannels){
	printf("Error loading image %s", filename);
	return -1;
	}

	// the filter  
	float mask[FILTRE_SIZE*FILTRE_SIZE] = { -1, -1, -1, -1, 8, -1, -1, -1, -1};


	unsigned char*data_out = (unsigned char*)malloc(width * height * desired_channels);

	// Memory allocation GPU
	unsigned char *gpu_data_in, *gpu_data_out;
	float * gpu_mask;
	
	cudaMalloc (( void **)&gpu_data_in, width * height * desired_channels*sizeof(float));
	cudaMalloc (( void **)&gpu_data_out, width * height * desired_channels*sizeof(float));
	cudaMalloc (( void **)&gpu_mask, FILTRE_SIZE*FILTRE_SIZE*sizeof(float));
	
	

	cudaMemcpy (gpu_data_in, data_in, width * height * desired_channels*sizeof(float) , cudaMemcpyHostToDevice);
	cudaMemcpy (gpu_mask, mask , FILTRE_SIZE*FILTRE_SIZE*sizeof(float), cudaMemcpyHostToDevice);
	


	// Set up the grid and block dimensions for the executions
	const unsigned int block_col = 32;
	const unsigned int block_row = 32;
	dim3 grid(height/block_col, width/ block_row, 1);
   	 dim3 threadBlock(block_col, block_row, 1);
  	cudaEventRecord(start, 0);

tilingKernelProcessing <<< grid, threadBlock >>>(gpu_data_in, gpu_mask,gpu_data_out,desired_channels,height, width);
	cudaEventRecord(stop, 0);
	
	cudaEventSynchronize(stop);
	float executionTime ;
	cudaEventElapsedTime(&executionTime, start, stop);
	
	
	
	
	
	
	cudaMemcpy (data_out, gpu_data_out, width * height * desired_channels, cudaMemcpyDeviceToHost);
	
	//printf("SECTION 1 executionTime: %f", executionTime);
	//cudaEventRecord(stop);
	stbi_write_jpg("sortie.jpg", height, width, 1, data_out, height);
	
	//  cudaEventSynchronize(stop);
	
  	//float milliseconds = 0;
	
 	// cudaEventElapsedTime(&milliseconds, start, stop);
	
       // printf("%f",milliseconds);
	free(data_in);
	free(data_out);
	cudaFree(gpu_data_in);
	cudaFree(gpu_data_out);
	cudaFree(gpu_mask);
	

}
