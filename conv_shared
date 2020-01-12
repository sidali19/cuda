#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda.h>

#include "cuda_runtime.h"

#define FILTRE_SIZE 3 
#define BLOCK_HEIGHT 16
#define BLOCK_WIDTH 8

#define SHARE_SIZE_HEIGHT (BLOCK_HEIGHT + FILTRE_SIZE -1)
#define SHARE_SIZE_WIDTH (BLOCK_WIDTH + FILTRE_SIZE -1)

__global__ 
void PictureKernel_Shared (unsigned char* dPin, unsigned char* dPout, float *mask, int height, int width)
{	
    __shared__ float N_ds[SHARE_SIZE_HEIGHT][SHARE_SIZE_WIDTH];	//block of share memory

     	// allocation in shared memory of image blocks
	int maskRadius = FILTRE_SIZE/2;
 
        int dest = threadIdx.y * SHARE_SIZE_HEIGHT + threadIdx.x;
        int destY = (threadIdx.y * BLOCK_HEIGHT + threadIdx.x)/SHARE_SIZE_HEIGHT;     //col of shared memory
        int destX = (threadIdx.y * BLOCK_WIDTH + threadIdx.x)%SHARE_SIZE_WIDTH;		//row of shared memory
        int srcY = blockIdx.y *BLOCK_HEIGHT + destY - maskRadius;  //row index to fetch data from input image
        int srcX = blockIdx.x *BLOCK_WIDTH + destX - maskRadius;	//col index to fetch data from input image
        if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
            N_ds[destY][destX] = dPin[(srcY *width +srcX) ];
        else
            N_ds[destY][destX] = 0;

       }

        __syncthreads();
	// Compute row and column number of dPin and dPout element
 		//compute kernel convolution
 	float accum = 0;
 	int y, x;
 	for (y= 0; y < FILTRE_SIZE; y++)
 		for(x = 0; x<FILTRE_SIZE; x++)
 			accum += N_ds[threadIdx.y + y][threadIdx.x + x] *mask[y * maskCols + x];

 	y = blockIdx.y * SHARE_SIZE_HEIGHT + threadIdx.y;
 	x = blockIdx.x * SHARE_SIZE_WIDTH + threadIdx.x;
 	if(y < height && x < width)
 		outputImageData[(y * width + x) ] = accum;
 	__syncthreads();

}

int main(void)
{
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
	const unsigned int block_col = 16;
	const unsigned int block_row = 8;
	dim3 grid(height/block_col, width/ block_row, 1);
	dim3 threadBlock(block_col, block_row, 1);


		PictureKernel <<< grid, threadBlock >>>(gpu_data_in, gpu_data_out, gpu_mask, height, width);
	
	
	cudaMemcpy (data_out, gpu_data_out, width * height * desired_channels, cudaMemcpyDeviceToHost);


	stbi_write_jpg("sortie.jpg", height, width, 1, data_out, height);

	
	free(data_in);
	free(data_out);
	cudaFree(gpu_data_in);
	cudaFree(gpu_data_out);
	cudaFree(gpu_mask);
	

}
