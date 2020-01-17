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
#include <iostream>
#include "cuda_runtime.h"

#define FILTRE_SIZE 3 
#define BLOCK_HEIGHT 32
#define BLOCK_WIDTH 32

#define SHARE_SIZE_HEIGHT (BLOCK_HEIGHT + FILTRE_SIZE -1)
#define SHARE_SIZE_WIDTH (BLOCK_WIDTH + FILTRE_SIZE -1)

#define TILE_WIDTH 32
#define MASKCOLS 3
#define MASKROWS 3


__global__ void ShareKernelProcessing(unsigned char* InputImageData, const float *kernel,
		unsigned char* outputImageData, int channels, int width, int height){

	__shared__ float N_ds[SHARE_SIZE_HEIGHT][SHARE_SIZE_WIDTH];  //block of image in shared memory



	// allocation in shared memory of image blocks
	int maskr = MASKROWS/2;
 
 		int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
 		int destY = dest/SHARE_SIZE_HEIGHT ;     //row of shared memory
 		int destX = dest%SHARE_SIZE_WIDTH;		//col of shared memory
 		int srcY = blockIdx.y *TILE_WIDTH + destY - maskr; // index to fetch data from input image
 		int srcX = blockIdx.x *TILE_WIDTH + destX - maskr; // index to fetch data from input image
 		int src = (srcY *width +srcX) * channels + k;   // index of input image
 		if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width){
 			N_ds[destY][destX] = InputImageData[src];  // copy element of image in shared memory
		}
 		else
             N_ds[destY][destX] = 0;
             
 		dest = threadIdx.y * TILE_WIDTH+ threadIdx.x + TILE_WIDTH * TILE_WIDTH;
 		destY = dest/SHARE_SIZE_HEIGHT;
		destX = dest%SHARE_SIZE_WIDTH;
		srcY = blockIdx.y *TILE_WIDTH + destY - maskr;
		srcX = blockIdx.x *TILE_WIDTH + destX - maskr;
		src = (srcY *width +srcX) * channels + k;
		if(destY < SHARE_SIZE_HEIGHT){
			if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
				N_ds[destY][destX] = InputImageData[src];
			else{
				N_ds[destY][destX] = 0;
		}

 		__syncthreads();


 		//compute kernel convolution
 		float accum = 0;
 		int y, x;
 		for (y= 0; y < MASKCOLS; y++)
 			for(x = 0; x<MASKROWS; x++)
			{
 		accum += N_ds[threadIdx.y + y][threadIdx.x + x] *kernel[y * MASKCOLS + x];
			
 		y = blockIdx.y * TILE_WIDTH + threadIdx.y;
 		x = blockIdx.x * TILE_WIDTH + threadIdx.x;
 		if(y < height && x < width)
 			outputImageData[(y * width + x) ] = accum;
 		__syncthreads();
			}

 	}



void cuda_error(cudaError_t err,const char *file,int line) {
	//cude check errors 
	if (err != cudaSuccess){
		printf("%s in %s at line %d\n" , cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

#define cuda_error_check(err) (cuda_error( err, __FILE__, __LINE__ ))

int main(int argc, char** argv)
{
        //declarations 
		int width = 0, height = 0, nchannels = 0;
		int num_iteration = 1000;
		float executionTime ;
		int const desired_channels = 1; // request to convert image to gray
		char const * const filename1 = argv[1]; 
		char const * const filename2 = "sortie.jpg";
		// Load the image 
		unsigned char* data_in = stbi_load(filename1, &width, &height, &nchannels, desired_channels);
		// check for errors 
		if (!data_in || !width || !height || !nchannels){
		printf("Error loading image %s", filename1);
		return -1;
		}

		// the filter mask 
		float mask[FILTRE_SIZE*FILTRE_SIZE] = { -1, -1, -1, -1, 8, -1, -1, -1, -1};


	

		// Memory allocation GPU
		unsigned char *gpu_data_in, *gpu_data_out;
		unsigned char*data_out = (unsigned char*)malloc(width * height * desired_channels);
		float * gpu_mask;
		
		cuda_error_check(cudaMalloc (( void **)&gpu_data_in, width * height * desired_channels*sizeof(unsigned char)));
		cuda_error_check(cudaMalloc (( void **)&gpu_data_out, width * height * desired_channels*sizeof(unsigned char)));
		cuda_error_check(cudaMalloc (( void **)&gpu_mask, FILTRE_SIZE*FILTRE_SIZE*sizeof(float)));

		// data copy  host to device 
		cuda_error_check(cudaMemcpy (gpu_data_in, data_in, width * height * desired_channels*sizeof(unsigned char) , cudaMemcpyHostToDevice));
		cuda_error_check(cudaMemcpy (gpu_mask, mask , FILTRE_SIZE*FILTRE_SIZE*sizeof(float), cudaMemcpyHostToDevice));
		
		// Set up the grid and block dimensions for the executions
		const unsigned int block_col = 32;
		const unsigned int block_row = 32;
		// creat cuda event to calculate time execution [start,stop]
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		// grid dimension 
		dim3 grid(height/block_col, width/ block_row, 1);
		// block dimension 
		dim3 threadBlock(block_col, block_row, 1);
		// start recording    
		cudaEventRecord(start, 0);
		for(int i=0; i < num_iteration; i++){
			// karnel call 
				ShareKernelProcessing<<< grid, threadBlock >>>(gpu_data_in,gpu_mask, gpu_data_out,desired_channels,height, width);
		}	
		cudaEventRecord(stop, 0);
		// stop recording 
		cudaEventSynchronize(stop);
		
		cudaEventElapsedTime(&executionTime, start, stop);
		// data copy  device to host
		cuda_error_check(cudaMemcpy (data_out, gpu_data_out, width * height * desired_channels, cudaMemcpyDeviceToHost));
		
		printf("Execution Time 1000 images : %f ms \n", executionTime);
		printf("Execution Time image : %f ms \n", executionTime/1000);
		//write the image 
		if(!stbi_write_jpg(filename2, height, width, 1, data_out, height))
		{
			printf("Error saving image %s \n", filename2);
			return (-1);
		}

		free(data_in);
		free(data_out);
		cudaFree(gpu_data_in);
		cudaFree(gpu_data_out);
		cudaFree(gpu_mask);
	

}
