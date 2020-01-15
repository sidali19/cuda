
#include <stdio.h>
#include <stdlib.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda.h>
#include "cuda_runtime.h"
#define FILTRE_SIZE 3 


__global__ 
void PictureKernel (unsigned char* dPin, unsigned char* dPout, float *mask, int height, int width)
{	

	// Compute row and column number of dPin and dPout element
	const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	int position = row*width+col ;
	int up = position - width;
	int down = position + width;
	// Each thread computes one element of dPout if in range
	if(row==0||col==0||(row==width-1)||(row==height-1))
	{
		dPout[position] = dPin[position];
		return;
	}
	


	float val = (mask[0]*dPin[up-1] + mask[1]*dPin[up] + mask[2]*dPin[up + 1]
			+ mask[3]*dPin[position-1] + mask[4]*dPin[position] + mask[5]*dPin[position+1]
			+ mask[6]*dPin[down -1] + mask[7]*dPin[down] + mask[8]*dPin[down+1]);
	dPout[position] = (val <= 0 ? 0 : (val >= 255 ? 255 : (unsigned char)val));
}

void cuda_error(cudaError_t err,const char *file,int line) {
	if (err != cudaSuccess){
		printf("%s in %s at line %d\n" , cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
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
	
	cuda_error( cudaMalloc (( void **)&gpu_data_in, width * height * desired_channels*sizeof(float)));
	
	cuda_error(cudaMalloc (( void **)&gpu_data_out, width * height * desired_channels*sizeof(float)));
	cuda_error(cudaMalloc (( void **)&gpu_mask, FILTRE_SIZE*FILTRE_SIZE*sizeof(float)));
	
	

	cuda_error(cudaMemcpy (gpu_data_in, data_in, width * height * desired_channels*sizeof(float) , cudaMemcpyHostToDevice));
	cuda_error(cudaMemcpy (gpu_mask, mask , FILTRE_SIZE*FILTRE_SIZE*sizeof(float), cudaMemcpyHostToDevice));
	


	// Set up the grid and block dimensions for the executions
	const unsigned int block_col = 16;
	const unsigned int block_row = 8;
	dim3 grid(height/block_col, width/ block_row, 1);
	dim3 threadBlock(block_col, block_row, 1);


		PictureKernel <<< grid, threadBlock >>>(gpu_data_in, gpu_data_out, gpu_mask, height, width);
	
	
	cuda_error(cudaMemcpy (data_out, gpu_data_out, width * height * desired_channels, cudaMemcpyDeviceToHost));


	stbi_write_jpg("sortie.jpg", height, width, 1, data_out, height);

	
	free(data_in);
	free(data_out);
	cudaFree(gpu_data_in);
	cudaFree(gpu_data_out);
	cudaFree(gpu_mask);
	

}

