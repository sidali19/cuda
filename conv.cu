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


__global__ void PictureKernel (unsigned char* dPin, unsigned char* dPout, float *mask, int height, int width)
{	

		// Compute row and column number of dPin and dPout element
		const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
		const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
		int position = row*width+col ;
	// Each thread computes one element of dPout if in range
	if(row==0||col==0||(row==width-1)||(row==height-1))
	{
		dPout[position] = dPin[position];
		return;
	}	
	float val = (mask[0]*dPin[position - width-1] + mask[1]*dPin[position - width] + mask[2]*dPin[position - width + 1]
			+ mask[3]*dPin[position-1] + mask[4]*dPin[position] + mask[5]*dPin[position+1]
			+ mask[6]*dPin[position + width -1] + mask[7]*dPin[position + width] + mask[8]*dPin[position + width+1]);
	dPout[position] = (val <= 0 ? 0 : (val >= 255 ? 255 : (unsigned char)val));
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
				PictureKernel <<< grid, threadBlock >>>(gpu_data_in, gpu_data_out,gpu_mask,height, width);
		}	
		cudaEventRecord(stop, 0);
		// stop recording 
		cudaEventSynchronize(stop);
		
		cudaEventElapsedTime(&executionTime, start, stop);
		// data copy  device to host
		cuda_error_check(cudaMemcpy (data_out, gpu_data_out, width * height * desired_channels, cudaMemcpyDeviceToHost));
		
		printf("ExecutionTime: %f", executionTime);
		printf("ExecutionTime image/seconde : %f", executionTime);
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
