
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda.h>

#include "cuda_runtime.h"

#define FILTRE_SIZE 3 

static void HandleError(cudaError_t err,const char *file,int line) {
	if (err != cudaSuccess){
		printf("%s in %s at line %d\n" , cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

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

inline unsigned int iDivUp(const unsigned int &a, const unsigned &b) { return (a%b != 0) ? (a / b + 1) : (a / b); }

int main(void)
{
	// Lecture et chargement de l'image dans le host en ligne de commande 
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

	size_t img_size = width * height * desired_channels;
	size_t h_size = 9 *sizeof(float);
	unsigned char*data_out = (unsigned char*)malloc(width * height * desired_channels);
	// Affichage des infos de l'image
	//cout << "Load the image successfully!"<< endl;
	//cout << "Width = "<< rows << " & Height = " << cols << endl;
	// Memory allocation GPU
	unsigned char *gpu_data_in, *gpu_data_out;
	float * gpu_mask;
	cudaMalloc(reinterpret_cast<void **>(&gpu_data_in), width * height * desired_channels*sizeof(float));
	cudaMalloc(reinterpret_cast<void **>(&gpu_data_out), width * height * desired_channels*sizeof(float));
	cudaMalloc(reinterpret_cast<void **>(&gpu_mask), FILTRE_SIZE*FILTRE_SIZE*sizeof(float));
	
	//	auto h_start = steady_clock::now();

	// Copie des donnees de host vers le device 
	cudaMemcpy (gpu_data_in, data_in, width * height * desired_channels*sizeof(float) , cudaMemcpyHostToDevice);
	cudaMemcpy (gpu_mask, mask , FILTRE_SIZE*FILTRE_SIZE*sizeof(float), cudaMemcpyHostToDevice);
	
	// appel du kernel
	//unsigned int iter = 10000;

	// Set up the grid and block dimensions for the executions
	const unsigned int block_col = 16;
	const unsigned int block_row = 8;
	dim3 grid(iDivUp(height, block_col), iDivUp(width, block_row), 1);
	dim3 threadBlock(block_col, block_row, 1);

	// **** CONVOLUTION STARTS HERE ! ****
		//float elapsed = 0;
		//cudaEvent_t start, stop;
	
		//HANDLE_ERROR(cudaEventCreate(&start));
		//HANDLE_ERROR(cudaEventCreate(&stop));

		//HANDLE_ERROR(cudaEventRecord(start, 0));

		//checkCudaErrors(cudaDeviceSynchroonise());
	
		//for(int i=0; i < iter; i++){
		PictureKernel <<< grid, threadBlock >>>(gpu_data_in, gpu_data_out, gpu_mask, height, width);
		//} 

		//checkCudaErrors(cudaDeviceSynchroonise());
	
		//HANDLE_ERROR(cudaEventRecord(stop, 0));
		//HANDLE_ERROR(cudaEventSynchronise(stop));

		//HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

		//HANDLE_ERROR(cudaEventDestroy(start));
		//HANDLE_ERROR(cudaEventDestroy(stop));
	
	//	float lasptime = elapsed;

	/*	cout << "Total Elapsed Time for the Kernel(GPU): "<< lasptime << " s " << endl;
		auto h_end = steady_clock::now();
		cout << "Total Elapsed Time(including data transfer): " << (duration<double>(h_end - h_start).count()) << "s\n " << endl;

		int pixel_second = (height * width) / lasptime;
	cout << "Performance  in pixel/s "<< pixel_second << endl;
	*/
	// **** CONVOLUTION ENDS HERE ! ****	
	
	cudaMemcpy (data_out, gpu_data_out, width * height * desired_channels, cudaMemcpyDeviceToHost);

	// Write convoluted image to file (.jpg)
	stbi_write_jpg("sortie.jpg", height, width, 1, data_out, height);
	printf("zebi");
	//Deallocation des memoires
	free(data_in);
	free(data_out);
	cudaFree(gpu_data_in);
	cudaFree(gpu_data_out);
	cudaFree(gpu_mask);
	
	printf("zebi");
	printf("\n");
}

