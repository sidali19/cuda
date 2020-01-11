#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda.h>

#include "cuda_runtime.h"

//#define PIX(a,b) get_pixel(in,w,a,b)

//inline unsigned char get_pixel(unsigned char * in, int w , int x, int y)
	//{
		//return in[x + y*w];
	//}
float b[3][3]={-1,-1,-1,-1,8,-1,-1,-1,-1};


_global_ void Conv ( unsigned char * in  ,int w,int h ,unsigned char *out  ) 
{
	float kernel[3][3]={-1,-1,-1,-1,8,-1,-1,-1,-1};

    //float kernel[3][3]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    float tmp;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int lin = blockIdx.x * blockDim.x + threadIdx.x;
	
	int i = lin;
    int j = col;
    int pixel = i*w+j;

    tmp =          in[pixel-w-1]     * kernel[0][0]
				 + in[pixel-1]     * kernel[1][0]
				 + in[pixel+w-1]     * kernel[2][0]				 
				 + in[pixel-w]     * kernel[0][1]
				 + in[pixel]     * kernel[1][1]
				 + in[pixel+w]     * kernel[2][1]
				 + in[pixel-w+1]     * kernel[0][2]
				 + in[pixel+1]     * kernel[1][2]
                 + in[pixel+w+1]     * kernel[2][2];
                 
/*
	tmp = in[(i*w-1) +(j-1)]  * kernel[0][0]
				 + in[i*w+ (j-1) ]     * kernel[1][0]
				 + in[(i*w+1) + (j-1)] * kernel[2][0]				 
				 + in[(i*w-1) + j]     * kernel[0][1]
				 + in[i*w  + j]        * kernel[1][1]
				 + in[(i*w+1) + j]     * kernel[2][1]
				 + in[(i*w-1)+ (j+1)]  * kernel[0][2]
				 + in[i *w+ (j+1)]     * kernel[1][2]
				 + in[(i*w+1) + (j+1)] * kernel[2][2];
*/
	//out[i+ h*j] =(tmp <= 0.0f ? 0 : (tmp >= 255.0 ? 255 :(unsigned char)tmp)) ;

	 out[i*w+ j]= (unsigned char)tmp;
}

int main(void)
{
	char const * const filename = "image02.jpg";
	char const * const filename1 = "cpu.png";
	char const * const filename2 = "gpu.png";
	int width =0, height=0, nchannels=0;//128*128 pixels  3 channels
	int const desired_channels =1;//convert image to gray

	//load image
	unsigned char * image_data = stbi_load(filename,&width,&height,&nchannels,desired_channels);

	int N = height*width;

	unsigned char * image_data1= stbi_load(filename,&width,&height,&nchannels,desired_channels);
	
	//check for errors
	if(!image_data || !width || !height || !nchannels)
	{
		printf("Error loading image %s \n", filename);
		return -1;
	}
	
	//check for errors
	if(!image_data1 || !width || !height || !nchannels)
	{
		printf("Error loading image %s \n", filename);
		return -1;
	}

	float a[height][width];
		

	//use the image data

	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			a[i][j]=(float)image_data[i*width+j];
		}	
	}

/*	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			printf(" a = %f",a[i][j]);
		}	
	}

	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			printf(" b = %f",b[i][j]);
		}	
	}
*/
    
	//calcule
	float c[height][width];
	
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			if((i*j)>0)
			{
				c[i][j]= a[i-1][j-1]*b[0][0]+a[i-1][j]*b[0][1]+a[i-1][j+1]*b[0][2]
						+a[i][j-1]*b[1][0]+a[i][j]*b[1][1]+a[i][j]*b[1][2]
						+a[i+1][j-1]*b[2][0]+a[i+1][j]*b[2][1]+a[i+1][j+1]*b[2][2];
			}
			else{
				c[i][j]=0;
			}
		}	
	}

	//save image
	int stride = width * desired_channels;
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			image_data[i*width+j]=(unsigned char)c[i][j];
		}	
	}
	if(!stbi_write_png(filename1,width,height,desired_channels,image_data,stride))
	{
		printf("Error saving image %s \n", filename1);
		return (-1);
	}


	//partie gpu image_data1
	unsigned char *dev_data;
	float *dev_a;
	unsigned char *dev_c;
    
    int taille = height*width * sizeof(unsigned char);
	cudaMalloc((void**)&dev_data,taille);
	
	cudaMalloc((void**)&dev_c,taille);

	cudaMemcpy(dev_data,image_data1,taille,cudaMemcpyHostToDevice);
    
    free(image_data1);

	//nbThreads = 128 th max
	
	dim3 blockSize ( 16 ,8 ,1) ;
	dim3 gridSize (height/16,width/8 ,1) ;	
	
	//calcule GPU
	Conv <<< gridSize , blockSize >>>( dev_data , width,height , dev_c  );

	unsigned char* s = (unsigned char *)malloc(taille);

	cudaMemcpy(dev_c,s,taille,cudaMemcpyDeviceToHost);
	
	//sortie gpu
	stride = width * desired_channels;
	
	if(!stbi_write_png(filename2,width,height,desired_channels,s,stride))
	{
		printf("Error saving image %s \n", filename2);
		return (-1);
	}
    
    //release the image memory buffer
    
    free(image_data);
    free(s);

	return 0;
}
