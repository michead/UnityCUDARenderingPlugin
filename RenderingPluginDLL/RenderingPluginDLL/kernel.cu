#include <GL/glew.h>
#include <GL/freeglut.h>
#include <gl/GLU.h>
#include <GL/gl.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <vector_types.h>

#include <stdio.h>
#include <math.h>

#include <Windows.h>

#define WINDOWS_MEAN_AND_LEAN
#define NOMINAX
#define EXPORT_API __declspec(dllexport)

static float g_Time;
static float g_Freq = 4.0f;

static float mesh_width;
static float mesh_height;

static GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

typedef void(*FuncPtr)(const char *);
FuncPtr Debug;

extern "C" EXPORT_API void SetDebugFunction(FuncPtr fp)
{
	Debug = fp;
}

extern "C" EXPORT_API void ComputeSineWave(float3* verts, int sideSize, float time)
{
	for (int i = 0; i < sideSize * sideSize; i ++) 
		verts[i].y = sin(verts[i].x * g_Freq + time) * cos(verts[i].z * g_Freq + time) * 0.2f;
}

__global__ void simple_vbo_kernel(float3 *pos, unsigned int sideSize, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// calculate uv coordinates
	// float u = x / (float)width;
	// float v = y / (float)height;
	// u = u*2.0f - 1.0f;
	// v = v*2.0f - 1.0f;

	// calculate simple sine wave pattern
	float freq = 4.0f;
	// float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	// write output vertex
	int index = y * sideSize + x;
	pos[index].y = sinf(pos[index].x * freq + time) * cosf(pos[index].z * freq + time) * 0.2f;
}

extern "C" EXPORT_API void ParallelComputeSineWave(float3* verts, int sideSize, float time)
{
	float3* dev_verts;

	cudaMalloc((void**)&dev_verts, sideSize * sideSize * sizeof(float3));
	cudaMemcpy(dev_verts, verts, sideSize * sideSize * sizeof(float3), cudaMemcpyHostToDevice);

	dim3 block(10, 10, 1);
	dim3 grid(sideSize / block.x, sideSize / block.y, 1);
	simple_vbo_kernel << < grid, block >> >(dev_verts, sideSize, time);

	cudaMemcpy(verts, dev_verts, sideSize * sideSize * sizeof(float3), cudaMemcpyDeviceToHost);
	cudaFree(dev_verts);
}


void launch_kernel(float3 *pos, unsigned int mesh_width, unsigned int mesh_height, float time)
{
	// execute the kernel
	// dim3 block(8, 8, 1);
	// dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	// simple_vbo_kernel << < grid, block >> >(pos, mesh_width, mesh_height, time);
}


void runCuda(struct cudaGraphicsResource **vbo_resource)
{
	// map OpenGL buffer object for writing from CUDA
	float3 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		*vbo_resource));
	//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	// execute the kernel
	//    dim3 block(8, 8, 1);
	//    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	//    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

	launch_kernel(dptr, mesh_width, mesh_height, g_Time);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
{
	assert(vbo);

	// TODO this GL calls need to be moved to the C# script

	// create buffer object
	// glGenBuffers(1, vbo);
	// glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = mesh_width * mesh_height * 3 * sizeof(float);
	// glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	// glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
}

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

	// glBindBuffer(1, *vbo);
	// glDeleteBuffers(1, vbo);

	*vbo = 0;
}

void cleanup()
{
	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
}