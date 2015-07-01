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

#define FREQ 4.0f
#define BLOCK_DIM_X 10
#define BLOCK_DIM_Y 10

typedef void(*FuncPtr)(const char *);
FuncPtr Debug;

float3* devVerts;
unsigned int meshSize;

static void* texPtr;

void UpdateVertsInTex()
{

}

extern "C" void EXPORT_API SetTextureFromUnity(void* texturePtr)
{
	texPtr = texturePtr;
}

extern "C" void EXPORT_API UnityRenderEvent(int eventID)
{
	UpdateVertsInTex();
}

extern "C" EXPORT_API void SetDebugFunction(FuncPtr fp)
{
	Debug = fp;
}

extern "C" EXPORT_API void ComputeSineWave(float3* verts, float time)
{
	for (int i = 0; i < meshSize * meshSize; i++)
		verts[i].y = sin(verts[i].x * FREQ + time) * cos(verts[i].z * FREQ + time) * 0.2f;
}

__global__ void simple_vbo_kernel(float3 *pos, unsigned int sideSize, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float freq = 4.0f;
	int index = y * sideSize + x;

	pos[index].y = sinf(pos[index].x * freq + time) * cosf(pos[index].z * freq + time) * 0.2f;
}

extern "C" EXPORT_API void ParallelComputeSineWave(float3* verts, float time)
{
	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
	dim3 grid(meshSize / block.x, meshSize / block.y, 1);
	simple_vbo_kernel << < grid, block >> >(devVerts, meshSize, time);

	cudaMemcpy(verts, devVerts, meshSize * meshSize * sizeof(float3), cudaMemcpyDeviceToHost);
}

extern "C" EXPORT_API void Init(float3* verts, unsigned int size)
{
	meshSize = size;

	cudaMalloc((void**)&devVerts, meshSize * meshSize * sizeof(float3));
	cudaMemcpy(devVerts, verts, meshSize * meshSize * sizeof(float3), cudaMemcpyHostToDevice);
}

extern "C" EXPORT_API void Cleanup()
{
	cudaFree(devVerts);
}