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
static struct cudaGraphicsResource* cgr;
GLuint texID;
struct cudaArray* cudaArray;
texture<float4, cudaTextureType2D, cudaReadModeElementType> tex;

static float unityTime;

extern "C" void EXPORT_API SetTimeFromUnity(float t) { unityTime = t; }

__global__ void simple_vbo_kernel(cudaSurfaceObject_t cso, dim3 dimension, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float freq = 4.0f;
	int index = y * dimension.x + x;

	float4 vert = surf2Dread<float4>(cso, (int)sizeof(float4)*x, y, cudaBoundaryModeClamp);
	vert.y = sinf(vert.x * freq + time) * cosf(vert.z * freq + time) * 0.2f;

	surf2Dwrite(vert, cso, (int)sizeof(float4)*x, y, cudaBoundaryModeClamp);
}

__global__ void simple_vbo_kernel_2(float3 *pos, unsigned int sideSize, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float freq = 4.0f;
	int index = y * sideSize + x;

	pos[index].y = sinf(pos[index].x * freq + time) * cosf(pos[index].z * freq + time) * 0.2f;
}

void UpdateVertsInTex()
{
	struct cudaResourceDesc description;
	memset(&description, 0, sizeof(description));
	description.resType = cudaResourceTypeArray;
	description.res.array.array = cudaArray;

	cudaSurfaceObject_t cso;
	checkCudaErrors((cudaCreateSurfaceObject(&cso, &description)));

	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
	dim3 grid(meshSize / block.x, meshSize / block.y, 1);
	simple_vbo_kernel << < grid, block >> >(cso, meshSize, unityTime);
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

extern "C" EXPORT_API void ParallelComputeSineWave(float3* verts, float time)
{
	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
	dim3 grid(meshSize / block.x, meshSize / block.y, 1);
	simple_vbo_kernel_2 << < grid, block >> >(devVerts, meshSize, time);

	cudaMemcpy(verts, devVerts, meshSize * meshSize * sizeof(float3), cudaMemcpyDeviceToHost);
}

extern "C" EXPORT_API void Init(float3* verts, unsigned int size, void* texturePtr, int textureID)
{
	meshSize = size;
	texPtr = texturePtr;
	texID = textureID;

	checkCudaErrors(cudaGraphicsGLRegisterImage(&cgr, texID, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
	checkCudaErrors(cudaGraphicsMapResources(1, &cgr));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cudaArray, cgr, 0, 0));
	if(!cudaBindTextureToArray(tex, cudaArray)) Debug("Error encountered while binding texture to array.");
}

extern "C" EXPORT_API void Cleanup()
{

	checkCudaErrors(cudaUnbindTexture(tex));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr, 0));
	cudaGLUnregisterBufferObject(texID);
	cudaGraphicsUnregisterResource(cgr);
}