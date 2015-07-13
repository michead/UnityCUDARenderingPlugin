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
#define BLOCK_DIM_X 11
#define BLOCK_DIM_Y 11

typedef void(*FuncPtr)(const char *);
FuncPtr Debug;

float3* devVerts;
unsigned int meshSize;

static struct cudaGraphicsResource* cgr;
static void* texPtr;
static GLuint texID;

static float unityTime;

extern "C" void EXPORT_API SetTimeFromUnity(float t) { unityTime = t; }


__global__ void simple_vbo_kernel(cudaSurfaceObject_t cso, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float freq = 4.0f;

	float4 vert = surf2Dread<float4>(cso, (int)sizeof(float4)*x, y, cudaBoundaryModeZero);
	vert.y = sinf(vert.x * freq + time) * cosf(vert.z * freq + time);

	surf2Dwrite(vert, cso, (int)sizeof(float4)*x, y, cudaBoundaryModeZero);
}

void CheckPluginErrors(cudaError err, const char* context)
{
	if (err != cudaSuccess)
	{
		const char* errName = cudaGetErrorName(err);
		const char* errString = cudaGetErrorString(err);

		char* errMessage = (char*)calloc(strlen(errName) + strlen(errString) + 8, sizeof(char));
		strcpy(errMessage, context);
		strcat(errMessage, " --> ");
		strcat(errMessage, errName);
		strcat(errMessage, ": ");
		strcat(errMessage, errString);

		Debug(errMessage);
	}
}

void UpdateVertsInTex()
{
	CheckPluginErrors(cudaGraphicsMapResources(1, &cgr), "Error encountered while mapping resource");

	cudaArray_t cudaArray;
	CheckPluginErrors(cudaGraphicsSubResourceGetMappedArray(&cudaArray, cgr, 0, 0), "Error encountered while mapping graphics resource to CUDA array.");

	cudaResourceDesc desc;
	desc.resType = cudaResourceTypeArray;
	desc.res.array.array = cudaArray;

	cudaSurfaceObject_t cso;
	CheckPluginErrors(cudaCreateSurfaceObject(&cso, &desc), "Error encountered while creating Surface Object.");

	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
	dim3 grid(meshSize / block.x, meshSize / block.y, 1);
	simple_vbo_kernel << < grid, block >> >(cso, unityTime);

	CheckPluginErrors(cudaGetLastError(), "Error in kernel execution.");
	CheckPluginErrors(cudaDestroySurfaceObject(cso), "Error encountered while destroying Surface Object.");
	CheckPluginErrors(cudaGraphicsUnmapResources(1, &cgr), "Error encountered while unmapping resource.");
	CheckPluginErrors(cudaStreamSynchronize(0), "Error in stream synchronization.");
}

extern "C" void EXPORT_API UnityRenderEvent(int eventID)
{
	UpdateVertsInTex();
}

extern "C" EXPORT_API void SetDebugFunction(FuncPtr fp)
{
	Debug = fp;
}

extern "C" EXPORT_API void Init(float3* verts, unsigned int size, void* tPtr)
{
	meshSize = size;
	texPtr = tPtr;
	texID = (GLuint)(size_t)(texPtr);

	CheckPluginErrors(cudaGraphicsGLRegisterImage(&cgr, texID, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone), "Error encountered while registering resource.");
}

extern "C" EXPORT_API void Cleanup()
{
	CheckPluginErrors(cudaGraphicsUnregisterResource(cgr), "Error encountered while unregistering resource.");
}