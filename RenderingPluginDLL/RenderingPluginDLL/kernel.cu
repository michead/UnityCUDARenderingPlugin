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

typedef void(*FuncPtr)(const char *);
FuncPtr Debug;

static int texSize;

static int* triangles;
static int triangleCount;
static int* devTArray;

static struct cudaGraphicsResource* cgr;
static void* texPtr;
static GLuint texID;

static struct cudaGraphicsResource* nCgr;
static void* nTexPtr;
static GLuint nTexID;

static float unityTime;

extern "C" void EXPORT_API SetTimeFromUnity(float t) { unityTime = t; }


__global__ void plugin_kernel(cudaSurfaceObject_t cso, cudaSurfaceObject_t nCso, int meshSize, int* triangles, int trCount, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float freq = 4.0f;
	float attenuation = 0.5f;

	float4 vert = surf2Dread<float4>(cso, (int)sizeof(float4)*x, y, cudaBoundaryModeZero);
	vert.y = sinf(vert.x * freq + time) * cosf(vert.z * freq + time) * attenuation;

	surf2Dwrite(vert, cso, (int)sizeof(float4) * x, y, cudaBoundaryModeZero);

	__syncthreads();

	int vertID = x * meshSize + y;

	float4 normal = { 0, 0, 0, 0 };

	for (int i = 0; i < trCount; i += 3)
	{
		if (triangles[i] == vertID || triangles[i + 1] == vertID || triangles[i + 2] == vertID)
		{
			int xx = triangles[i] / meshSize;
			int yy = triangles[i] % meshSize;

			float4 vertA = surf2Dread<float4>(cso, (int)sizeof(float4) *  xx, yy, cudaBoundaryModeZero);

			xx = triangles[i + 1] / meshSize;
			yy = triangles[i + 1] % meshSize;

			float4 vertB = surf2Dread<float4>(cso, (int)sizeof(float4) *  xx, yy, cudaBoundaryModeZero);

			xx = triangles[i + 2] / meshSize;
			yy = triangles[i + 2] % meshSize;

			float4 vertC = surf2Dread<float4>(cso, (int)sizeof(float4) *  xx, yy, cudaBoundaryModeZero);

			float3 vecA = { vertB.x - vertA.x, vertB.y - vertA.y, vertB.z - vertA.z };
			float3 vecB = { vertC.x - vertA.x, vertC.y - vertA.y, vertC.z - vertA.z };

			float3 cross = { vecA.y * vecB.z - vecA.z * vecB.y, vecA.z * vecB.x - vecA.x * vecB.z, vecA.x * vecB.y - vecA.y * vecB.x };
			float lenCross = sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);

			if (lenCross > 0.0f)
			{
				cross.x /= lenCross;
				cross.y /= lenCross;
				cross.z /= lenCross;
			}

			normal.x += cross.x;
			normal.y += cross.y;
			normal.z += cross.z;
		}

		float lenNormal = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);

		if (lenNormal > 0.0f)
		{
			normal.x /= lenNormal;
			normal.y /= lenNormal;
			normal.z /= lenNormal;
		}

		surf2Dwrite(normal, nCso, (int)sizeof(float4) * x, y, cudaBoundaryModeZero);
	}
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
	// Vertex data
	CheckPluginErrors(cudaGraphicsMapResources(1, &cgr), "Error encountered while mapping resource");

	cudaArray_t cudaArray;
	CheckPluginErrors(cudaGraphicsSubResourceGetMappedArray(&cudaArray, cgr, 0, 0), "Error encountered while mapping graphics resource to CUDA array.");

	cudaResourceDesc desc;
	desc.resType = cudaResourceTypeArray;
	desc.res.array.array = cudaArray;

	cudaSurfaceObject_t cso;
	CheckPluginErrors(cudaCreateSurfaceObject(&cso, &desc), "Error encountered while creating Surface Object.");

	// Normal data
	CheckPluginErrors(cudaGraphicsMapResources(1, &nCgr), "Error encountered while mapping resource");

	cudaArray_t nCudaArray;
	CheckPluginErrors(cudaGraphicsSubResourceGetMappedArray(&nCudaArray, nCgr, 0, 0), "Error encountered while mapping graphics resource to CUDA array.");

	cudaResourceDesc nDesc;
	nDesc.resType = cudaResourceTypeArray;
	nDesc.res.array.array = nCudaArray;

	cudaSurfaceObject_t nCso;
	CheckPluginErrors(cudaCreateSurfaceObject(&nCso, &nDesc), "Error encountered while creating Surface Object.");

	dim3 block(texSize, texSize, 1);
	dim3 grid(texSize / block.x, texSize / block.y, 1);
	plugin_kernel << < grid, block >> >(cso, nCso, texSize, devTArray, triangleCount, unityTime);

	CheckPluginErrors(cudaGetLastError(), "Error in kernel execution.");

	CheckPluginErrors(cudaDestroySurfaceObject(cso), "Error encountered while destroying Surface Object.");
	CheckPluginErrors(cudaGraphicsUnmapResources(1, &cgr), "Error encountered while unmapping resource.");

	CheckPluginErrors(cudaDestroySurfaceObject(nCso), "Error encountered while destroying Surface Object.");
	CheckPluginErrors(cudaGraphicsUnmapResources(1, &nCgr), "Error encountered while unmapping resource.");

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

extern "C" EXPORT_API void Init(int size, void* tPtr, void* nTPtr, int* tr, int trCount)
{
	texSize = size;

	triangles = tr;
	triangleCount = trCount;

	texPtr = tPtr;
	texID = (GLuint)(size_t)(texPtr);

	CheckPluginErrors(cudaGraphicsGLRegisterImage(&cgr, texID, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone), "Error encountered while registering resource.");

	nTexPtr = nTPtr;
	nTexID = (GLuint)(size_t)(nTexPtr);

	CheckPluginErrors(cudaGraphicsGLRegisterImage(&nCgr, nTexID, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone), "Error encountered while registering resource.");

	CheckPluginErrors(cudaMalloc((void**)&devTArray, trCount * sizeof(int)), "Error encountered while allocating memory on device");
	CheckPluginErrors(cudaMemcpy(devTArray, triangles, trCount * sizeof(int), cudaMemcpyHostToDevice), "Error encountered while copying array to device");
}

extern "C" EXPORT_API void Cleanup()
{
	CheckPluginErrors(cudaGraphicsUnregisterResource(cgr), "Error encountered while unregistering resource.");
	CheckPluginErrors(cudaGraphicsUnregisterResource(nCgr), "Error encountered while unregistering resource.");

	CheckPluginErrors(cudaFree(devTArray), "Error encountered while freeing memory on device");
}