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
#include <cutil_math.h>

#include <vector_types.h>
#include <vector>

#include <stdio.h>
#include <sstream>
#include <math.h>

#include <Windows.h>

#define WINDOWS_MEAN_AND_LEAN
#define NOMINAX
#define EXPORT_API __declspec(dllexport)

#define MAX_BLOCK_SIZE_X 32
#define MAX_BLOCK_SIZE_Y 32

typedef void(*FuncPtr)(const char *);
FuncPtr Debug;

static int texSize;

static int* triangles;
static int triangleCount;
static int* faceIndex;
static int* faceOffset;
static int faceCount;
static int* devTArray;
static float3* devFNArray;
static int* devFIArray;
static int* devFOArray;

static float3* faceNormals;

static cudaGraphicsResource_t* resources;
static cudaSurfaceObject_t cso;
static cudaSurfaceObject_t nCso;

static dim3 block, grid, blockF, gridF;

static struct cudaGraphicsResource* cgr;
static void* texPtr;
static GLuint texID;

static struct cudaGraphicsResource* nCgr;
static void* nTexPtr;
static GLuint nTexID;

static float unityTime;

extern "C" void EXPORT_API SetTimeFromUnity(float t) { unityTime = t; }


__global__ void vertex_kernel(cudaSurfaceObject_t cso, int meshSize, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float freq = 4.0f;
	float attenuation = 0.5f;

	float4 vert = surf2Dread<float4>(cso, (int)sizeof(float4)*x, y, cudaBoundaryModeZero);
	vert.y = sinf(vert.x * freq + time) * cosf(vert.z * freq + time) * attenuation;

	surf2Dwrite(vert, cso, (int)sizeof(float4) * x, y, cudaBoundaryModeZero);
}

__global__ void face_kernel(cudaSurfaceObject_t cso, int* triangles, float3* normals, int meshSize)
{
	unsigned int i = (blockIdx.x*blockDim.x + threadIdx.x) * 3;

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

	float3 cross = make_float3(vecA.y*vecB.z - vecA.z*vecB.y, vecA.z*vecB.x - vecA.x*vecB.z, vecA.x*vecB.y - vecA.y*vecB.x);

	normals[i / 3] = cross;
}

__global__ void normal_kernel(cudaSurfaceObject_t nCso, float3* normals, int meshSize, int* faceIndex, int* faceOffset)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int vertID = x * meshSize + y;

	float4 normal = { 0 };

	for (int i = faceOffset[vertID]; i < faceOffset[vertID + 1]; i++)
	{
		float3 fNormal = normals[faceIndex[i]];

		normal.x += fNormal.x;
		normal.y += fNormal.y;
		normal.z += fNormal.z;
	}

	surf2Dwrite(normalize(normal), nCso, (int)sizeof(float4) * x, y, cudaBoundaryModeZero);
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
	vertex_kernel << < grid, block >> >(cso, texSize, unityTime);
	CheckPluginErrors(cudaGetLastError(), "Error in vertex_kernel execution.");
	CheckPluginErrors(cudaDeviceSynchronize(), "Error in device synchronization.");

	face_kernel << < gridF, blockF >> >(cso, devTArray, devFNArray, texSize);
	CheckPluginErrors(cudaGetLastError(), "Error in face_kernel execution.");
	CheckPluginErrors(cudaDeviceSynchronize(), "Error in device synchronization.");

	normal_kernel << < grid, block >> >(nCso, devFNArray, texSize, devFIArray, devFOArray);
	CheckPluginErrors(cudaGetLastError(), "Error in normal_kernel execution.");
	CheckPluginErrors(cudaDeviceSynchronize(), "Error in device synchronization.");
}

extern "C" void EXPORT_API UnityRenderEvent(int eventID)
{
	UpdateVertsInTex();
}

extern "C" EXPORT_API void SetDebugFunction(FuncPtr fp)
{
	Debug = fp;
}

void InitializeFaceIndex()
{
	faceNormals = (float3*)malloc((triangleCount / 3) * sizeof(float3));

	faceOffset = (int*)malloc((texSize * texSize + 1) * sizeof(int));
	faceCount = 0;

	for (int i = 0; i < texSize * texSize; i++)
	{
		faceOffset[i] = faceCount;

		for (int j = 0; j < triangleCount; j += 3)
		{
			if (triangles[j] == i || triangles[j + 1] == i || triangles[j + 2] == i)
				faceCount++;
		}
	}

	faceOffset[texSize * texSize] = texSize * texSize;

	faceIndex = (int*)malloc(faceCount * sizeof(int));

	int k = 0;

	for (int i = 0; i < texSize * texSize; i++)
	{
		for (int j = 0; j < triangleCount; j += 3)
		{
			if (triangles[j] == i || triangles[j + 1] == i || triangles[j + 2] == i)
			{
				faceIndex[k] = j / 3;
				k++;
			}
		}
	}
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

	resources = (cudaGraphicsResource_t*)malloc(sizeof(cudaGraphicsResource_t) * 2);
	resources[0] = cgr;
	resources[1] = nCgr;
	CheckPluginErrors(cudaGraphicsMapResources(2, resources), "Error encountered while mapping resources");

	cudaArray_t cudaArray;
	CheckPluginErrors(cudaGraphicsSubResourceGetMappedArray(&cudaArray, cgr, 0, 0), "Error encountered while mapping graphics resource to CUDA array.");

	cudaResourceDesc desc;
	desc.resType = cudaResourceTypeArray;
	desc.res.array.array = cudaArray;

	CheckPluginErrors(cudaCreateSurfaceObject(&cso, &desc), "Error encountered while creating Surface Object.");

	cudaArray_t nCudaArray;
	CheckPluginErrors(cudaGraphicsSubResourceGetMappedArray(&nCudaArray, nCgr, 0, 0), "Error encountered while mapping graphics resource to CUDA array.");

	cudaResourceDesc nDesc;
	nDesc.resType = cudaResourceTypeArray;
	nDesc.res.array.array = nCudaArray;

	CheckPluginErrors(cudaCreateSurfaceObject(&nCso, &nDesc), "Error encountered while creating Surface Object.");

	block = dim3(MAX_BLOCK_SIZE_X, MAX_BLOCK_SIZE_Y, 1);
	grid = dim3(texSize / block.x, texSize / block.y, 1);

	blockF = dim3(MAX_BLOCK_SIZE_X, 1, 1);
	gridF = dim3(trCount / (blockF.x * 3), 1, 1);

	InitializeFaceIndex();

	CheckPluginErrors(cudaMalloc((void**)&devFIArray, faceCount * sizeof(int)), "Error encountered while allocating memory on device");
	CheckPluginErrors(cudaMemcpy(devFIArray, faceIndex, faceCount * sizeof(int), cudaMemcpyHostToDevice), "Error encountered while copying array to device");

	CheckPluginErrors(cudaMalloc((void**)&devFOArray, (texSize * texSize + 1) * sizeof(int)), "Error encountered while allocating memory on device");
	CheckPluginErrors(cudaMemcpy(devFOArray, faceOffset, (texSize * texSize + 1) * sizeof(int), cudaMemcpyHostToDevice), "Error encountered while copying array to device");

	CheckPluginErrors(cudaMalloc((void**)&devFNArray, (triangleCount / 3) * sizeof(float3)), "Error encountered while allocating memory on device");
}

extern "C" EXPORT_API void Cleanup()
{
	free(faceIndex);
	free(faceOffset);
	free(faceNormals);

	CheckPluginErrors(cudaDestroySurfaceObject(cso), "Error encountered while destroying Surface Object.");
	CheckPluginErrors(cudaDestroySurfaceObject(nCso), "Error encountered while destroying Surface Object.");
	CheckPluginErrors(cudaGraphicsUnmapResources(2, resources), "Error encountered while unmapping resources.");
	free(resources);

	CheckPluginErrors(cudaGraphicsUnregisterResource(cgr), "Error encountered while unregistering resource.");
	CheckPluginErrors(cudaGraphicsUnregisterResource(nCgr), "Error encountered while unregistering resource.");

	CheckPluginErrors(cudaFree(devTArray), "Error encountered while freeing memory on device");
	CheckPluginErrors(cudaFree(devFIArray), "Error encountered while freeing memory on device");
	CheckPluginErrors(cudaFree(devFOArray), "Error encountered while freeing memory on device");
	CheckPluginErrors(cudaFree(devFNArray), "Error encountered while freeing memory on device");
}