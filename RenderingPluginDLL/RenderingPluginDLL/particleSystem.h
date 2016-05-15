#pragma once

#include <vector_types.h>

using namespace std;

class ParticleSystem
{
public:
	ParticleSystem(int numParticles = 0);
	
	int numParticles;

	virtual float3* evalF(float3* state) = 0;

	float3* state;

	unsigned faceCount, vertCount, stateCount;
	unsigned* verts;
};