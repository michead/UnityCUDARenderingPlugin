#pragma once

#include <vector>
#include "particleSystem.h"
#include "spring.h"

using namespace std;

class SpringSystem : public ParticleSystem
{
public:
	int* faces;

	float shearRLen;
	float flexRLen;

	vector<vector<Spring>> springs;

	virtual void populateSprings() = 0;
	virtual void computeStructuralSprings() = 0;
	virtual void computeShearSprings() = 0;
	virtual void computeFlexSprings() = 0;
};