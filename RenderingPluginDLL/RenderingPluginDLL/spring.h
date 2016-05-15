#pragma once

#include <vector_types.h>

#define EPSILON 0.02f

struct Spring
{
	int point;
	float stiffness;
	float rest_length;
};