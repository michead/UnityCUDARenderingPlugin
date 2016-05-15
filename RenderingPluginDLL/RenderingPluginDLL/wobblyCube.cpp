#include "wobblyCube.h"

WobblyCube::WobblyCube()
{
	numParticles = CUBE_WIDTH * CUBE_HEIGHT * CUBE_WIDTH;

	shearRLen = REST_LENGTH * sqrt(2.f);
	flexRLen = 2 * REST_LENGTH;

	state = (float3*)malloc(CUBE_WIDTH * CUBE_LENGTH * CUBE_HEIGHT * sizeof(float3) * 2);
	faces = (int*)malloc(4 * 
						((CUBE_LENGTH - 1) * (CUBE_HEIGHT - 1) + 
						(CUBE_WIDTH - 1) * (CUBE_HEIGHT - 1) + 
						(CUBE_LENGTH - 1) * (CUBE_WIDTH - 1)) * 
						sizeof(int) * 
						3);
	
	vertCount = 0;
	stateCount = 0;
	for (unsigned k = 0; k < CUBE_WIDTH; k++)
	{
		for (unsigned i = 0; i < CUBE_LENGTH; i++)
		{
			for (unsigned j = 0; j < CUBE_HEIGHT; j++)
			{
				if (k == 0 || 
					k == CUBE_WIDTH - 1 || 
					i == 0 || 
					i == CUBE_LENGTH - 1 || 
					j == 0 || 
					j == CUBE_HEIGHT - 1) vertCount++;
			}
		}
	}

	verts = (unsigned*)malloc(vertCount * sizeof(unsigned));

	unsigned vCount = 0;
	for (unsigned k = 0; k < CUBE_WIDTH; k++)
	{
		for (unsigned i = 0; i < CUBE_LENGTH; i++)
		{
			for (unsigned j = 0; j < CUBE_HEIGHT; j++)
			{
				if (k == 0 ||
					k == CUBE_WIDTH - 1 ||
					i == 0 ||
					i == CUBE_LENGTH - 1 ||
					j == 0 ||
					j == CUBE_HEIGHT - 1)
				{
					verts[vCount++] = stateCount;
				}

				float3 pos = { i * SPACING, j * SPACING, k * SPACING };

				state[stateCount++] = pos;
				state[stateCount++] = float3();
			}
		}
	}

	faceCount = 0;
	computeFaces();
	populateSprings();
}

WobblyCube::~WobblyCube()
{
	free(state);
	free(faces);
}

void WobblyCube::populateSprings()
{
	computeStructuralSprings();
	computeShearSprings();
	computeFlexSprings();
}

void WobblyCube::computeFlexSprings()
{
	for (unsigned k = 0; k < CUBE_WIDTH; k++)
	{
		for (unsigned i = 0; i < CUBE_LENGTH; i++)
		{
			for (unsigned j = 0; j < CUBE_HEIGHT; j++)
			{

				vector<Spring> v;

				// LEFT
				if (i > 1)
				{
					Spring spring = { indexOf(i - 2, j, k), SPRING_COSTANT, flexRLen };
					v.push_back(spring);
				}

				// RIGHT
				if (i < CUBE_LENGTH - 2)
				{
					Spring spring = { indexOf(i + 2, j, k), SPRING_COSTANT, flexRLen };
					v.push_back(spring);
				}

				// DOWN
				if (j > 1)
				{
					Spring spring = { indexOf(i, j - 2, k), SPRING_COSTANT, flexRLen };
					v.push_back(spring);
				}

				// UP
				if (j < CUBE_HEIGHT - 2)
				{
					Spring spring = { indexOf(i, j + 2, k), SPRING_COSTANT, flexRLen };
					v.push_back(spring);
				}

				// FORTH
				if (k < CUBE_WIDTH - 2)
				{
					Spring spring = { indexOf(i, j, k + 2), SPRING_COSTANT, flexRLen };
					v.push_back(spring);
				}

				// BACK
				if (k > 1)
				{
					Spring spring = { indexOf(i, j, k - 2), SPRING_COSTANT, flexRLen };
					v.push_back(spring);
				}

				springs.push_back(v);
			}
		}
	}
}

void WobblyCube::computeShearSprings()
{
	for (unsigned k = 0; k < CUBE_WIDTH; k++)
	{
		for (unsigned i = 0; i < CUBE_LENGTH; i++)
		{
			for (unsigned j = 0; j < CUBE_HEIGHT; j++)
			{

				vector<Spring> v;

				// K
				// BOTTOM LEFT
				if (i > 0 && j > 0)
				{
					Spring spring = { indexOf(i - 1, j - 1, k), SPRING_COSTANT, shearRLen };
					springs[indexOf(i, j, k) * 0.5].push_back(spring);
				}

				// BOTTOM RIGHT
				if (i < CUBE_LENGTH - 1 && j > 0)
				{
					Spring spring = { indexOf(i + 1, j - 1, k), SPRING_COSTANT, shearRLen };
					springs[indexOf(i, j, k) * 0.5].push_back(spring);
				}

				// TOP LEFT
				if (i > 0 && j < CUBE_HEIGHT - 1)
				{
					Spring spring = { indexOf(i - 1, j + 1, k), SPRING_COSTANT, shearRLen };
					springs[indexOf(i, j, k) * 0.5].push_back(spring);
				}

				// TOP RIGHT
				if (i < CUBE_LENGTH - 1 && j < CUBE_HEIGHT - 1)
				{
					Spring spring = { indexOf(i + 1, j + 1, k), SPRING_COSTANT, shearRLen };
					springs[indexOf(i, j, k) * 0.5].push_back(spring);
				}

				// I
				// BOTTOM LEFT
				if (k > 0 && j > 0)
				{
					Spring spring = { indexOf(i, j - 1, k - 1), SPRING_COSTANT, shearRLen };
					springs[indexOf(i, j, k) * 0.5].push_back(spring);
				}

				// BOTTOM RIGHT
				if (k < CUBE_WIDTH - 1 && j > 0)
				{
					Spring spring = { indexOf(i, j - 1, k + 1), SPRING_COSTANT, shearRLen };
					springs[indexOf(i, j, k) * 0.5].push_back(spring);
				}

				// TOP LEFT
				if (k > 0 && j < CUBE_HEIGHT - 1)
				{
					Spring spring = { indexOf(i, j + 1, k - 1), SPRING_COSTANT, shearRLen };
					springs[indexOf(i, j, k) * 0.5].push_back(spring);
				}

				// TOP RIGHT
				if (k < CUBE_WIDTH - 1 && j < CUBE_HEIGHT - 1)
				{
					Spring spring = { indexOf(i, j + 1, k + 1), SPRING_COSTANT, shearRLen };
					springs[indexOf(i, j, k) * 0.5].push_back(spring);
				}

				// J
				// BOTTOM LEFT
				if (k > 0 && i > 0)
				{
					Spring spring = { indexOf(i - 1, j, k - 1), SPRING_COSTANT, shearRLen };
					springs[indexOf(i, j, k) * 0.5].push_back(spring);
				}

				// BOTTOM RIGHT
				if (k < CUBE_WIDTH - 1 && i > 0)
				{
					Spring spring = { indexOf(i - 1, j, k + 1), SPRING_COSTANT, shearRLen };
					springs[indexOf(i, j, k) * 0.5].push_back(spring);
				}

				// TOP LEFT
				if (k > 0 && i < CUBE_HEIGHT - 1)
				{
					Spring spring = { indexOf(i + 1, j, k - 1), SPRING_COSTANT, shearRLen };
					springs[indexOf(i, j, k) * 0.5].push_back(spring);
				}

				// TOP RIGHT
				if (k < CUBE_WIDTH - 1 && i < CUBE_HEIGHT - 1)
				{
					Spring spring = { indexOf(i + 1, j, k + 1), SPRING_COSTANT, shearRLen };
					springs[indexOf(i, j, k) * 0.5].push_back(spring);
				}
			}
		}
	}
}

void WobblyCube::computeStructuralSprings()
{
	for (unsigned k = 0; k < CUBE_WIDTH; k++)
	{
		for (unsigned i = 0; i < CUBE_LENGTH; i++)
		{
			for (unsigned j = 0; j < CUBE_HEIGHT; j++)
			{

				vector<Spring> v;

				// LEFT
				if (i > 0)
				{
					Spring spring = { indexOf(i - 1, j, k), SPRING_COSTANT, REST_LENGTH };
					v.push_back(spring);
				}

				// RIGHT
				if (i < CUBE_LENGTH - 1)
				{
					Spring spring = { indexOf(i + 1, j, k), SPRING_COSTANT, REST_LENGTH };
					v.push_back(spring);
				}

				// DOWN
				if (j > 0)
				{
					Spring spring = { indexOf(i, j - 1, k), SPRING_COSTANT, REST_LENGTH };
					v.push_back(spring);
				}

				// UP
				if (j < CUBE_HEIGHT - 1)
				{
					Spring spring = { indexOf(i, j + 1, k), SPRING_COSTANT, REST_LENGTH };
					v.push_back(spring);
				}

				// FORTH
				if (k < CUBE_WIDTH - 1)
				{
					Spring spring = { indexOf(i, j, k + 1), SPRING_COSTANT, REST_LENGTH };
					v.push_back(spring);
				}

				// BACK
				if (k > 0)
				{
					Spring spring = { indexOf(i, j, k - 1), SPRING_COSTANT, REST_LENGTH };
					v.push_back(spring);
				}

				springs.push_back(v);
			}
		}
	}
}

float3* WobblyCube::evalF(float3* state)
{
	float3* f = (float3*)malloc(stateCount * sizeof(float3));

	for (unsigned i = 0; i < stateCount; i += 2)
	{
		if ((i / 2) % CUBE_HEIGHT == 0)
		{
			f[i] = state[i + 1];
			f[i + 1] = {0, 0, 0};

			continue;
		}

		vector<Spring> spring = springs[i / 2];
		float3 forces;

		float3 gravity = { 0, -9.81f * MASS, 0 };
		float3 viscous_drag = { (-DAMPING) * state[i + 1].x, (-DAMPING) * state[i + 1].y, (-DAMPING) * state[i + 1].z };

		forces = { gravity.x + viscous_drag.x, gravity.y + viscous_drag.y, gravity.z + viscous_drag.z };

		int sSize = spring.size();

		for (unsigned j = 0; j < sSize; j++)
		{
			float3 dist = { state[i].x - state[spring[j].point].x, state[i].y - state[spring[j].point].y, state[i].z - state[spring[j].point].z };
			float dist_abs = sqrt(dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
			float3 dist_norm = { dist.x / dist_abs, dist.y / dist_abs, dist.z / dist_abs };
			float3 spring_force = {
				(-spring[j].stiffness) * (dist_abs - spring[j].rest_length) * dist_norm.x,
				(-spring[j].stiffness) * (dist_abs - spring[j].rest_length) * dist_norm.y,
				(-spring[j].stiffness) * (dist_abs - spring[j].rest_length) * dist_norm.z };

			forces = { forces.x + spring_force.x, forces.y + spring_force.y, forces.z + spring_force.z };
		}

		f[i] = state[i + 1];
		f[i + 1] = forces;
	}

	return f;
}

unsigned WobblyCube::indexOf(unsigned l, unsigned h, unsigned w)
{
	return (w * CUBE_LENGTH * CUBE_HEIGHT + l * CUBE_HEIGHT + h) * 2;
}

void WobblyCube::computeFaces()
{
	// BACK FACE
	for (unsigned i = 0; i < CUBE_LENGTH - 1; i++)
	{
		for (unsigned j = 0; j < CUBE_HEIGHT - 1; j++)
		{
			faces[faceCount++] = indexOf(i, j, 0);
			faces[faceCount++] = indexOf(i, j + 1, 0);
			faces[faceCount++] = indexOf(i + 1, j, 0);

			faces[faceCount++] = indexOf(i, j + 1, 0);
			faces[faceCount++] = indexOf(i + 1, j + 1, 0);
			faces[faceCount++] = indexOf(i + 1, j, 0);
		}
	}

	// FRONT FACE
	for (unsigned i = 0; i < CUBE_LENGTH - 1; i++)
	{
		for (unsigned j = 0; j < CUBE_HEIGHT - 1; j++)
		{
			faces[faceCount++] = indexOf(i, j, CUBE_WIDTH - 1);
			faces[faceCount++] = indexOf(i + 1, j, CUBE_WIDTH - 1);
			faces[faceCount++] = indexOf(i, j + 1, CUBE_WIDTH - 1);

			faces[faceCount++] = indexOf(i + 1, j, CUBE_WIDTH - 1);
			faces[faceCount++] = indexOf(i + 1, j + 1, CUBE_WIDTH - 1);
			faces[faceCount++] = indexOf(i, j + 1, CUBE_WIDTH - 1);
		}
	}

	// LEFT FACE
	for (unsigned k = 0; k < CUBE_WIDTH - 1; k++)
	{
		for (unsigned j = 0; j < CUBE_HEIGHT - 1; j++)
		{
			faces[faceCount++] = indexOf(0, j, k);
			faces[faceCount++] = indexOf(0, j, k + 1);
			faces[faceCount++] = indexOf(0, j + 1, k);

			faces[faceCount++] = indexOf(0, j, k + 1);
			faces[faceCount++] = indexOf(0, j + 1, k + 1);
			faces[faceCount++] = indexOf(0, j + 1, k);
		}
	}

	// RIGHT FACE
	for (unsigned k = 0; k < CUBE_WIDTH - 1; k++)
	{
		for (unsigned j = 0; j < CUBE_HEIGHT - 1; j++)
		{
			faces[faceCount++] = indexOf(CUBE_LENGTH - 1, j, k);
			faces[faceCount++] = indexOf(CUBE_LENGTH - 1, j + 1, k);
			faces[faceCount++] = indexOf(CUBE_LENGTH - 1, j, k + 1);

			faces[faceCount++] = indexOf(CUBE_LENGTH - 1, j + 1, k);
			faces[faceCount++] = indexOf(CUBE_LENGTH - 1, j + 1, k + 1);
			faces[faceCount++] = indexOf(CUBE_LENGTH - 1, j, k + 1);
		}
	}

	// BOTTOM FACE
	for (unsigned i = 0; i < CUBE_LENGTH - 1; i++)
	{
		for (unsigned k = 0; k < CUBE_WIDTH - 1; k++)
		{
			faces[faceCount++] = indexOf(i, 0, k);
			faces[faceCount++] = indexOf(i + 1, 0, k);
			faces[faceCount++] = indexOf(i, 0, k + 1);

			faces[faceCount++] = indexOf(i + 1, 0, k);
			faces[faceCount++] = indexOf(i + 1, 0, k + 1);
			faces[faceCount++] = indexOf(i, 0, k + 1);
		}
	}

	// UPPER FACE
	for (unsigned i = 0; i < CUBE_LENGTH - 1; i++)
	{
		for (unsigned k = 0; k < CUBE_WIDTH - 1; k++)
		{
			faces[faceCount++] = indexOf(i, CUBE_HEIGHT - 1, k);
			faces[faceCount++] = indexOf(i, CUBE_HEIGHT - 1, k + 1);
			faces[faceCount++] = indexOf(i + 1, CUBE_HEIGHT - 1, k);

			faces[faceCount++] = indexOf(i, CUBE_HEIGHT - 1, k + 1);
			faces[faceCount++] = indexOf(i + 1, CUBE_HEIGHT - 1, k + 1);
			faces[faceCount++] = indexOf(i + 1, CUBE_HEIGHT - 1, k);
		}
	}
}